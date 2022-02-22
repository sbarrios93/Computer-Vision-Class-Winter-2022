import numpy as np
from numpy.linalg import inv, svd, det, norm

from render import as_homogeneous, homogenize

import torch
import torch.optim as optim
from lietorch import SE3
from scipy.spatial.transform import Rotation as rot
from vis import vis_3d, o3d_pc, draw_camera


def skew(xs):

    """
    xs: [n, 3]
    Create a skew-symmetric matrix from a vector.
    |---|      |------------|
    | A |      | 0   -C   B |
    | B |  ->  | C    0  -A |
    | C |      | -B   A   0 |
    |---|      |------------|
    """
    assert xs.shape[-1] == 3, "xs must be of shape [n, 3]"
    # unpack column vector
    x0, x1, x2 = (xs[..., i] for i in range(xs.shape[-1]))

    zeros = np.zeros_like(x0)
    skewed = np.stack([zeros, -x2, x1, x2, zeros, -x0, -x1, x0, zeros])
    shape_ = xs.shape[:-1] + (3, 3)

    return skewed.reshape(*shape_)


def project(pts_3d, K, pose):
    P = K @ inv(pose)
    x1 = homogenize(pts_3d @ P.T)
    return x1


def pose_to_se3_embed(pose):
    R, t = pose[:3, :3], pose[:3, -1]
    tau = t
    phi = rot.from_matrix(R).as_quat()  # convert to quaternion
    embed = np.concatenate([tau, phi], axis=0)
    embed = torch.as_tensor(embed)
    return embed


def as_torch_tensor(*args):
    return [torch.as_tensor(elem) for elem in args]


def torch_project(pts_3d, K, se3_pose):
    P = K @ se3_pose.inv().matrix()
    x1 = pts_3d @ P.T
    x1 = homogenize(x1)
    return x1


def bundle_adjustment(x1s, x2s, full_K, p1, p2, pred_pts):
    embed1 = pose_to_se3_embed(p1)
    embed2 = pose_to_se3_embed(p2)

    embed1.requires_grad_(True)
    embed2.requires_grad_(True)

    x1s, x2s, full_K, pred_pts = as_torch_tensor(x1s, x2s, full_K, pred_pts)

    pred_pts.requires_grad_(True)

    lr = 1e-3
    # optimizer = optim.SGD([embed1, embed2, pred_pts], lr=lr, momentum=0.9)
    optimizer = optim.Adam([embed1, embed2, pred_pts], lr=lr)

    n_steps = 10000
    for i in range(n_steps):
        optimizer.zero_grad()

        p1 = SE3.InitFromVec(embed1)
        p2 = SE3.InitFromVec(embed2)

        x1_hat = torch_project(pred_pts, full_K, p1)
        x2_hat = torch_project(pred_pts, full_K, p2)
        err1 = torch.norm((x1_hat - x1s), dim=1)
        err1 = err1.mean()
        err2 = torch.norm((x2_hat - x2s), dim=1)
        err2 = err2.mean()
        err = (err1 + err2) / 2

        err.backward()
        optimizer.step()

        if (i % (n_steps // 10)) == 0:
            print(f"step {i}, err: {err.item()}")

    p1 = SE3.InitFromVec(embed1).matrix().detach().numpy()
    p2 = SE3.InitFromVec(embed2).matrix().detach().numpy()
    pred_pts = pred_pts.detach().numpy()
    return p1, p2, pred_pts


def eight_point_algorithm(x1s, x2s):
    # estimate the fundamental matrix
    # your code here
    assert x1s.shape == x2s.shape, "x1s and x2s must have the same shape"
    x1s = x1s.reshape(-1, 3)
    x2s = x2s.reshape(-1, 3)

    A = np.einsum("ki,kj->kji", x1s, x2s).reshape(-1, 9)

    U, s, V_t = svd(A.T @ A)

    # cond number is the ratio of top rank / last rank singular values
    # when you solve Ax = b, you take s[0] / s[-1]. But in vision,
    # we convert the problem above into the form Ax = 0. The nullspace is reserved for the solution.
    # This is called a "homogeneous system of equations" in linear algebra. This might be the reason
    # why homogeneous coordiantes are called homogeneous.
    # hence s[0] / s[-2].
    cond = s[0] / s[-2]
    print(f"condition number {cond}")

    F = V_t[-1, :].reshape(3, 3)

    F = enforce_rank_2(F)
    return F


def enforce_rank_2(F):
    # your code here
    U, s, V_t = svd(F)

    # set smallest singular value to 0
    s[-1] = 0

    F = U @ np.diag(s) @ V_t
    return F


def normalized_eight_point_algorithm(x1s, x2s, img_w, img_h):
    # your code here
    assert x1s.shape == x2s.shape, "x1s and x2s must have the same shape"
    T_x1 = np.array(
        [
            [2 / img_w, 0, -1],
            [0, 2 / img_h, -1],
            [0, 0, 1],
        ]
    )

    T_x2 = np.array(
        [
            [2 / img_w, 0, -1],
            [0, 2 / img_h, -1],
            [0, 0, 1],
        ]
    )

    x1s = x1s @ T_x1.T
    x2s = x2s @ T_x2.T
    F = eight_point_algorithm(x1s, x2s)
    # your code here
    F = T_x2.T @ F @ T_x1
    return F


def triangulate(P1, x1s, P2, x2s):
    # x1s: [n, 3]
    assert x1s.shape == x2s.shape, "x1s and x2s must have the same shape"
    x1s = x1s.reshape(-1, 1, 3)
    x2s = x2s.reshape(-1, 1, 3)

    n = len(x1s)

    # you can follow this and write it in a vectorized way, or you can do it
    # row by row, entry by entry

    A = np.zeros((n, 4, 4))

    A[:, 0, :] = x1s[..., 0] * P1[2, :] - P1[0, :]
    A[:, 1, :] = x1s[..., 1] * P1[2, :] - P1[1, :]
    A[:, 2, :] = x2s[..., 0] * P2[2, :] - P2[0, :]
    A[:, 3, :] = x2s[..., 1] * P2[2, :] - P2[1, :]

    _, _, V = svd(A)

    pts = V[:, -1, :]
    pts = pts / pts[..., -1].reshape(-1, 1)
    return pts


def t_and_R_from_pose_pair(p1, p2):
    """the R and t that transforms points from pose 1's local frame to pose 2's local frame"""

    r1, r2 = p1[:3, :3], p2[:3, :3]
    t1, t2 = p1[:3, -1], p2[:3, -1]

    t = np.linalg.inv(r2) @ (t1 - t2)
    R = r2.T @ r1

    return t, R


def pose_pair_from_t_and_R(t, R):
    """since we only have their relative orientation, the first pose
    is fixed to be identity
    """
    p1 = np.eye(4)
    # your code here
    p2 = np.vstack([np.hstack([R, t.reshape(-1, 1)]), ([0, 0, 0, 1])])
    return inv(p1), inv(p2)


def essential_from_t_and_R(t, R):
    # your code here
    E = skew(t) @ R
    return np.squeeze(E)


def t_and_R_from_essential(E):
    """this has even more ambiguity. there are 4 compatible (t, R) configurations
    out of which only 1 places all points in front of both cameras

    That the rank-deficiency in E induces 2 valid R is subtle...
    """
    # your code here; get t
    # get t from left-null space of E
    U, s, V_t = svd(E)
    t = U[:, -1]
    t_mat = skew(t.reshape(1, -1))[0]

    # now solve procrustes to get back R

    # R_90_degree matrix
    W = skew(np.array([0.0, 0.0, 1.0]))
    W[..., 2, 2] += 1.0
    W = W.squeeze()

    R = U @ W.T @ V_t

    # makes sure R has det 1, and that we have 2 possible Rs
    R1 = R * det(R)
    U[:, 2] = -U[:, 2]
    R = U @ W.T @ V_t
    R2 = R * det(R)

    four_hypothesis = [
        [t, R1],
        [-t, R1],
        [t, R2],
        [-t, R2],
    ]
    return four_hypothesis


def disambiguate_four_chirality_by_triangulation(four, x1s, x2s, full_K, draw_config=False):
    # note that our camera is pointing towards its negative z axis
    num_infront = np.array([0, 0, 0, 0])
    four_pose_pairs = []

    for i, (t, R) in enumerate(four):
        p1, p2 = pose_pair_from_t_and_R(t, R)

        P1 = full_K @ inv(p1)
        P2 = full_K @ inv(p2)

        pts = triangulate(P1, x1s, P2, x2s)[..., :3]

        R1, R2 = p1[:3, :3], p2[:3, :3]
        t1, t2 = p1[:3, 3], p2[:3, 3]
        C1, C2 = -R1.T @ t1, -R2.T @ t2

        P_in_1 = R1[2, :] @ (pts - C1).T
        P_in_2 = R2[2, :] @ (pts - C2).T

        nv1 = np.sum(P_in_1 < 0)
        nv2 = np.sum(P_in_2 < 0)
        num_infront[i] = nv1 + nv2
        four_pose_pairs.append((p1, p2))
        if draw_config:
            vis_3d(
                1500,
                1500,
                o3d_pc(_throw_outliers(pts)),
                draw_camera(full_K, p1, 1600, 1200),
                draw_camera(full_K, p2, 1600, 1200),
            )

    i = np.argmax(num_infront)
    t, R = four[i]
    p1, p2 = four_pose_pairs[i]
    return p1, p2, t, R


def F_from_K_and_E(K, E):
    # your code
    return np.linalg.inv(K.T) @ E @ np.linalg.inv(K)


def E_from_K_and_F(K, F):
    # your code
    return K.T @ F @ K


def _throw_outliers(pts):
    pts = pts[:, :3]
    mask = (np.abs(pts) > 10).any(axis=1)
    return pts[~mask]


def align_B_to_A(B, p1, p2, A):
    # B, A: [n, 3]
    assert B.shape == A.shape
    A = A[:, :3]
    B = B[:, :3]
    p1 = p1.copy()
    p2 = p2.copy()

    a_centroid = A.mean(axis=0)
    b_centroid = B.mean(axis=0)

    A = A - a_centroid
    B = B - b_centroid
    p1[:3, -1] -= b_centroid
    p2[:3, -1] -= b_centroid

    centroid = np.array([0, 0, 0])
    # root mean squre from centroid
    scale_a = (norm((A - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    scale_b = (norm((B - centroid), axis=1) ** 2).mean(axis=0) ** 0.5
    rms_ratio = scale_a / scale_b

    B = B * rms_ratio
    p1[:3, -1] *= rms_ratio
    p2[:3, -1] *= rms_ratio

    U, s, V_t = svd(B.T @ A)
    R = U @ V_t
    assert np.allclose(det(R), 1), "not special orthogonal matrix"
    new_B = B @ R  # note that here there's no need to transpose R... lol... this is subtle
    p1[:3] = R.T @ p1[:3]
    p2[:3] = R.T @ p2[:3]

    new_B = new_B + a_centroid
    new_B = as_homogeneous(new_B)
    p1[:3, -1] += a_centroid
    p2[:3, -1] += a_centroid
    return new_B, p1, p2
