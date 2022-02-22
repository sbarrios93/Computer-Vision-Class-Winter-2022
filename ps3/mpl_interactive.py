import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from render import homogenize
from matplotlib import cm


N_COLORS = 10
colors = cm.get_cmap("hsv", N_COLORS)


class Plot():
    def __init__(self, ax, data):
        self.ax = ax
        h, w, _ = data['img'].shape
        # hard set the axis limit so that line plot later doesn't cause figures to shrink
        ax.set(xlim=(0, w), ylim=(h, 0))
        self.data = data
        # self.ax.axis('off')
        # self.ax.get_xaxis().set_visible(False)
        # self.ax.get_yaxis().set_visible(False)

        self.ephemeral_artists = []
        self.reset_artists()
        self.render_visual()

        # self.tx_data = []
        # self.txArtist = ax.text(
        #     0, -0.4, s='', transform=ax.transAxes
        # )
        # self.render_text('inited', overwrite=True)

        self.color_i = 0

    def reset_artists(self, clean_buffer=True):
        for vagabond in self.ephemeral_artists:
            vagabond.remove()
        self.ephemeral_artists = []

    def render_visual(self):
        self.ax.imshow(self.data['img'])

    def change_color(self):
        self.color_i = (self.color_i + 1) % N_COLORS

    def coord_press(self, x, y, button):
        self.change_color()
        assert button is not None
        if button == 1:
            self.render_mouse_clicker(x, y)
        else:
            self.reset_artists()

    def coord_query(self, x, y, button):
        self.change_color()
        if button == 1:
            self.draw_epipolar_line(x, y)
        else:
            self.reset_artists()

    def render_mouse_clicker(self, x, y):
        marker = self.ax.scatter(x, y, s=200, color=colors(self.color_i), marker='x')
        self.ephemeral_artists.append(marker)

    def draw_epipolar_line(self, x, y):
        F = self.data['F']
        point = np.array([x, y, 1])
        which_view = self.data['i']
        F = F if which_view == 1 else F.T
        line = F @ point
        _, _, V_t = np.linalg.svd(line.reshape(1, 3), full_matrices=True)
        points = homogenize(V_t[-2:])
        points = points[:, :2]
        line = self.ax.axline(points[0], points[1], color=colors(self.color_i))
        self.ephemeral_artists.append(line)

    def render_text(self, text, ephemeral=False, overwrite=True):
        if not text:
            if overwrite:
                self.tx_data = []
            return

        if overwrite:
            content = [text]
        else:
            content = self.tx_data + [text]

        if not ephemeral:
            self.tx_data = content

        content = '\n'.join(content)
        self.txArtist.set_text(content)


class Visualizer():
    """This is a generally useful tool for interactive visualization.
    You are encouraged to use this for your project.
    """
    def __init__(self):
        self.init_state()
        self.pressed = False

    def init_state(self):
        self.fig, self.canvas, self.plots = None, None, None

    def __del__(self):
        self.clear_state()

    def clear_state(self):
        if self.fig is not None:
            self.disconnect()
            plt.close(self.fig)
            self.init_state()

    def connect(self):
        decor = lambda x: x  # not using jupyter notebook for this hw
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', decor(self.on_press))
        self.cidrelease = self.canvas.mpl_connect(
            'button_release_event', decor(self.on_release))
        self.cidmotion = self.canvas.mpl_connect(
            'motion_notify_event', decor(self.on_motion))

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        self.pressed = True
        ax_in_focus = event.inaxes
        if ax_in_focus is None:
            return
        x, y, button = int(event.xdata), int(event.ydata), event.button
        for k, plot in self.plots.items():
            if ax_in_focus == plot.ax:
                plot.coord_press(x, y, button)
            else:
                plot.coord_query(x, y, button)

        self.canvas.draw()  # this is critical; otherwise state machine update not reflected

    def on_release(self, event):
        self.pressed = False

    def on_motion(self, event):
        pass

    def vis(self, img1, img2, F):
        imgs = [img1, img2]

        self.clear_state()
        num_per_row = 2
        nrows = 1
        fig = plt.figure(figsize=(15, 8), constrained_layout=False)

        self.fig = fig
        self.canvas = fig.canvas
        self.plots = dict()

        gs = GridSpec(nrows, num_per_row, figure=fig)
        for i in range(2):
            ax = fig.add_subplot(gs[i // num_per_row, i % num_per_row])
            ax.set_title(f"view {i + 1}")
            self.plots[i] = Plot(ax, {
                "img": imgs[i],
                "i": i,
                "F": F
            })
        self.connect()
        plt.show()
