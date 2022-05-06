import rasterio
import pyproj

import logging
import copy

import numpy as np
import datetime as dt

from matplotlib import pyplot as plt
import matplotlib.animation as animation

plt.ioff()

logger = logging.getLogger()
formatter = logging.Formatter(fmt="%(asctime)s - %(message)s")
loggingfile = logging.FileHandler(
    "mb_log_{}.log".format(dt.datetime.utcnow().isoformat())
)
loggingfile.setFormatter(formatter)
loggingfile.setLevel("DEBUG")
logger.addHandler(loggingfile)


class MapBlinker(object):
    def __init__(
        self,
        file1,
        file2,
        window_size=1000,
        blink=200,
        save_snapshots=True,
        snapshot_size=50,
        snapshot_root="snap",
        max_frac_zero=0.7,
        logger=logger,
    ):

        self.logger = logger

        self.file1 = file1
        self.file2 = file2

        self.window_size = window_size
        self.blink = blink

        self.save_snapshots = save_snapshots
        self.snapshot_size = snapshot_size
        self.snapshot_root = snapshot_root

        self.max_frac_zero = max_frac_zero

        self.continue_loop = True

        # read data
        self.dataset1 = rasterio.open(file1)
        self.dataset2 = rasterio.open(file2)

        assert self.dataset1.crs == self.dataset2.crs

        self.crs = self.dataset1.crs

        self.transform = pyproj.Transformer.from_crs(
            "epsg:{}".format(self.crs.to_epsg()), "epsg:4326"
        )
        self.itransform = pyproj.Transformer.from_crs(
            "epsg:4326", "epsg:{}".format(self.crs.to_epsg())
        )

        self.define_common_bounds()

        self.define_windows()

        # self.display_map(10,10)

        # self.run_display_loop()

    def run_display_loop(self):
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                if not self.continue_loop:
                    return

                self.current_col_number = i
                self.current_row_number = j
                self.display_map()

    def define_common_bounds(self):
        self.bounds1 = self.dataset1.bounds
        self.bounds2 = self.dataset2.bounds

        self.common_bounds = [
            max(self.bounds1.left, self.bounds2.left),
            max(self.bounds1.bottom, self.bounds2.bottom),
            min(self.bounds1.right, self.bounds2.right),
            min(self.bounds1.top, self.bounds2.top),
        ]

        print(
            "{:30} [{:7.1f}\t{:7.1f}\t{:7.1f}\t{:7.1f}]".format(
                self.file1,
                self.bounds1.left,
                self.bounds1.bottom,
                self.bounds1.right,
                self.bounds1.top,
            )
        )
        print(
            "{:30} [{:7.1f}\t{:7.1f}\t{:7.1f}\t{:7.1f}]".format(
                self.file2,
                self.bounds2.left,
                self.bounds2.bottom,
                self.bounds2.right,
                self.bounds2.top,
            )
        )
        print(
            "{:30} [{:7.1f}\t{:7.1f}\t{:7.1f}\t{:7.1f}]".format(
                "Common",
                self.common_bounds[0],
                self.common_bounds[1],
                self.common_bounds[2],
                self.common_bounds[3],
            )
        )

        lat_min, lon_min = self.transform.transform(
            self.common_bounds[0], self.common_bounds[1], radians=False
        )
        lat_max, lon_max = self.transform.transform(
            self.common_bounds[2], self.common_bounds[3], radians=False
        )

        print(
            "{:30} [{:6.4f}\t{:6.4f}\t{:6.4f}\t{:6.4f}]".format(
                "Common Lat-Long", lat_min, lat_max, lon_min, lon_max
            )
        )

    def define_windows(self):

        self.num_cols = np.int(
            np.ceil(self.common_bounds[2] - self.common_bounds[0]) / self.window_size
        )
        self.num_rows = np.int(
            np.ceil(self.common_bounds[3] - self.common_bounds[1]) / self.window_size
        )

        print(
            "With a window size of {} m, we will have {}x{} images".format(
                self.window_size, self.num_rows, self.num_cols
            )
        )

        self.lefts = self.common_bounds[0] + np.arange(self.num_cols) * self.window_size
        self.bottoms = (
            self.common_bounds[1] + np.arange(self.num_rows) * self.window_size
        )

    def display_map(self, fig=None, ax=None):

        self.current_data1 = self.get_data(
            1, self.current_col_number, self.current_row_number
        )
        self.current_data2 = self.get_data(
            2, self.current_col_number, self.current_row_number
        )

        if not _enough_common_good_pixels(
            self.current_data1, self.current_data2, self.max_frac_zero
        ):
            print("Not enough common pixels")
            return

        self.current_data1 = _scale_and_colorize(self.current_data1)
        self.current_data2 = _scale_and_colorize(self.current_data2)

        if ax is None and fig is not None:
            ax = fig.get_axes[0]

        if ax is not None and fig is None:
            fig = ax.get_figure()

        if fig is None and ax is None:
            self.fig, ax = plt.subplots(
                figsize=(10, 10), nrows=1, ncols=1, squeeze=True
            )

        axim1 = ax.imshow(self.current_data1, aspect=1, animated=True)
        axim2 = ax.imshow(self.current_data2, aspect=1, animated=True)

        anim = animation.ArtistAnimation(
            self.fig,
            [[axim1], [axim2]],
            repeat=True,
            interval=self.blink,
            repeat_delay=self.blink,
            blit=True,
        )

        mouse_connect_id = self.fig.canvas.mpl_connect(
            "button_press_event", self.onclick
        )
        key_connect_id = self.fig.canvas.mpl_connect("key_press_event", self.onkey)

        plt.show(block=True)

    def onkey(self, event):
        if event.key == "q" or event.key == "Q":
            self.continue_loop = False
            plt.close(self.fig)
        if event.key == "n" or event.key == "N":
            plt.close(self.fig)

    def onclick(self, event):
        x, y = (
            event.xdata,
            event.ydata,
        )  # in data pixels. this is not the same as the offset.

        X = x * 1.1 + self.lefts[self.current_col_number]  # convert to meters
        Y = (
            y * -1.1 + self.bottoms[self.current_row_number] + self.window_size
        )  # convert to meters

        # print("Click at x, y, X, Y = ({:4.1f},{:4.1f})\t==\t({:6.1f}, {:6.1f})".format(x,y,X,Y))
        lat, lon = self.transform.transform(X, Y, radians=False)
        print("Lat long = ({:7.5f}, {:7.5f})".format(lat, lon))

        if self.save_snapshots:
            self.make_snapshots(X, Y, lat, lon)

    def make_snapshots(self, X, Y, lat, lon):
        left = X - self.snapshot_size
        right = X + self.snapshot_size
        bottom = Y - self.snapshot_size
        top = Y + self.snapshot_size

        snapshot_window1 = self.dataset1.window(left, bottom, right, top)
        snapshot_data1 = self.dataset1.read(
            window=snapshot_window1, masked=True, out_dtype="float32"
        )

        snapshot_window2 = self.dataset2.window(left, bottom, right, top)
        snapshot_data2 = self.dataset2.read(
            window=snapshot_window2, masked=True, out_dtype="float32"
        )

        fig, axes = plt.subplots(
            figsize=(16, 7), nrows=2, ncols=4, sharex=True, sharey=True
        )
        axes[0][0].imshow(snapshot_data1[0, :, :], aspect=1, cmap="viridis")
        axes[0][0].set_title("File 1, Band 1")
        axes[0][1].imshow(snapshot_data1[1, :, :], aspect=1, cmap="viridis")
        axes[0][1].set_title("File 1, Band 2")
        axes[0][2].imshow(snapshot_data1[2, :, :], aspect=1, cmap="viridis")
        axes[0][2].set_title("File 1, Band 3")
        axes[0][3].imshow(snapshot_data1[3, :, :], aspect=1, cmap="viridis")
        axes[0][3].set_title("File 1, Band 4")

        axes[1][0].imshow(snapshot_data2[0, :, :], aspect=1, cmap="viridis")
        axes[1][0].set_title("File 2, Band 1")
        axes[1][1].imshow(snapshot_data2[1, :, :], aspect=1, cmap="viridis")
        axes[1][1].set_title("File 2, Band 2")
        axes[1][2].imshow(snapshot_data2[2, :, :], aspect=1, cmap="viridis")
        axes[1][2].set_title("File 2, Band 3")
        axes[1][3].imshow(snapshot_data2[3, :, :], aspect=1, cmap="viridis")
        axes[1][3].set_title("File 2, Band 4")

        fig.suptitle(
            "Multiband images at (lat, lon) = ({:7.5f},{:7.5f})".format(lat, lon)
        )

        plt.tight_layout()
        plt.savefig("{}_{:6.1f}_{:6.1f}.png".format(self.snapshot_root, X, Y))
        plt.close(fig)

    def get_data(self, source, col_number, row_number):

        assert source == 1 or source == 2

        left = self.lefts[col_number]
        bottom = self.bottoms[row_number]

        right = left + self.window_size
        top = bottom + self.window_size

        if source == 1:
            window = self.dataset1.window(left, bottom, right, top)
            data = self.dataset1.read(window=window, masked=True, out_dtype="float32")

        if source == 2:
            window = self.dataset2.window(left, bottom, right, top)
            data = self.dataset2.read(window=window, masked=True, out_dtype="float32")

        data = np.ma.masked_less(data, 1)

        return copy.copy(data)


def _scale_and_colorize(data):
    # data has size (4, M, N) as returned from the dataset.
    # output has size (M, N, 3) and is flipped up-down and left-right

    for i in range(4):
        data[i, :, :] = data[i, :, :] - np.ma.mean(data[i, :, :])
        data[i, :, :] = data[i, :, :] / 5 / np.ma.std(data[i, :, :])

    # We have 4 bands. We need to map it to RGB somehow. The easiest is to ignore band 4.
    # The bandpasses in microns are B1:0.45 - 0.52, B2:0.52 - 0.59, B3:0.62 - 0.68, B4:0.77 - 0.86.
    # TODO: Search for a better combination

    data = np.moveaxis(
        data[:3, :, :], 0, -1
    )  # shift the color axis to the end. ignore band 4.

    return data


def _enough_common_good_pixels(data1, data2, min_fraction):
    """check if there are enough common good pixels

    Here data is [4, M, N] masked arrays as return from the dataset
    """
    combined_mask = data1.mask + data2.mask

    if np.count_nonzero(combined_mask) > data1.shape[1] * data1.shape[2] * (
        min_fraction
    ):
        return False
    else:
        return True
