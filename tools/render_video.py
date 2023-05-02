import argparse
import os
import numpy as np
import pandas as pd
import csv
import cv2
import re
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm


def rand_cmap(nlabels, type="bright", first_color_black=True, last_color_black=False, verbose=False):
    """Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks.

    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    import colorsys

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (np.random.uniform(low=0.0, high=1), np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.9, high=1))
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colorbar, colors
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap


def plot_sequence(
    predictions_path="/home/nathan.candre/other/trackformer_results/city_pexels-kabirou-kanlanfeyi-9850032.txt",
    frames_path="/data/stot/datasets_trackformer/fmv/frames/city_pexels-kabirou-kanlanfeyi-9850032",
    output_dir="/home/nathan.candre/other/trackformer_results/videos/city_pexels-kabirou-kanlanfeyi-9850032",
    delimiter=" ",
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reader = csv.reader(open(predictions_path), delimiter=delimiter)

    next(reader)

    keys = [
        "frame_id",
        "track_id",
        "xmin",
        "ymin",
        "width",
        "height",
        "score",
        "label",
        "center_latitude",
        "center_longitude",
    ]
    types = [int, int, float, float, float, float, float, int]
    rows = [dict(zip(keys, [eltype(el) for el, eltype in zip(row, types)])) for row in reader][1:]

    frames = {}

    for row in rows:
        if row["frame_id"] in frames:
            frames[row["frame_id"]].append(row)
        else:
            frames[row["frame_id"]] = [row]

    cmap = rand_cmap(500, type="bright", first_color_black=False, last_color_black=False)

    sorted_imgs = sorted(os.listdir(frames_path))
    for iframe, img_name in enumerate(tqdm(sorted_imgs)):
        img_path = os.path.join(frames_path, img_name)
        img = cv2.imread(img_path)[:, :, (2, 1, 0)]
        height, width, _ = img.shape

        fig = plt.figure()
        fig.set_size_inches(width / 96, height / 96)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)

        if iframe + 1 in frames:
            for bbox in frames[iframe + 1]:
                ax.add_patch(
                    plt.Rectangle(
                        (bbox["xmin"], bbox["ymin"]),
                        bbox["width"],
                        bbox["height"],
                        fill=False,
                        linewidth=2.0,
                        color=cmap(bbox["track_id"]),
                    )
                )
                annotate_color = cmap(bbox["track_id"])

        plt.axis("off")
        plt.draw()
        plt.savefig(os.path.join(output_dir, os.path.basename(img_path)), dpi=96)
        plt.close()


def frames_to_video(output_path, frame_ext="png"):
    # ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p out.mp4

    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            "30",
            "-pattern_type",
            "glob",
            "-i",
            os.path.join(output_path, "*." + frame_ext),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            os.path.join(output_path, "output.mp4"),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path")
    parser.add_argument("frames_path")
    parser.add_argument("output_path")
    parser.add_argument("-d", "--delimiter", default=" ")
    parser.add_argument("-ext", "--frame_ext", default="png")

    args = parser.parse_args()
    predictions_path, frames_path, output_path = args.predictions_path, args.frames_path, args.output_path
    delimiter, frame_ext = args.delimiter, args.frame_ext

    output_path = os.path.join(output_path, os.path.splitext(os.path.basename(frames_path))[0])

    predictions_path = "/home/nathan.candre/other/MOTRv2_results/tracker/uav0000126_00001_v.txt"
    frames_path = "/data/stot/datasets_mot/visdrone/MOT/train/sequences/uav0000126_00001_v"
    output_path = "/home/nathan.candre/other/MOTRv2_results/videos"

    plot_sequence(predictions_path, frames_path, output_path, delimiter)
    frames_to_video(output_path, frame_ext)
