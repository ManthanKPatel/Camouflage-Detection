""" make video from predicted masks and true images """

import os
import os.path as osp
import re
from glob import glob
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image


def overlay_mask(
    img_path: Path, mask_path: Path, save_path: Path, replace: bool = False, mask_alpha=0.5
):
    # make the parent directory if it doesn't exist of save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # make save path jpg
    save_path = save_path.with_suffix(".jpg")
    # if save_path exists, delete it
    if save_path.exists() and replace:
        save_path.unlink()

    if save_path.exists() and not replace:
        return save_path

    # load images as numpy arrays
    img = np.array(Image.open(img_path)) # image is 3 channels
    mask = np.array(Image.open(mask_path)) # mask is a 1 channel image
    # make 50% transparent
    mask = mask / 255
    print(f"Saving file: {save_path.name}")
    # multiply the mask with the images all 3 channels
    for i in range(3):
        img[:, :, i] = img[:, :, i] * mask * mask_alpha + img[:,:,i] * (1- mask_alpha)


    Image.fromarray(img).save(save_path)
    return save_path


def make_video(folder: Path, video_path: Path = None, replace: bool = False):
    """make video with all the images in the folder and save it withing the folder"""

    images = sorted(folder.glob("*.jpg"))
    if video_path is None:
        video_name = folder.name + ".mp4"
        video_path = folder / video_name
    # if save_path exists, delete it
    if video_path.exists() and replace:
        video_path.unlink()

    if video_path.exists() and not replace:
        return video_path

    # make video
    print(f"Making video: {video_path.name}")
    with imageio.get_writer(str(video_path), mode="I") as writer:
        for image in images:
            image = imageio.imread(str(image))
            writer.append_data(image)

    return video_path


if __name__ == "__main__":
    predicted_masks = Path(r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\MoCA_Result\MoCA")
    true_images = Path(r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Dataset\MoCA-Mask\MoCA_Video\TestDataset_per_sq")

    # step 1: overlay predicted masks on true images
    for folder in predicted_masks.iterdir():
        print(f"Processing folder: {folder.name}")
        for mask in folder.iterdir():
            # skip if a folder
            if mask.is_dir():
                continue

            # skip if not a png file
            if mask.suffix != ".png":
                continue

            num = re.findall(r'\d+', mask.name)[0]

            img_path = true_images / folder.name / "Imgs" / str(num)
            # img_path = true_images / "Imgs" / mask.name

            # make image path to jpg
            img_path = img_path.with_suffix(".jpg")

            save_path = predicted_masks / folder.name / "overlays" / mask.name
            overlay_mask(img_path, mask, save_path, replace=True)

        break


    # step 2: make video from overlaid images
    for folder in predicted_masks.iterdir():
        print(f"Processing folder: {folder.name}")
        overlays_folder = folder / "overlays"
        make_video(
            overlays_folder,
            video_path=Path(str(folder / folder.name) + ".mp4"),
            replace=True,
        )
        break


