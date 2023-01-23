import os
from pathlib import Path

import numpy as np
from glob import glob
import os.path as osp
import imageio
from PIL import Image
import cv2


def result_plot(mask_path, true_root_path, save_path):
    img_directory = os.listdir(true_root_path)
    i = 0
    img_list = []
    # pred_list = []

    for k, img_gt in enumerate(img_directory):
        img_list.append(img_gt)
        # print(img_list)

    for i in range(len(img_list)):
        # plt.figure(figsize=(10, 20))
        img_gt = Image.open(true_root_path + "/" + img_list[i])
        # plt.subplot(1, 2, 1)
        # plt.imshow(img_gt)

        images = img_gt.convert('RGB')
        img_copy = np.copy(images)

        # img_pred = Image.open(pred_root_path+"/"+ img_list[i])
        mask = cv2.imread(os.path.join(mask_path, img_list[i].split(".")[0] + '.png'))
        # print(mask)
        # img_pred = img_pred.astype(np.uint8)
        _, masks = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # print(masks)
        img_final = apply_mask(img_copy, masks)
        imageio.imwrite(save_path + '/' + img_list[i], img_final)
        # plt.subplot(1, 2, 2)
        # plt.imshow(img_final)


def video_maker(image_folder, video_name):
    image_folder = image_folder
    video_name = video_name

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def mask_load_moca(data_root, mask_root):
    for scene in os.listdir(osp.join(data_root)):
        print("Scene name:", scene)
        image = sorted(glob(osp.join(data_root, scene, 'Imgs', '*.jpg')))
        file = sorted(os.listdir(osp.join(data_root, scene, 'Imgs')))
        mask = sorted(glob(osp.join(mask_root, scene, '*.png')))
        directory = scene

        # Parent Directory path
        parent_dir = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\MoCA_Result\Masked_img\try1"
        # Path
        path = os.path.join(parent_dir, directory)
        # Create the directory
        if not os.path.exists(path):
            os.mkdir(path)

        for i in range(len(image)):
            images = rgb_loader(image[i])

            masks = cv2.imread(mask[i])
            # _, masks = cv2.threshold(masks, 127, 255, cv2.THRESH_BINARY)

            img_copy = np.copy(images)
            new_image = apply_mask(img_copy, masks)
            # new_image = overlay_mask()
            new_name = str(scene) + '_' + str(file[i])
            imageio.imwrite(path + '/' + new_name, new_image)


def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    # image = np.squeeze(image)
    # color = np.array([255, 255, 255], dtype='uint8')
    # mask = np.where(mask != 0, color, image)
    # image = cv2.addWeighted(image, 0.1, masked_img, 0.4, 0)
    mask_alpha = 0.5
    mask = mask / 255
    # print(f"Saving file: {save_path.name}")
    # multiply the mask with the images all 3 channels
    # for i in range(3):
    image = image * mask * mask_alpha + image * (1 - mask_alpha)
    return image


if __name__ == "__main__":

    # data_root = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Dataset\MoCA-Mask\MoCA_Video\TestDataset_per_sq"
    # mask_root = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\MoCA_Result\MoCA"
    # save_path = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\MoCA_Result\Masked_img"

    # mask_load_moca(data_root, mask_root)


    # data_root = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Dataset\Military_data\CamouflageData\img"
    # mask_root = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\2020-CVPR-SINet-New\Military"
    # save_path = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\2020-CVPR-SINet-New\Masked_Military"
    # result_plot(mask_root, data_root, save_path)
    parent_dir = r"D:\University of Toronto\MIE1517 - Intro to Deep Learning\Project\SINet\Result\MoCA_Result\Masked_img\try1"
    for scene in os.listdir(osp.join(parent_dir)):
        print("Scene name:", scene)
        # image = sorted(glob(osp.join(parent_dir, scene, 'Imgs', '*.jpg')))
        directory = scene

        # Parent Directory path
        # Path
        path = os.path.join(parent_dir, directory)
        # Create the directory
        if not os.path.exists(path):
            os.mkdir(path)
        image_folder = path
        video_name = scene + '.mp4'
        video_maker(image_folder, video_name)
