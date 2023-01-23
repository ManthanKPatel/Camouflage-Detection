import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.videodataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import imageio

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
#parser.add_argument('--model_path', type=str,
 #                   default='/content/gdrive/My Drive/DLProject/Models/MoCA/SINet_30.pth')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/SINet_MoCA/SINet_batch20_50.pth')
#parser.add_argument('--test_save', type=str,
#                    default='/content/gdrive/My Drive/DLProject/Results/MoCA/bsize_10/')
parser.add_argument('--test_save', type=str,
                    default='./Result/MoCA_Result/')
opt = parser.parse_args()
#opt, unknown = parser.parse_known_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['MoCA']:
    save_path = opt.test_save + '/' + dataset
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    # test_loader = test_dataset(data_root = '/content/gdrive/MyDrive/DLProject/Dataset/MoCA-Mask/MoCA_Video/TestDataset_per_sq/',
    #                           testsize=opt.testsize)
    test_loader = test_dataset(data_root='./Dataset/MoCA-Mask/MoCA_Video/TestDataset_per_sq/',
                               testsize=opt.testsize)

    img_count = 1
    tot_mae = 0.0
    for iteration in range(test_loader.size):
        # load data
        image, gt, name, scene = test_loader.load_data()
        image1 = image
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze

        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # misc.im save(save_path+name, cam)
        new_name = str(scene) + '_' + str(name)
        imageio.imwrite(save_path + '/' + new_name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        tot_mae += mae
        # Applying Mask on image
        # image1 = image1.detach().cpu().numpy()
        # print(image1)
        # print("image shape:", image1.shape)
        # print("mask shape:", cam.shape)
        # new_image = apply_mask(image1, cam, [255, 255, 255], alpha=0.5)
        # imageio.imwrite(mask_path + '/' + new_name, new_image)
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
print("Average MAE over whole military dataset is:  ", (tot_mae/test_loader.size))