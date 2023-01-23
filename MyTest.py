import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import imageio



parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_batch10_50.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['Military']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    # test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
    #                            gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
    #                            testsize=opt.testsize)

    test_loader = test_dataset(image_root='./Dataset/Military_data/CamouflageData/img/',
                               gt_root='./Dataset/Military_data/CamouflageData/gt/',
                               testsize=opt.testsize)


    img_count = 1
    tot_mae = 0.0
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
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
        #misc.imsave(save_path+name, cam)
        imageio.imwrite(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        tot_mae += mae
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1
print("Average MAE over whole military dataset is:  ", (tot_mae/test_loader.size))
print("\n[Congratulations! Testing Done]")
