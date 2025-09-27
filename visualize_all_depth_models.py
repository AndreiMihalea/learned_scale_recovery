import torch
import cv2
import os
import glob
import numpy as np

from paper_plots_and_data.evaluate_depth_eigen import batch_post_process_disparity
from utils.custom_transforms import get_data_transforms
from utils.learning_helpers import disp_to_depth, load_obj, data_and_model_loader, depth_to_disp
from data.kitti_loader_eigen import KittiLoaderPytorch
from vis import plot_disp


def highlight_differences(frame, depth_map1, depth_map2, threshold=0.1, circle_radius=10):
    """
    Highlights the main differences between two depth maps by circling them.

    Parameters:
        frame: np.ndarray
            Original RBG frame.
        depth_map1: np.ndarray
            The first depth map.
        depth_map2: np.ndarray
            The second depth map.
        threshold: float
            The threshold for detecting significant differences.
        circle_radius: int
            The radius of the circles to draw around differences.

    Returns:
        output_image: np.ndarray
            The output image with circles drawn around differences.
    """
    # Normalize the depth maps to the same scale for comparison
    norm_map1 = cv2.normalize(depth_map1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm_map2 = cv2.normalize(depth_map2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute the absolute difference
    diff = cv2.absdiff(norm_map1, norm_map2)

    # Threshold the differences to focus on large differences
    _, diff_thresh = cv2.threshold(diff, int(threshold * 255), 255, cv2.THRESH_BINARY)


    # Find contours of the differences
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an output image to visualize the differences
    output_image_1 = plot_disp(depth_map1)
    output_image_1 = cv2.cvtColor(output_image_1.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    output_image_2 = plot_disp(depth_map2)
    output_image_2 = cv2.cvtColor(output_image_2.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

    # Draw circles around the contours
    for contour in contours:
        # Compute the center and radius of the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius > circle_radius:  # Only circle large differences
            cv2.circle(output_image_1, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(output_image_2, (int(x), int(y)), int(radius), (0, 0, 255), 2)

    output_image = np.hstack((output_image_1, output_image_2))

    return output_image


if __name__ == '__main__':
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    path_to_ws = '/home/andrei/workspace/nemodrive/learned_scale_recovery/'  ##update this
    path_to_dset_downsized = '/HDD_2TB/storage/KITTI/kitti_eigen_split/'

    model_list = ['kitti_eigen_pre1_unscaled', 'kitti_eigen_prefull_unscaled',
                  'kitti_eigen_pre1_m1', 'kitti_eigen_pre1_m2', 'kitti_eigen_pre1_m3',
                  'kitti_eigen_prefull_m1', 'kitti_eigen_prefull_m2', 'kitti_eigen_prefull_m3']
    depth_models_dict = {}
    base_model_dir = os.path.join(path_to_ws, 'results')

    save_dir = '/home/andrei/Documents/Facultate/PhD/learned_scale_recovery/depth_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cam_height = 1.70
    post_process = True  # use the standard post-processing that flips images, recomputes depth, and merges with unflipped depth
    benchmark = 'eigen_benchmark'  ### eigen_benchmark for improved gt, 'eigen' for standard benchmark

    splits_dir = '{}/data/splits/{}'.format(path_to_ws, benchmark)

    gt_path = os.path.join(splits_dir, "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

    for model in model_list:
        dir = os.path.join(base_model_dir, model)

        config = load_obj('{}/config'.format(dir))

        config['data_dir'] = path_to_dset_downsized + config['img_resolution'] + '_res/'  #
        config['minibatch'] = 1
        config['load_pretrained'] = True
        config['data_format'] = 'eigen'
        config['device'] = torch.device('cpu')

        pretrained_depth_path = glob.glob('{}/**depth**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]
        pretrained_pose_path = glob.glob('{}/**pose**best-loss-val_seq-**-test_seq-{}**.pth'.format(dir, ''))[0]

        _, models, device = data_and_model_loader(config, pretrained_depth_path, pretrained_pose_path, seq=None)
        depth_model = models[0]

        depth_model = depth_model.train(False).eval()

        depth_models_dict[model] = depth_model

    test_dset = KittiLoaderPytorch(config, None, mode=benchmark, transform_img=get_data_transforms(config)['test'])
    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False,
                                                    num_workers=8)

    with torch.no_grad():
        for k, data in enumerate(test_dset_loaders):
            # if k % 20 != 0:
            #     continue
            gt_depth = gt_depths[k]
            gt_height, gt_width = gt_depth.shape[:2]

            depths_dict = {}
            disps_dict = {}

            target_img, source_imgs, lie_alg, intrinsics, flow_imgs = data
            target_img, source_imgs, intrinsics = target_img['color_left'], source_imgs['color_left'], \
            intrinsics['color_left']
            target_img = target_img.to(device)
            B = target_img.shape[0]

            if post_process == True:
                # Post-processed results require each image to have two forward passes
                target_img = torch.cat((target_img, torch.flip(target_img, [3])), 0)

            for model in model_list:
                depth_model = depth_models_dict[model]
                disparities = depth_model(target_img, epoch=50)
                disps, depths = disp_to_depth(disparities[0], config['min_depth'], config['max_depth'])

                # scale_recovery(depths[:depths.shape[0]//2].to(device), intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:], config['camera_height'], True)
                # print(target_img.cpu().numpy().shape, disps.cpu().numpy().shape)
                img = cv2.cvtColor(target_img.cpu().numpy()[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(SAVE_DIR, f'{k:04}_img.png'), img * 255.)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # disp = plot_disp(disps[0].cpu().numpy().transpose(1, 2, 0))
                # disp = cv2.cvtColor(disp.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(SAVE_DIR, f'{k:04}_disp.png'), disp)
                # cv2.imshow('disp', disp)
                # cv2.waitKey(0)

                pred_disp = disps.cpu()[:, 0].numpy()

                if post_process == True:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                depth = 30 * depths
                depth = depth.cpu()[:, 0].numpy()
                depths_dict[model] = depth
                disps_dict[model] = pred_disp.astype(np.float32)

            for model in model_list:
                save_file = os.path.join(save_dir, f'{k}_{model}.png')
                disp = plot_disp(disps_dict[model][0], save_file)
                # cv2.imshow('disp', disps_dict['kitti_eigen_pre1_m1'][0])
                # cv2.waitKey(0)
                # cv2.imwrite(save_file, disp)

            pred_disp = cv2.resize(disps_dict['kitti_eigen_pre1_m3'][0], (gt_width, gt_height))
            pred_depth = 30 / pred_disp

            pred_depth = np.clip(pred_depth, 0, 80)
            # print(np.max(gt_depth), np.max(pred_depth))
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            gt_depth[gt_depth == 0] = 1e-4

            gt_disp = 30 / gt_depth



            # out = highlight_differences(img,
            #                             disps_dict['kitti_eigen_pre1_m1'][0],
            #                             disps_dict['kitti_eigen_pre1_m2'][0], threshold=0.1, circle_radius=10)
            # cv2.imshow('out', img)
            # cv2.waitKey(0)
            # save_file = os.path.join(save_dir, f'{k}_rgb.png')
            # cv2.imwrite(save_file, img * 255)
