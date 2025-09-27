import numpy as np
import torch
import sys
sys.path.append('../')
import validate
from pyslam.metrics import TrajectoryMetrics
from utils.learning_helpers import save_obj, load_obj, data_and_model_loader
import os
from validate import compute_trajectory as tt
import glob
import visualizers
from liegroups import SE3
import matplotlib.pyplot as plt

if __name__=='__main__':
    path_to_ws = '/home/andrei/workspace/nemodrive/learned_scale_recovery/'
    path_to_dset_downsized = '/HDD_2TB/storage/KITTI/kitti_odometry_downsized/'
    seq_list = ['09_02', '10_02']
    method_list = [
        # 'FT-K GR',
        # 'FT-K N',
        # 'FT-K GR+N',
        'FT-K-Full GR',
        # 'FT-K-Full N',
        # 'FT-K-Full GR+N',
        'Learned scale',
    ]
    dir_list = [
        # path_to_ws+'results/kitti_odom_pre1_m1',
        # path_to_ws+'results/kitti_odom_pre1_m2',
        # path_to_ws+'results/kitti_odom_pre1_m3',
        path_to_ws+'results/kitti_odom_prefull_m1',
        # path_to_ws+'results/kitti_odom_prefull_m2',
        # path_to_ws+'results/kitti_odom_prefull_m3',
        path_to_ws + 'results/final_models/vo-kitti-scaled-202102182020',
    ]
    other_method_list = [
        # 'Monodepth2',
        # 'Manydepth',
        # 'SC-SfMLearner',
        # 'DRO',
        'DNet*',
        'PackNet',
        'VADepth',
        'FUMET',
        # 'Dynadepth',
    ]
    other_dir_list = [
        # '/home/andrei/workspace/nemodrive/monodepth2/results/monodepth2/monodepth2_scaled_lse/trans_rot_odom_{}.npy',
        # '/home/andrei/workspace/nemodrive/monodepth2/results/manydepth/manydepth_scaled_lse/trans_rot_odom_{}.npy',
        # '/home/andrei/workspace/nemodrive/SC-SfMLearner-Release/results/to_del/{}_trans_rot.npy',
        # '/home/andrei/workspace/nemodrive/dro-sfm/results/eval/pose_scaled_lse/poses_{}_23.npy',
        '/home/andrei/workspace/nemodrive/monodepth2/results/dnet/dnet_scaled_lse/trans_rot_odom_{}.npy',
        '/home/andrei/workspace/nemodrive/packnet-sfm/results/{}_trans_rot.npy',
        '/home/andrei/workspace/nemodrive/monodepth2/results/vadepth/vadepth_scaled_lse/trans_rot_odom_{}.npy',
        '/home/andrei/workspace/nemodrive/monodepth2/results/fumet_vadepth/fumet_scaled_lse/trans_rot_odom_{}.npy',
        # '/home/andrei/workspace/nemodrive/monodepth2/results/dynadepth/R18/dynadepth_unscaled/trans_rot_odom_{}.npy',
    ]
    use_gt_rot = False

    os.makedirs('figures', exist_ok=True)

    for seq in seq_list:
        tm_dict = {}
        print('sequence: {}'.format(seq))
        seq_name = seq.split('_')[0]
        for idx, (method, dir) in enumerate(zip(method_list, dir_list)):
            tm_dict[method] = {}
            results_dir = dir + '/results/scale/'
            config = load_obj('{}/config'.format(dir))
            config['test_seq'] = [seq]
            config['data_dir'] = path_to_dset_downsized+config['img_resolution'] + '_res/'
            dpc = 'dpc' in config and config['dpc']
            # mode = config['pose_output_type']
            if dpc:
                prefix = 'dpc'
            else:
                prefix=''
            test_dset_loaders, _, _ = data_and_model_loader(config, None, None, seq=seq)

            data = load_obj('{}/{}_plane_fit'.format(results_dir, config['test_seq'][0]))
            fwd_pose_vec1 = data['fwd_pose_vec1']
            inv_pose_vec1 = data['inv_pose_vec1']
            gt_pose_vec = data['gt_pose_vec']

            # if config['dpc'] == False:
            #     prefix = ''
            # if config['dpc'] == True:
            #     prefix = prefix = 'dpc-'

            unscaled_pose_vec = fwd_pose_vec1

            if use_gt_rot == True:
                unscaled_pose_vec[:,3:6] = gt_pose_vec[:,3:6]

            scaled_pose_vec = np.array(unscaled_pose_vec)
            # scaled_pose_vec[:,0:3] = scaled_pose_vec[:,0:3]*np.repeat(data['my_scale_factor'],3,axis=1)

            ## Compute Trajectories
            if idx == 0:
                gt_traj = test_dset_loaders.dataset.raw_gt_trials[0]
                gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]

            unscaled_est, _, _, _ = tt(unscaled_pose_vec,gt_traj,method='learned')
            scaled_est, _, _, _ = tt(scaled_pose_vec,gt_traj, method='scaled')

            unscaled_est_se3 = [SE3.from_matrix(T, normalize=True) for T in unscaled_est]
            scaled_se3 = [SE3.from_matrix(T, normalize=True) for T in scaled_est]

            unscaled_est_tm = TrajectoryMetrics(gt_traj_se3, unscaled_est_se3, convention = 'Twv')
            scaled_tm = TrajectoryMetrics(gt_traj_se3, scaled_se3, convention = 'Twv')

            tm_dict[method] = {
                'learned': unscaled_est_tm,
                'scaled': scaled_tm,
            }

        for method, dir in zip(other_method_list, other_dir_list):
            tm_dict[method] = {}

            try:
                scaled_pose_vec = np.load(dir.format(int(seq_name)))
            except:
                scaled_pose_vec = np.load(dir.format(seq_name))

            scaled_est, _, _, _ = tt(scaled_pose_vec,gt_traj,method='unscaled')
            scaled_est_se3 = [SE3.from_matrix(T, normalize=True) for T in scaled_est]
            scaled_est_tm = TrajectoryMetrics(gt_traj_se3, scaled_est_se3, convention = 'Twv')

            tm_dict[method] = {
                'scaled': scaled_est_tm,
            }
            
        plotting_dict = {
            **
            {
                method: tm_dict[method]['learned'] for method in method_list
            },
            **
            {
                method: tm_dict[method]['scaled'] for method in other_method_list
            }
        }

        print(plotting_dict)
        
        est_vis = visualizers.TrajectoryVisualizer(plotting_dict)
        plt.figure()
        if use_gt_rot==True:
            fig, ax = est_vis.plot_topdown(which_plane='xz', outfile = 'figures/test-seq-{}_gt_rot_my_methods.png'.format(seq), title=r'{}'.format(seq))
        else:
            fig, ax = est_vis.plot_topdown(which_plane='xz', outfile = 'figures/test-seq-{}-mybest-vs-learned.png'.format(seq), title=r'{}'.format(seq))
