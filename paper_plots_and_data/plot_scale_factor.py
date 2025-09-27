import numpy as np
import torch

import sys
sys.path.append('../')
import validate
from utils.learning_helpers import save_obj, load_obj, data_and_model_loader
import os
from validate import compute_trajectory as tt
import glob
import csv
import matplotlib.pyplot as plt
import matplotlib

# Removes the XWindows backend (useful for producing plots via tmux without -X)
matplotlib.use('TkAgg')
matplotlib.rcParams["font.size"] = 26

path_to_ws = '/home/andrei/workspace/nemodrive/learned_scale_recovery/'
path_to_dset_downsized = '/HDD_2TB/storage/KITTI/kitti_odometry_downsized/'
# seq_list = ['00_02', '02_02', '06_02', '07_02', '08_02', '05_02', '09_02', '10_02'] 
seq_list = ['05_02', '09_02', '10_02'] 
method_list = [
    # 'FT-K-Full GR',
    # 'FT-K-Full N',
    'FT-K-Full GR+N',
    # 'Learned scale',
]

dir_list = [
    # path_to_ws + 'results/kitti_odom_prefull_m1',
    # path_to_ws + 'results/kitti_odom_prefull_m2',
    path_to_ws + 'results/kitti_odom_prefull_relaxed_box_m3',
    # path_to_ws + 'results/final_models/vo-kitti-scaled-202102182020',
]
epochs = [1, 8, 10, 16, 19]

csv_header1 = ['Method', 'Epoch']
# csv_header2 = ['', 'Train', '', '', '','', 'Val', 'Test']
csv_header3 = [''] + seq_list + ['Mean']
csv_header4 = [''] + [f'Epoch_{epoch}' for epoch in epochs] + ['Mean']

with open('scale_variance_full.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header1)
    # writer.writerow(csv_header2)
    writer.writerow(csv_header4)
    
    scale_factors = {}
    for method, dir in zip(method_list, dir_list):
        scale_factors[method] = {}
        scale_factor_std_dev_list = []

        results_dir = dir

        data = load_obj('{}/scale_factor'.format(results_dir))

        print(data.keys())

        for epoch in epochs:
            scale_factor = data[epoch]
            scale_factor_mean = np.average(scale_factor)
            scale_factor_std = np.std(scale_factor)
            scale_factor_std_dev_list.append(scale_factor_std)
            scale_factors[method][epoch] = scale_factor
            print('{} {} scale factor: {}'.format(method, epoch, scale_factor_mean))
            print('{} {} scale factor std. dev.: {}'.format(method, epoch, scale_factor_std))


        mean_std = np.mean(scale_factor_std_dev_list)
        scale_factor_std_dev_list.append(mean_std)
        scale_factor_std_dev_list = ["%.4f" % e for e in scale_factor_std_dev_list]
        writer.writerow([method] + scale_factor_std_dev_list)

for epoch in epochs:
    plt.figure(figsize=(10, 6))
    plt.xlabel('Iteration')
    plt.ylabel('Scaling factor')
    if epoch == 1:
        plt.ylim([0.35, 2.85])
    else:
        plt.ylim([0.8, 1.2])
    alphas = [1., 0.8, 0.6, 0.5]
    for method, alpha in zip(method_list, alphas):
        scale_factor = scale_factors[method][epoch]
        plt.plot(scale_factor[:1000], label=method, alpha=alphas[0], linewidth=2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'figures/{epoch}-scale-m3.png')

        
# for seq in seq_list:
#     plt.figure()
#     plt.tight_layout()
#     plt.subplots_adjust(bottom=0.15)
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.tick_params(labelsize=22)
#     plt.grid()
#     plt.plot(scale_factors['scaled'][seq], linewidth=2, label='Scaled', rasterized=True)
#     plt.plot(scale_factors['unscaled'][seq], linewidth=2, label='Unscaled', rasterized=True)
#     plt.legend(fontsize=15)
#     plt.ylim([0.6,2.4])
#     plt.ylabel('Scale Factor', fontsize=22)
#     plt.xlabel('Timestep', fontsize=22)
#     plt.savefig('figures/seq-{}-scale.png'.format(seq))