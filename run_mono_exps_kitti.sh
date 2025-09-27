#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)
method=m3
host_data_path=$(bash get_host_data_path.sh)
###KITTI Monocular Experiments

## Train on Odometry dataset:
python3 run_mono_training.py --img_resolution 'med' --flow_type 'classical' \
 --camera_height 1.70 --stereo_baseline 0.52 \
 --pretrained_plane_dir 'results/plane-model-eigen-202101201842' \
 --data_dir "${host_data_path}/andreim/kitti/kitti_odometry_downsized" \
 --estimator 'orbslam' --estimator_type 'mono' \
 --train_seq '00_02' '02_02' '06_02' '07_02' '08_02' '00_03' '02_03' '06_03' '07_03' '08_03' '11_02' '11_03' '13_02' '13_03' '14_02' '14_03' '15_02' '15_03' '16_02' '16_03' '19_02' '19_03' --val_seq '05_02' --test_seq '09_02' \
 --date $d --name test_pose --lr 1e-4 --wd 0 --num_epochs 21 --lr_decay_epoch 4 \
 --save_results --scaling_method $method \
 --pretrained_dir results/oxford_pre1_$method


## Train on Eigen split (for depth eval)
# python3 run_mono_training.py \
#   --flow_type 'classical' --data_format 'eigen' --camera_height 1.70 --stereo_baseline 0.52 \
#   --pretrained_plane_dir 'results/plane-model-eigen-202101201842' \
#   --data_dir "${host_data_path}andreim/kitti_eigen_split" \
#   --estimator 'orbslam' --estimator_type 'mono' --date $d --name kitti_eigen_nonnan_$method \
#   --lr 1e-4 --wd 0 --num_epochs 45 --lr_decay_epoch 12 --save_results --scaling_method $method \
#   --pretrained_dir results/kitti_eigen_pre1_$method

### Plane Segmentation Network Training

    ## KITTI Odometry Splits
# python3 run_plane_training.py  --pretrained_dir 'results/final_models/vo-kitti-unscaled-202102201302' --data_dir '/media/datasets/KITTI-eigen-split' --date $d --lr 1e-4 --num_epochs 10 --lr_decay_epoch 6 --save_results


    ## KITTI Eigen Split
# python3 run_plane_training.py  --pretrained_dir 'results/final_models/eigen-depth-eval-unscaled-202102131534' --data_dir '/media/datasets/KITTI-odometry-downsized-stereo' --date $d --lr 1e-4 --num_epochs 10 --lr_decay_epoch 6 --save_results






