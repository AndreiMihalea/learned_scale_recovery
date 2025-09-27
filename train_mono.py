import time
import torch
from utils.learning_helpers import *
from learned_scale_recovery.models.stn import *
from data.kitti_loader import process_sample_batch
from data.kitti_loader_stereo import process_multi_pair_sample_batch
from tqdm import tqdm

def compute_pose_consistency_loss(poses, poses_inv):
    pose_consistency_loss = 0

    for pose, pose_inv in zip(poses, poses_inv):
        t_s1 = pose[:,0:6]
        t_s1_inv = pose_inv[:,0:6]
        pose_consistency_loss += (t_s1 + t_s1_inv).abs()

    return pose_consistency_loss.mean()

def solve_pose(pose_model, target_img, source_img_list, flow_imgs):
    poses, poses_inv = [], []

    flow_imgs_fwd, flow_imgs_back = flow_imgs
    
    for source_img, flow_img_fwd, flow_img_back in zip(source_img_list, flow_imgs_fwd, flow_imgs_back):
        # For PairAttentionPoseNet, we need to prepare image pairs
        # Create pairs: [target, source] for forward pose and [source, target] for inverse pose
        pair_fwd = torch.stack([target_img, source_img], dim=1)  # [B, 2, C, H, W]
        pair_inv = torch.stack([source_img, target_img], dim=1)  # [B, 2, C, H, W]
        
        # Get translation and rotation separately
        trans_fwd, rot_fwd = pose_model(pair_fwd)
        trans_inv, rot_inv = pose_model(pair_inv)
        
        # Combine translation and rotation into single pose vector [B, 6]
        pose = torch.cat([trans_fwd.squeeze(2), rot_fwd.squeeze(2)], dim=1)
        pose_inv = torch.cat([trans_inv.squeeze(2), rot_inv.squeeze(2)], dim=1)
       
        poses.append(pose)
        poses_inv.append(pose_inv)
        
    return poses, poses_inv


def solve_multi_pair_pose(pose_model, pair_imgs_list, flow_imgs):
    """Solve poses for multiple image pairs using PairAttentionPoseNet"""
    poses, poses_inv = [], []
    
    flow_imgs_fwd, flow_imgs_back = flow_imgs
    
    for i, (src_img, tgt_img) in enumerate(pair_imgs_list):
        # Create pairs: [src, tgt] for forward pose and [tgt, src] for inverse pose
        pair_fwd = torch.stack([src_img, tgt_img], dim=1)  # [B, 2, C, H, W]
        pair_inv = torch.stack([tgt_img, src_img], dim=1)  # [B, 2, C, H, W]
        
        # Get translation and rotation separately
        trans_fwd, rot_fwd = pose_model(pair_fwd)
        trans_inv, rot_inv = pose_model(pair_inv)
        
        # Combine translation and rotation into single pose vector [B, 6]
        pose = torch.cat([trans_fwd.squeeze(2), rot_fwd.squeeze(2)], dim=1)
        pose_inv = torch.cat([trans_inv.squeeze(2), rot_inv.squeeze(2)], dim=1)
       
        poses.append(pose)
        poses_inv.append(pose_inv)
        
    return poses, poses_inv


class Trainer():
    def __init__(self, config, models, loss, optimizer):
        self.config = config
        self.device = config['device']
        self.depth_model = models[0]
        self.pose_model = models[1]
        self.optimizer = optimizer
        self.loss = loss


    def forward(self, dset, epoch, phase):
        dev = self.device
        start = time.time()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.depth_model.train(True)
        else:  
            self.depth_model.train(False)  
            self.depth_model.eval()
        if phase == 'train' and self.config['freeze_depthnet'] is False: self.pose_model.train(True)
        else:
            self.pose_model.train(False)
            self.pose_model.eval()

        dset_size = len(dset)
        running_loss = None         
            # Iterate over data.
        for batch_num, data in tqdm(enumerate(dset), total=dset_size):
            target_img, source_img_list, gt_lie_alg_list, vo_lie_alg_list, flow_imgs, intrinsics, target_img_aug, \
            source_img_aug_list, gt_lie_alg_aug_list, vo_lie_alg_aug_list, intrinsics_aug = process_sample_batch(data, self.config)
                
            pose = []
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                batch_size = target_img_aug.shape[0]
                
                ## compute disparities in same batch          
                imgs = torch.cat([target_img_aug] + source_img_aug_list, 0)
                disparities = self.depth_model(imgs, epoch=epoch)
                target_disparities = [disp[0:batch_size] for disp in disparities]
                source_disp_1 = [disp[batch_size:(2*batch_size)] for disp in disparities]
                source_disp_2 = [disp[2*batch_size:(3*batch_size)] for disp in disparities]
                
                disparities = [target_disparities, source_disp_1, source_disp_2]

                if target_disparities[0].median() <=0.0000001 and target_disparities[0].mean() <=0.00000001:
                    print("warning - depth est has failed")


                poses, poses_inv = solve_pose(self.pose_model, target_img_aug, source_img_aug_list, flow_imgs)

                # print('fwd',poses[0][0,2].item(), 'gt', gt_lie_alg_list[0][0,2].item())
                # print('back',poses_inv[0][0,2].item(), 'gt inv',-gt_lie_alg_list[0][0,2].item())                    

                minibatch_loss=0  
                losses = self.loss(source_img_list, target_img, [poses, poses_inv], disparities, intrinsics_aug,
                                   epoch=epoch, batch_idx=batch_num)
                
                
                ## pose losses (simpler to add here than in the main loss function)
                if self.config['l_gt_supervised']==True and epoch > 0:
                    for i in range(0, len(source_img_list)):
                        gt_lie_alg = gt_lie_alg_aug_list[i].clone()
                        gt_lie_alg[:,0:3] = gt_lie_alg[:,0:3]/30
                        losses['l_gt_supervised'] = self.config['l_gt_supervised_weight']*torch.pow(10*(poses[i] - gt_lie_alg),2).mean()
                        losses['l_gt_supervised'] += self.config['l_gt_supervised_weight']*torch.pow(10*(poses_inv[i] + gt_lie_alg),2).mean() 
                        losses['total'] += losses['l_gt_supervised'] 
                                        
                if self.config['l_pose_consist']==True:
                    losses['l_pose_consist'] = self.config['l_pose_consist_weight']*compute_pose_consistency_loss(poses, poses_inv)
                    losses['total'] += losses['l_pose_consist']
                    

                minibatch_loss += losses['total']
                
                if running_loss is None:
                    running_loss = losses
                else:
                    for key, val in losses.items():
                        if val.item() != 0:
                            running_loss[key] += val.data
                                
                if phase == 'train':   
                    minibatch_loss.backward()
                    self.optimizer.step()
        
        print("{} epoch completed in {} seconds.".format(phase, timeSince(start)))  
        if epoch > 0:          
            ### REMOVE ****
            # running_loss = {'vo': minibatch_loss.item()}
            # running_loss['total'] = minibatch_loss.item()  
            # running_loss['l_reconstruct_forward'] = 0
            # running_loss['l_reconstruct_inverse'] = 0
            ###
            
            for key, val in running_loss.items():
                running_loss[key] = val.item()/float(batch_num)
            print('{} Loss: {:.6f}'.format(phase, running_loss['total']))
            return running_loss
        else:
            return None


class MultiPairTrainer():
    def __init__(self, config, models, loss, optimizer):
        self.config = config
        self.device = config['device']
        self.depth_model = models[0]
        self.pose_model = models[1]
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, dset, epoch, phase):
        dev = self.device
        start = time.time()
        if phase == 'train' and self.config['freeze_depthnet'] is False: 
            self.depth_model.train(True)
        else:  
            self.depth_model.train(False)  
            self.depth_model.eval()
        if phase == 'train' and self.config['freeze_posenet'] is False: 
            self.pose_model.train(True)
        else:
            self.pose_model.train(False)
            self.pose_model.eval()

        dset_size = len(dset)
        running_loss = None         
        
        # Iterate over data.
        for batch_num, data in tqdm(enumerate(dset), total=dset_size):
            # Process multi-pair data
            (pair_imgs_list, pair_imgs_aug_list, gt_lie_alg_list, vo_lie_alg_list, 
             gt_lie_alg_aug_list, vo_lie_alg_aug_list, flow_imgs, intrinsics, 
             intrinsics_aug, pair_indices) = process_multi_pair_sample_batch(data, self.config)
                
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                batch_size = pair_imgs_aug_list[0][0].shape[0]
                
                # Compute disparities for all images in the sequence
                all_imgs = []
                for pair_imgs in pair_imgs_aug_list:
                    all_imgs.extend(pair_imgs)
                
                # Remove duplicates while preserving order
                unique_imgs = []
                seen = set()
                for img in all_imgs:
                    img_id = id(img)
                    if img_id not in seen:
                        unique_imgs.append(img)
                        seen.add(img_id)
                
                imgs = torch.cat(unique_imgs, 0)
                disparities = self.depth_model(imgs, epoch=epoch)
                
                # Organize disparities by pair
                disparities_by_pair = []
                for i, (src_img, tgt_img) in enumerate(pair_imgs_aug_list):
                    # Find indices of src and tgt images in unique_imgs
                    src_idx = unique_imgs.index(src_img)
                    tgt_idx = unique_imgs.index(tgt_img)
                    
                    src_disp = [disp[src_idx*batch_size:(src_idx+1)*batch_size] for disp in disparities]
                    tgt_disp = [disp[tgt_idx*batch_size:(tgt_idx+1)*batch_size] for disp in disparities]
                    disparities_by_pair.append([tgt_disp, src_disp])

                # Solve poses for all pairs
                poses, poses_inv = solve_multi_pair_pose(self.pose_model, pair_imgs_aug_list, flow_imgs)

                minibatch_loss = 0  
                losses = self.loss(pair_imgs_list, pair_imgs_aug_list[0][0], [poses, poses_inv], 
                                 disparities_by_pair, intrinsics_aug, epoch=epoch, batch_idx=batch_num)
                
                # Pose losses (simpler to add here than in the main loss function)
                if self.config['l_pose_consist']:
                    pose_consistency_loss = compute_pose_consistency_loss(poses, poses_inv)
                    minibatch_loss += self.config['l_pose_consist_weight'] * pose_consistency_loss
                    losses['l_pose_consist'] = pose_consistency_loss.item()

                if phase == 'train':
                    minibatch_loss.backward()        
                    self.optimizer.step()

                if running_loss is None:
                    running_loss = {}
                    for key, value in losses.items():
                        running_loss[key] = value
                else:
                    for key, value in losses.items():
                        running_loss[key] += value

        epoch_loss = {}
        for key, value in running_loss.items():
            epoch_loss[key] = value / float(dset_size)

        print('{} Loss: {:.6f}'.format(phase, epoch_loss.get('l_reconstruct_forward', 0) + epoch_loss.get('l_reconstruct_inverse', 0)))
        print('{} epoch completed in {} seconds.'.format(phase, timeSince(start)))
        return epoch_loss
