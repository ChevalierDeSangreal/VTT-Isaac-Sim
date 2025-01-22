import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from .rotate_utils import euler_angles_to_matrix

class AgileLoss:
    def __init__(self, batch_size, device):
        self.device = device
        self.batch_size = batch_size

        self.direction = torch.zeros(batch_size, device=device)
        self.distance = torch.zeros(batch_size, device=device)
        self.h = torch.zeros(batch_size, device=device)
        self.ori = torch.zeros(batch_size, device=device)
        self.vel = torch.zeros(batch_size, device=device)

        self.direction_hor = torch.zeros(batch_size, device=device)
        self.direction_ver = torch.zeros(batch_size, device=device)

        self.aux = torch.zeros(batch_size, device=device)

    def reset(self, reset_idx):
        self.direction[reset_idx] = 0
        self.distance[reset_idx] = 0
        self.h[reset_idx] = 0
        self.ori[reset_idx] = 0
        self.vel[reset_idx] = 0

        self.direction_ver[reset_idx] = 0
        self.direction_hor[reset_idx] = 0

        self.aux[reset_idx] = 0

def agile_lossVer0(loss:AgileLoss, quad_state, tar_state, tar_h, tar_ori, tar_dis, step, dt, init_vec):
    """
    Same as agile_lossVer4 in isaac gym
    In version 3 the direction loss is simply the norm 2 value of theta
    However, the horizontal acceleration is related to theta
    When theta in Z axis is less than 45, it's totally acceptable
    """
    ori = quad_state[:, 3:6].clone()
    vel = quad_state[:, 6:9].clone()

    
    z_coords = torch.full((quad_state.size(0), 1), tar_h, dtype=quad_state.dtype, device=quad_state.device)
    tar_pos = torch.cat((tar_state[:, :2].clone(), z_coords), dim=1)
    tar_vel = tar_state[:, 6:9]
    
    dis = (tar_pos[:, :3].clone() - quad_state[:, :3].clone())
    rel_vel = (tar_vel.clone() - vel.clone())
    
    norm_hor_dis = torch.norm(dis[:, :2], dim=1, p=2)

    new_loss = AgileLoss(loss.batch_size, loss.device)

    rotation_matrices = euler_angles_to_matrix(ori, convention='XYZ')
    direction_vector = rotation_matrices @ init_vec
    direction_vector = direction_vector.squeeze()
    # print(direction_vector.shape)
    new_loss.direction_hor = torch.exp(1 - F.cosine_similarity(dis[:, :2], direction_vector[:, :2])) - 1
    new_loss.direction_ver = (torch.exp(torch.relu(torch.abs(direction_vector[:, 2]) - 0.7)) - 1)
    
    new_loss.direction_hor = (new_loss.direction_hor * step + new_loss.direction_hor) / (step + 1)
    new_loss.direction_ver = (new_loss.direction_ver * step + new_loss.direction_ver) / (step + 1)
    new_loss.direction = new_loss.direction_ver * 100 + new_loss.direction_hor
    # new_loss.direction = loss_direction.clone()

    # tmp_norm_hor_dis = torch.clamp(norm_hor_dis, max=5)
    # norm_hor_vel = torch.norm(vel[:, :2], dim=1, p=2)
    # loss_speed = torch.abs(tmp_norm_hor_dis - norm_hor_vel)
    loss_velocity = torch.norm(rel_vel, dim=1, p=2)
    new_loss.vel = (loss.vel * step + loss_velocity) / (step + 1)
    # new_loss.vel = loss_velocity.clone()
    # new_loss.vel = loss_speed.clone()

    loss_distance = torch.abs(norm_hor_dis - tar_dis)
    new_loss.distance = (loss.distance * step + loss_distance) / (step + 1)
    # new_loss.distance = loss_distance.clone()
    
    loss_h = torch.abs(quad_state[:, 2] - tar_h)
    new_loss.h = (loss.h * step + loss_h) / (step + 1)
    # new_loss.h = loss_h.clone()
    
    # pitch and roll are expected to be zero
    # loss_ori = torch.norm(tar_ori[:, :2] - ori[:, :2], dim=1, p=2)
    # loss_ori = torch.norm(tar_ori - ori, dim=1, p=2) - 6
    # loss_ori = torch.norm(tar_ori - ori, dim=1, p=2)
    loss_ori = 100 / (100 - 99 * torch.abs(direction_vector[:, 2]))
    new_loss.ori = (loss.ori * step + loss_ori) / (step + 1)
    # new_loss.ori = loss_ori.clone()


    # action(body rate) ---> ori & acc ---> vel ---> pos
    # loss_final = new_loss.direction + new_loss.h * 10 + new_loss.ori + new_loss.distance + new_loss.vel
    # loss_final = new_loss.ori + new_loss.distance# + new_loss.direction

    loss_final = 1 * new_loss.ori + 100 * new_loss.distance + 1 * new_loss.vel + 50 * new_loss.direction + 100 * new_loss.h
    # loss_final = new_loss.distance + new_loss.h

    return loss_final, new_loss