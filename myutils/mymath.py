import torch
import math

def circular_motion(centers, radius, current_positions, time_interval, speed):
    """
    计算物体在圆周运动中的新位置。

    参数:
    centers (torch.Tensor): 圆心坐标，形状为 (batch_size, 2)，每行是 (cx, cy)
    radius (float or torch.Tensor): 圆的半径，可以是标量或形状为 (batch_size,) 的 Tensor
    current_positions (torch.Tensor): 物体当前坐标，形状为 (batch_size, 2)，第三列是 z 坐标（忽略）
    time_interval (float): 时间间隔
    speed (float or torch.Tensor): 物体的线速度大小，可以是标量或形状为 (batch_size,) 的 Tensor

    返回:
    torch.Tensor: 物体经过时间间隔后的新坐标，形状为 (batch_size, 2)
    """
    cx = centers[:, 0]  # 提取圆心的 x 坐标
    cy = centers[:, 1]  # 提取圆心的 y 坐标
    x = current_positions[:, 0]  # 提取物体的 x 坐标
    y = current_positions[:, 1]  # 提取物体的 y 坐标

    # 计算当前角度（弧度）
    current_angle = torch.atan2(y - cy, x - cx)

    # 计算角速度 (ω = v / r)
    angular_velocity = speed / radius

    # 计算经过时间间隔后的新角度
    new_angle = current_angle + angular_velocity * time_interval

    # 计算新坐标
    new_x = cx + radius * torch.cos(new_angle)
    new_y = cy + radius * torch.sin(new_angle)


    # 组合新坐标
    new_positions = torch.stack([new_x, new_y], dim=1)
    return new_positions

# # 示例用法
# centers = torch.tensor([
#     [0, 0],  # 物体 1 的圆心
#     [1, 1],  # 物体 2 的圆心
#     [2, 2],  # 物体 3 的圆心
#     [3, 3]   # 物体 4 的圆心
# ], dtype=torch.float32)  # 形状为 (batch_size, 2)

# radius = 5  # 半径（可以是标量或形状为 (batch_size,) 的 Tensor）
# current_positions = torch.tensor([
#     [5, 0, 1],  # 物体 1 的坐标 (x, y, z)
#     [6, 1, 2],  # 物体 2 的坐标 (x, y, z)
#     [7, 2, 3],  # 物体 3 的坐标 (x, y, z)
#     [8, 3, 4]   # 物体 4 的坐标 (x, y, z)
# ], dtype=torch.float32)  # 形状为 (batch_size, 3)

# time_interval = 1  # 时间间隔
# speed = 2  # 物体线速度大小（可以是标量或形状为 (batch_size,) 的 Tensor）

# new_positions = circular_motion(centers, radius, current_positions, time_interval, speed)
# print("新位置:")
# print(new_positions)