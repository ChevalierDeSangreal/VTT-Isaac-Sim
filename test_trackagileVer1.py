import argparse
import sys

from omni.isaac.lab.app import AppLauncher



parser = argparse.ArgumentParser(description="Script to train VTT with differentiable dynamics")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of training iterations.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--docker", action="store_true", default=False, help="Whether process in docker")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime
import time
import pytz

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


from omni.isaac.lab.envs import (
	DirectMARLEnv,
	DirectMARLEnvCfg,
	DirectRLEnvCfg,
	ManagerBasedRLEnvCfg,
	multi_agent_to_single_agent,
)
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab.utils.assets import retrieve_file_path

from dynamics import IsaacGymDynamics
import envs
from model import TrackAgileModuleVer1
from myutils import AgileLoss, agile_lossVer0, euler_angles_to_matrix

"""
for train_trackagileVer11.py
Use relative distance as input
Use velocity
Use acceleration in body coordinate
"""

@hydra_task_config(args_cli.task, "train_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
	"""Train with RL-Games agent."""
	# override configurations with non-hydra CLI arguments
	env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
	env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

	# randomly sample a seed if seed = -1
	if args_cli.seed == -1:
		args_cli.seed = random.randint(0, 10000)

	agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
	agent_cfg["params"]["config"]["max_epochs"] = (
		args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
	)
	if args_cli.checkpoint is not None:
		resume_path = retrieve_file_path(args_cli.checkpoint)
		agent_cfg["params"]["load_checkpoint"] = True
		agent_cfg["params"]["load_path"] = resume_path
		print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
	elif agent_cfg["params"]["load_checkpoint"]:
		if args_cli.docker:
			agent_cfg['params']['load_path'] = '/workspace/isaaclab' + agent_cfg['params']['load_path']
		else:
			agent_cfg['params']['load_path'] = '/home/wangzimo/VTT/IsaacLab' + agent_cfg['params']['load_path']
		print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

	# set the environment seed (after multi-gpu config for updated rank from agent seed)
	# note: certain randomizations occur in the environment initialization so we set the seed here
	env_cfg.seed = agent_cfg["params"]["seed"]
	env_cfg.check_fov = agent_cfg["params"]["config"]["check_fov"]

	# specify directory for logging experiments
	# log_root_path = os.path.join("logs", "VTT", agent_cfg["params"]["config"]["name"])
	# log_root_path = os.path.abspath(log_root_path)
	log_root_path = "workspace/isaaclab/source/VTT/test_runs"
	print(f"[INFO] Logging experiment in directory: {log_root_path}")
	# specify directory for logging runs
	# log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now(tz = pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S"))
	log_dir = os.path.join(datetime.now(tz = pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S"), agent_cfg["params"]["config"]["name"])
    # set directory into agent config
	# logging directory path: <train_dir>/<full_experiment_name>
	agent_cfg["params"]["config"]["train_dir"] = log_root_path
	# agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

	# dump the configuration into log-directory
	if args_cli.docker:
		dump_yaml(os.path.join('/workspace/isaaclab/source/VTT/test_runs/', log_dir, "params", "env.yaml"), env_cfg)
		dump_yaml(os.path.join('/workspace/isaaclab/source/VTT/test_runs/', log_dir, "params", "agent.yaml"), agent_cfg)
		writer = SummaryWriter(os.path.join('/workspace/isaaclab/source/VTT/test_runs/', log_dir, "graphs"))
	else:
		dump_yaml(os.path.join('/home/wangzimo/VTT/IsaacLab/source/VTT/test_runs/', log_dir, "params", "env.yaml"), env_cfg)
		dump_yaml(os.path.join('/home/wangzimo/VTT/IsaacLab/source/VTT/test_runs/', log_dir, "params", "agent.yaml"), agent_cfg)
		writer = SummaryWriter(os.path.join('/home/wangzimo/VTT/IsaacLab/source/VTT/test_runs/', log_dir, "graphs"))

	if args_cli.docker:
		image_path = '/workspace/isaaclab/source/VTT/camera_output/frames/'
	else:
		image_path = '/home/wangzimo/VTT/IsaacLab/source/VTT/camera_output/frames/'

	device = env_cfg.sim.device
	num_envs = env_cfg.scene.num_envs
	seq_length = agent_cfg["params"]["config"]["seq_length"]

	# create isaac environment
	env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array") # None

	dynamic = IsaacGymDynamics()
	model = TrackAgileModuleVer1(device=device).to(device)
	if agent_cfg["params"]["load_checkpoint"]:
		model.load_model(agent_cfg["params"]["load_path"])

	tar_ori = torch.zeros((num_envs, 3)).to(device)
	init_vec = torch.tensor([[1.0, 0.0, 0.0]] * num_envs, device=device).unsqueeze(-1)

	model.eval()

	# tmp_action = torch.zeros((num_envs, 12)).to(device)
	# tmp_action[:, 2] = 3



	with torch.no_grad():
		for epoch in range(agent_cfg["params"]["config"]["max_epochs"]):
			
			print(f"Epoch {epoch} begin...")
			old_loss = AgileLoss(num_envs, device=device)
			timer = torch.zeros((num_envs,), device=device)
			input_buffer = torch.zeros(seq_length, num_envs, 9+3).to(device)
			num_reset = 0

			state_buf = env.reset()
			now_quad_state = state_buf["robot"]
			tar_state = state_buf["target"]

			# env.save_image(typ="semantic_segmentation", name=f'{0}.png', idx=0)
			# break

			for step in range(agent_cfg["params"]["config"]["len_sample"]):
				if not step % 10:
					print(f"Step {step}...")
				rel_dis = tar_state[:, :3] - now_quad_state[:, :3]
			
				world_to_body = dynamic.world_to_body_matrix(now_quad_state[:, 3:6].detach())
				body_to_world = torch.transpose(world_to_body, 1, 2)

				body_rel_dis = torch.matmul(world_to_body, torch.unsqueeze(rel_dis, 2)).squeeze(-1)
				body_vel = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 6:9], 2)).squeeze(-1)
				body_acc = torch.matmul(world_to_body, torch.unsqueeze(now_quad_state[:, 9:], 2)).squeeze(-1)
				tmp_input = torch.cat((body_vel, body_acc, now_quad_state[:, 3:6], body_rel_dis), dim=1)

				tmp_input = tmp_input.unsqueeze(0)
				input_buffer = input_buffer[1:].clone()
				input_buffer = torch.cat((input_buffer, tmp_input), dim=0)
				
				action = model.decision_module(input_buffer.clone())
				# print(tmp_input[0][0], action[0])
				# exit(0)
				new_state_dyn, acceleration = dynamic(now_quad_state, action, env.cfg.sim.dt)
				state_buf = env.step(new_state_dyn.detach())
				# state_buf = env.step(tmp_action)
				# tmp_action[:, 5] += 0.05
				new_state_sim = state_buf["robot"]
				tar_state = state_buf["target"]

				# env.save_image(typ="semantic_segmentation", name=f'{step}.png', idx=0)
				env.save_image(name=f'{step}.png', idx=0, path=image_path)

				tar_pos = tar_state[:, :3].detach()
				now_quad_state = new_state_dyn
				
				reset_buf, reset_idx = env.check_reset_out()
				not_reset_buf = torch.logical_not(reset_buf)
				num_reset += len(reset_idx)
				input_buffer[:, reset_idx] = 0


				loss, new_loss = agile_lossVer0(old_loss, now_quad_state, tar_state, 7, tar_ori, 2, timer, env.cfg.sim.dt, init_vec)
				old_loss = new_loss

				state_buf = env.reset_idxs(env_ids=reset_idx)
				now_quad_state[reset_idx] = state_buf['robot'][reset_idx].detach()
				tar_state = state_buf['target']
				
				rotation_matrices = euler_angles_to_matrix(now_quad_state[:, 3:6], convention='XYZ')
				direction_vector = rotation_matrices @ init_vec
				direction_vector = direction_vector.squeeze()

				cos_sim = F.cosine_similarity(direction_vector, rel_dis, dim=1)
				theta = torch.acos(cos_sim)
				theta_degrees = theta * 180.0 / torch.pi

				cos_sim_hor = F.cosine_similarity(direction_vector[:, :2], rel_dis[:, :2], dim=1)
				theta_hor = torch.acos(cos_sim_hor)
				theta_degrees_hor = theta_hor * 180.0 / torch.pi
				

				item_tested = 0
				horizon_dis = torch.norm(now_quad_state[item_tested, :2] - tar_pos[item_tested, :2], dim=0, p=4)
				speed = torch.norm(now_quad_state[item_tested, 6:9], dim=0, p=2)

				if reset_buf[item_tested]:
					loss[item_tested] = float('nan')

				writer.add_scalar(f'Total Loss', loss[item_tested], step)
				writer.add_scalar(f'Direction Loss/sum', old_loss.direction[item_tested], step)
				writer.add_scalar(f'Direction Loss/xy', old_loss.direction_hor[item_tested], step)
				writer.add_scalar(f'Direction Loss/z', old_loss.direction_ver[item_tested], step)
				writer.add_scalar(f'Distance Loss', old_loss.distance[item_tested], step)
				writer.add_scalar(f'Velocity Loss', old_loss.vel[item_tested], step)
				writer.add_scalar(f'Orientation Loss', old_loss.ori[item_tested], step)
				writer.add_scalar(f'Height Loss', old_loss.h[item_tested], step)

				writer.add_scalar(f'Orientation/X', direction_vector[item_tested, 0], step)
				writer.add_scalar(f'Orientation/Y', direction_vector[item_tested, 1], step)
				writer.add_scalar(f'Orientation/Z', direction_vector[item_tested, 2], step)
				writer.add_scalar(f'Orientation/Theta', theta_degrees[item_tested], step)
				writer.add_scalar(f'Orientation/ThetaXY', theta_degrees_hor[item_tested], step)
				writer.add_scalar(f'Acceleration/X', acceleration[item_tested, 0], step)
				writer.add_scalar(f'Acceleration/Y', acceleration[item_tested, 1], step)
				writer.add_scalar(f'Acceleration/Z', acceleration[item_tested, 2], step)
				writer.add_scalar(f'Horizon Distance', horizon_dis, step)
				writer.add_scalar(f'Position/X', now_quad_state[item_tested, 0], step)
				writer.add_scalar(f'Position/Y', now_quad_state[item_tested, 1], step)
				writer.add_scalar(f'Target Position/X', tar_pos[item_tested, 0], step)
				writer.add_scalar(f'Target Position/Y', tar_pos[item_tested, 1], step)
				writer.add_scalar(f'Velocity/X', now_quad_state[item_tested, 6], step)
				writer.add_scalar(f'Velocity/Y', now_quad_state[item_tested, 7], step)
				writer.add_scalar(f'Distance/X', tar_pos[item_tested, 0] - now_quad_state[item_tested, 0], step)
				writer.add_scalar(f'Distance/Y', tar_pos[item_tested, 1] - now_quad_state[item_tested, 1], step)
				writer.add_scalar(f'Action/F', action[item_tested, 0], step)
				writer.add_scalar(f'Action/X', action[item_tested, 1], step)
				writer.add_scalar(f'Action/Y', action[item_tested, 2], step)
				writer.add_scalar(f'Action/Z', action[item_tested, 3], step)
				writer.add_scalar(f'Speed/Z', now_quad_state[item_tested, 8], step)
				writer.add_scalar(f'Speed', speed, step)
				writer.add_scalar(f'Height', now_quad_state[item_tested, 2], step)


				old_loss.reset(reset_idx=reset_idx)
				timer = timer + 1
				timer[reset_idx] = 0

			ave_loss = torch.sum(loss) / num_envs
			print(f"Epoch {epoch}, Ave loss = {ave_loss}, num reset = {num_reset}")
			break

	writer.close()
	env.close()
	print("Testing Complete!")
	# env.reset()
	# # env.save_image()
	# tmp = 0
	# while 1:
	# 	tmp += 10
	# 	# tmp = tmp % 500
	# 	action[:, 2] = tmp / 100 + 3
	# 	# print("Origin Action:", action, tmp, tmp / 100)
	# 	# print(tmp)
	# 	state_buf = env.get_states()
	# 	print("Robot position after reset:", state_buf["robot"][0])
	# 	state_buf = env.step(action)
	# 	# pos = obs["policy"][0][:3]
	# 	# print(f"tmp: {tmp}, position: {pos}")
	# 	# if tmp > 10:
	# 	#     break
	# 	# print(env.get_target_seg_id())
	# 	# print(env.get_states())
	# 	# env.save_image()
	# 	# break
	# 	reset_buf, reset_idx = env.check_my_reset()
	# 	if len(reset_idx):
	# 		print(reset_buf)
	# 		break


	# # close the simulator
	# env.close()


if __name__ == "__main__":
	# run the main function
	main()
	# close sim app
	simulation_app.close()