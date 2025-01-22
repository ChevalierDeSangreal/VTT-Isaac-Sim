# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import math
import matplotlib.pyplot as plt

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_from_angle_axis
from omni.isaac.lab.sensors import TiledCameraCfg, TiledCamera, Camera, CameraCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CRAZYFLIE_CFG  # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG


from PIL import Image
# import sys
# sys.path.append('/workspace/isaaclab/source/VTT')
from myutils import quaternion_to_matrix, matrix_to_euler_angles, euler_angles_to_matrix, matrix_to_quaternion, rand_circle_point


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 1
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True
    check_fov = False # if true, reset when target get out of view, will be updated in agent_cfg
    fpv = True # if true, the camera will be placed to robot_position, otherwise it will be quadrotor

    ui_window_class_type = QuadcopterEnvWindow
    rerender_on_reset = True
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=2e-2,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8, env_spacing=20, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # target: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Target")
    target: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Target",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/VTT/assets/neo11.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            copy_from_source=False,
            semantic_tags= [("class", "target")],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "dummy": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
    wall: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Wall",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/isaaclab/source/VTT/assets/table_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            copy_from_source=False,
            semantic_tags= [("class", "wall")],
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(            pos=(0.0, 0.0, 0),
        ),
    )
    thrust_to_weight = 1.9
    moment_scale = 0.01
    # print("!!!!!!!!!!!!!!!!!!!!!!!!", quat_from_angle_axis(torch.tensor([[0, 1, 0]]), torch.tensor([math.pi / 2]))[0])
    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, -0.2), rot=(1, 0, 0, 0), convention="world"),
        data_types=["rgb", "distance_to_camera", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=33.6, vertical_aperture=22.38, clipping_range=(0.1, 10.0)
        ), # 70H * 50V
        width=1224,
        height=1224,
        colorize_semantic_segmentation=True,
        # semantic_filter = ["class"]
    )

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        # self.set_debug_vis(self.cfg.debug_vis)

        self._num_envs = self.scene.cfg.num_envs

        self.state_buf = {}
        self.state_buf["robot"] = torch.zeros((self._num_envs, 12)).to(self.device)
        self.state_buf["target"] = torch.zeros((self._num_envs, 12)).to(self.device)
        
        self.image_buf = {}
        self.image_buf["rgb"] = None
        self.image_buf["depth"] = None
        self.image_buf["seg"] = None


        self.count_step = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        self.tar_acc_norm = 2
        self.tar_acc_intervel = 125 # How many time steps will acceleration change once
        self.tar_acc = torch.zeros((self._num_envs, 2), dtype=torch.float, device=self.device)

        self.target_seg_id = next((id_ for id_, info in self._tiled_camera.data.info['semantic_segmentation']['idToLabels'].items() if info.get('class') == "quadrotor"), None)
        
        # note: the number of steps might vary depending on how complicated the scene is.
        for _ in range(12):
            self.sim.render()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._target = Articulation(self.cfg.target)
        self._wall = RigidObject(self.cfg.wall)
        # create camera
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.sensors["tiled_camera"] = self._tiled_camera
        self.scene.articulations["target"] = self._target
        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["wall"] = self._wall

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        pass

    def _apply_action(self, action):


        root_state = self._robot.data.default_root_state.clone()
        root_state[:, :3] = action[:, :3] # Position
        # adjust for camera bias
        if self.cfg.fpv:
            root_state[:, 2] += 0.2
        root_state[:, :3] += self._terrain.env_origins
        root_state[:, 3:7] = self.euler2qua(action[:, 3:6]) # Attitude
        root_state[:, 7:10] = action[:, 6:9] # Velocity
        root_state[:, 10:] = action[:, 9:] # Acceleration
        self._robot.write_root_state_to_sim(root_state)

        root_state = self._target.data.root_state_w.clone()
        inv_acc_idx = torch.nonzero((self.count_step % self.tar_acc_intervel) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        inv_acc_idx = torch.nonzero((self.count_step %( self.tar_acc_intervel * 2)) == 0).squeeze(-1)
        self.tar_acc[inv_acc_idx] *= -1
        change_acc_idx = torch.nonzero(((self.count_step % (self.tar_acc_intervel * 4)) == 0)).squeeze(-1)
        if len(change_acc_idx):
            self.tar_acc[change_acc_idx] = rand_circle_point(len(change_acc_idx), self.tar_acc_norm, self.device)
        # print("self.cfg.sim.dt:", self.tar_acc * self.cfg.sim.dt)
        # set position
        root_state[:, 2] = 3
        # set linearvels
        root_state[:, 7:9] += self.tar_acc * self.cfg.sim.dt
        root_state[:, 9] = 0
        # set angvels
        root_state[:, 10:13] = 0
        # set quats
        root_state[:, 3:7] = 0
        root_state[:, 3] = 1
        self._target.write_root_state_to_sim(root_state)
        

    def _update_states(self):
        robot_state = self._robot.data.root_state_w.clone()
        robot_state[:, :3] -= self._terrain.env_origins
        target_state = self._target.data.root_state_w.clone()
        target_state[:, :3] -= self._terrain.env_origins
        self.state_buf["robot"] = robot_state
        self.state_buf["target"] = target_state
        return
    
    def _update_images(self):
        self.image_buf["rgb"] = self._tiled_camera.data.output["rgb"].clone()
        self.image_buf["seg"] = self._tiled_camera.data.output["semantic_segmentation"].clone()
        self.image_buf["depth"] = self._tiled_camera.data.output["distance_to_camera"].clone()
        return
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def reset_idxs(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self._num_envs:
            env_ids = self._robot._ALL_INDICES

        self.count_step[env_ids] = 0

        self._robot.reset(env_ids)
        self._target.reset(env_ids)
        self._wall.reset(env_ids)
        super()._reset_idx(env_ids)

        self.tar_acc[env_ids] = rand_circle_point(len(env_ids), self.tar_acc_norm, self.device)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :2] = 0
        default_root_state[:, 2] = 3
        # adjust for camera bias
        if self.cfg.fpv:
            default_root_state[:, 2] += 0.2
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:] = 0
        default_root_state[:, 3] = 1

        # default_root_state[:, 3] = 0.954
        # default_root_state[:, 4] = 0
        # default_root_state[:, 5] = 0.301
        # default_root_state[:, 6] = 0
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # print("Robot position in reset:", default_root_state[0, :3])


        joint_pos = self._target.data.default_joint_pos[env_ids]
        joint_vel = self._target.data.default_joint_vel[env_ids]
        default_root_state = self._target.data.default_root_state[env_ids]
        default_root_state[:, 0] = 3
        default_root_state[:, 1] = 0
        default_root_state[:, 2] = 3
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 7:] = 0
        self._target.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._target.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._target.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # print("Target position in reset:", default_root_state[0, :3])

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        # print("WHAT THE FUCKKKKKKKKK?")
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()
            # print("YYYYYYYYYYYYYYYYES!!!!!!!!!!!")

        self._update_states()
        # update image buf
        self._update_images()

        return self.get_states()

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        pass
        # self.goal_pos_visualizer.visualize(self._desired_pos_w)


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idxs(indices)



        # print(f"Robot position (in world): {self._robot.data.root_state_w[0, :3] - self._terrain.env_origins[0, :3]}")
        # print(f"Robot full state: {self._robot.data.root_state_w[0]}")
        # print(f"Target position (in world): {self._target.data.root_state_w[0, :3] - self._terrain.env_origins[0, :3]}")
        # print(f"Target full state: {self._target.data.root_state_w[0]}")
        # print(f"Target velocity (in world): {self._target.data.root_state_w[0, 7:10]}")
        
        # return observations
        return self.get_states()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # # add action noise
        # if self.cfg.action_noise_model:
        #     action = self._action_noise_model.apply(action)

        # # process actions
        # self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # print("Here I am!!!")
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action(action)
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
        
        
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)


        # # -- reset envs that terminated/timed-out and log the episode information
        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)
        #     # if sensors are added to the scene, make sure we render to reflect changes in reset
        #     if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
        #         self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update states buf
        self._update_states()
        # update image buf
        self._update_images()


        self.count_step += 1

        # print(f"Root position (in world): {self._robot.data.root_state_w[0, :3] - self._terrain.env_origins[0, :3]}")
        # print(f"Robot full state: {self._robot.data.root_state_w[0]}")
        # print(f"Target position (in world): {self._target.data.root_state_w[0, :3] - self._terrain.env_origins[0, :3]}")
        # print(f"Target velocity (in world): {self._target.data.root_state_w[0, 7:10]}")
        # print(f"Target full state: {self._target.data.root_state_w[0]}")
        # # add observation noise
        # # note: we apply no noise to the state space (since it is used for critic networks)
        # if self.cfg.observation_noise_model:
        #     self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])


        # return observations, rewards, resets and extras
        return self.get_states()
    
    def qua2euler(self, qua):
        rotation_matrices = quaternion_to_matrix(
            qua)
        euler_angles = matrix_to_euler_angles(
            rotation_matrices, "XYZ")#[:, [2, 1, 0]]
        return euler_angles

    def euler2qua(self, euler):
        rotation_matrices = euler_angles_to_matrix(euler, "XYZ")
        qua = matrix_to_quaternion(rotation_matrices)
        return qua
    
    def get_images(self, typ=None):
        if typ is None:
            return self.image_buf
        else:
            return self.image_buf[typ]
    
    def get_states(self):
        robot_state = torch.zeros((self.num_envs, 12)).to(self.device)
        robot_state[:, :3] = self.state_buf["robot"][:, :3]

        # adjust camera bias
        if self.cfg.fpv:
            robot_state[:, 2] -= 0.2
        robot_state[:, 3:6] = self.qua2euler(self.state_buf["robot"][:, 3:7]) # orientation
        robot_state[:, 6:9] = self.state_buf["robot"][:, 7:10] # linear acceleration
        robot_state[:, 9:12] = self.state_buf["robot"][:, 10:13] # angular acceleration
        
        target_state = torch.zeros((self.num_envs, 12)).to(self.device)
        target_state[:, :3] = self.state_buf["target"][:, :3]
        target_state[:, 3:6] = self.qua2euler(self.state_buf["target"][:, 3:7]) # orientation
        target_state[:, 6:9] = self.state_buf["target"][:, 7:10] # linear acceleration
        target_state[:, 9:12] = self.state_buf["target"][:, 10:13] # angular acceleration
        return {"robot":robot_state, "target":target_state}
    
    def check_reset_out(self):

        if self.cfg.check_fov:
            print("???????????????")
            seg_image = self.image_buf["seg"].clone()
            print(self.target_seg_id)
            print(type(seg_image), type(seg_image == self.target_seg_id), type(self.target_seg_id))
            seg_image = (seg_image == self.target_seg_id).int() * seg_image
            sum_seg_image = torch.sum(seg_image, dim=(1, 2))
            out_sight = torch.where(sum_seg_image == 0, torch.tensor(1, device=self.device), torch.tensor(0, device=self.device)).squeeze(-1)
            out_sight_idx = torch.nonzero(out_sight).squeeze(-1)


        ones = torch.ones((self._num_envs,), device=self.device)
        out_space = torch.zeros((self._num_envs,), device=self.device)
        robot_state = self.state_buf["robot"]
        out_space = torch.where(torch.logical_or(robot_state[:, 0] > 15, robot_state[:, 0] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(robot_state[:, 1] > 15, robot_state[:, 1] < -15), ones, out_space)
        out_space = torch.where(torch.logical_or(robot_state[:, 2] > 15, robot_state[:, 2] < 0), ones, out_space)
        out_space = torch.where(torch.any(torch.isnan(robot_state[:, :3]), dim=1).bool(), ones, out_space)
        out_space_idx = torch.nonzero(out_space).squeeze(-1)

        out_time = self.episode_length_buf > self.max_episode_length
        out_time_idx = torch.nonzero(out_time).squeeze(-1)

        if self.cfg.check_fov:
            reset_buf = torch.logical_or(out_space, torch.logical_or(out_sight, out_time))
        else:
            reset_buf = torch.logical_or(out_space, out_time)
        reset_idx = torch.nonzero(reset_buf).squeeze(-1)

        return reset_buf, reset_idx

    def save_image(self, typ='rgb', path='/workspace/isaaclab/source/VTT/camera_output/frames/', name='tmp.png', idx=0):
        # print("Segmentation Info", self._tiled_camera.data.info['semantic_segmentation'])
        # image_input = self._tiled_camera.data.output['instance_segmentation_fast']
        # image_input = self._tiled_camera.data.output["semantic_segmentation"]
        image_input = self._tiled_camera.data.output[typ]
        
        # image_input = self._tiled_camera.data.output['rgb']
        file_path = path + name
        image_to_visualize = image_input[idx].cpu().numpy()

        image = Image.fromarray(image_to_visualize)
        image.save(file_path)
        # ---------------------
        # # ---------------------
        # plt.imshow(image_to_visualize, cmap='viridis')  # 可以根据需要更改 colormap
        # plt.colorbar()  # 添加颜色条以显示值范围
        # plt.title(f"Visualizing Image Input: Batch {0}")
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.savefig(file_path)
        # plt.close()