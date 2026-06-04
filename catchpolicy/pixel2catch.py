import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import TiledCamera, save_images_to_file, ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform, quat_from_euler_xyz
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg

import os
import torch
from typing import Sequence, Dict
import math
import numpy as np
import cv2
import time
from collections import deque

##
# Pre-defined configs
##
from torch._tensor import Tensor
from .pixel2catch_cfg import DynamicCatchEnvCfg

class DynamicCatchEnv(DirectMARLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: DynamicCatchEnvCfg

    def __init__(self, cfg: DynamicCatchEnvCfg,
                 render_mode: str | None = None, **kwargs):
        
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.num_dofs = self.Catcher.num_joints

        joint_pos_limits = self.Catcher.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower_limits = joint_pos_limits[0, :, 0]
        self.dof_lower_limits[0] = 2.0944 #2.35619
        # self.dof_lower_limits[1] = -2.0944
        # self.dof_lower_limits[2] = 1.5708
        self.dof_lower_limits[3] = 2.0944
        self.dof_lower_limits[4] = -1.91986
        self.dof_lower_limits[5] = 2.79253

        # thumb joint limit
        self.dof_lower_limits[6] = 0.2630

        self.dof_upper_limits = joint_pos_limits[0, :, 1]
        self.dof_upper_limits[0] = 4.18879 #3.92699
        # self.dof_upper_limits[1] = -0.610865
        # self.dof_upper_limits[2] = 2.0944 #2.61799
        # self.dof_upper_limits[3] = 3.14159
        self.dof_upper_limits[4] = -1.5708
        self.dof_upper_limits[5] = 3.49066

        self.dof_speed_scales = torch.ones_like(self.dof_lower_limits)
        self.palm_link_idx = self.Catcher.find_bodies("palm")[0][0]
        self.wrist_3_link_idx = self.Catcher.find_bodies("wrist_3_link")[0][0]
        self.thumb_tip_idx = self.Catcher.find_bodies("link_03")[0][0]
        self.index_tip_idx = self.Catcher.find_bodies("link_13")[0][0]
        self.middle_tip_idx = self.Catcher.find_bodies("link_23")[0][0]
        self.ring_tip_idx = self.Catcher.find_bodies("link_33")[0][0]

        # list of actuated joints
        self.dof_indices = list()
        for joint_name in self.cfg.actuated_joint_names:
            self.dof_indices.append(self.Catcher.joint_names.index(joint_name))

        # buffers for position targets
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_joint_pos = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # sliding window average
        self.window_size = 100
        self.success_window = deque(maxlen=self.window_size)
        self.track_window = deque(maxlen=self.window_size)
        self.sliding_success_rate = torch.zeros(1, device=self.device)
        self.sliding_track_rate = torch.zeros(1, device=self.device)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.tracked = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.consecutive_tracked = torch.zeros(1, dtype=torch.float, device=self.device)
        self.avg_factor = torch.tensor(self.cfg.avg_factor, dtype=torch.float, device=self.device)

        # buffers for states
        self.states_seq = torch.zeros((self.num_envs, self.cfg.num_states_frame*(self.cfg.num_stacks)), device=self.device)

        # buffers for obs
        self.arm_obs_buf = torch.zeros((self.num_envs, self.cfg.num_obs_arm*(self.cfg.num_stacks)), device=self.device)
        self.hand_obs_buf = torch.zeros((self.num_envs, self.cfg.num_obs_hand*(self.cfg.num_stacks)), device=self.device)

        # default object position
        self.object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_rot[:,0] = 1.0

        # define initial distance
        self.relative_hand_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_dist_hand_obj = torch.zeros((self.num_envs), device=self.device)
        self.prev_dist_thumb_obj = torch.zeros((self.num_envs), device=self.device)
        self.prev_dist_index_obj = torch.zeros((self.num_envs), device=self.device)
        self.prev_dist_middle_obj = torch.zeros((self.num_envs), device=self.device)
        self.prev_dist_ring_obj = torch.zeros((self.num_envs), device=self.device)
        self.prev_object_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_catch_step = torch.zeros((self.num_envs), device=self.device)

        self.prev_bbox = torch.zeros((self.num_envs,4,2), device=self.device)
        self.prev_bbox_info = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # buffers for resulting
        self.total_timesteps = 0
        self.total_ep = torch.zeros((self.num_envs), device=self.device)
        self.track_ep = torch.zeros((self.num_envs), device=self.device)
        self.success_ep = torch.zeros((self.num_envs), device=self.device)
        self.avg_track = torch.zeros(1, device=self.device)
        self.avg_success = torch.zeros(1, device=self.device)

        # default buffers
        self.track_buf = torch.zeros((self.num_envs), device=self.device)
        self.success_buf = torch.zeros((self.num_envs), device=self.device)
        self.trial_num = torch.zeros((self.num_envs), device=self.device)
        self.track_step = torch.zeros((self.num_envs), device=self.device)
        self.catch_step = torch.zeros((self.num_envs), device=self.device)
        self.miss_step = torch.zeros((self.num_envs), device=self.device)

        self.frame_count = 0

        # random release
        self.object_spawn_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_spawn_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_throw_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_release_delay = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
        self.object_released = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        # action buffers for delay handling
        self.applied_actions = {
            "arm": torch.zeros((self.num_envs, 6), device=self.device),
            "hand": torch.zeros((self.num_envs, 13), device=self.device),
        }

        self.actions = {
            "arm": torch.zeros((self.num_envs, 6), device=self.device),
            "hand": torch.zeros((self.num_envs, 13), device=self.device),
        }

        self.pixel_cnt = torch.zeros((self.num_envs), device=self.device)
        self.prev_cnt = torch.zeros((self.num_envs), device=self.device)

        # self.max_obs_latency_step = 4
        # self.max_action_latency_step = 4
        self.max_pf_latency_step = 3
        # self.latency_arm_buf = torch.zeros(
        #     (self.num_envs, self.max_obs_latency_step, self.cfg.num_obs_arm),
        #     device=self.device,
        # )
        # self.latency_hand_buf = torch.zeros(
        #     (self.num_envs, self.max_obs_latency_step, self.cfg.num_obs_hand),
        #     device=self.device,
        # )
        # self.latency_arm_action_buf = torch.zeros(
        #     (self.num_envs, self.max_action_latency_step, 6),
        #     device=self.device,
        # )
        # self.latency_hand_action_buf = torch.zeros(
        #     (self.num_envs, self.max_action_latency_step, 13),
        #     device=self.device,
        # )
        self.latency_pf_buf = torch.zeros(
            (self.num_envs, self.max_pf_latency_step, 6),
            device=self.device,
        )
        # self.current_obs_latency = torch.randint(
        #     0, self.max_obs_latency_step, (self.num_envs,), device=self.device
        # )
        # self.current_act_latency = torch.randint(
        #     0, self.max_action_latency_step, (self.num_envs,), device=self.device
        # )
        self.current_pf_latency = torch.randint(
            0, self.max_pf_latency_step, (self.num_envs,), device=self.device
        )

        # episodic return
        self._track_rewards_arm = deque(maxlen=150)
        self._track_rewards_hand = deque(maxlen=150)
        self.cumulative_reward_arm = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.cumulative_reward_hand = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

        self._track_success_rate = deque(maxlen=150)
        self.episode_success_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

        self._track_track_rate = deque(maxlen=150)
        self.episode_track_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)

        # markers
        # self.palm_marker = VisualizationMarkers(self.cfg.palm_marker_cfg)

    def _setup_scene(self):
        self.Catcher = Articulation(self.cfg.Catcher)
        self._object = RigidObject(self.cfg.objects)
        self.center_camera = TiledCamera(self.cfg.center_camera)
        self._table = RigidObject(self.cfg.table_cfg)
        self._table_contact = ContactSensor(self.cfg.table_contact)

        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        self.scene.articulations["Catcher"] = self.Catcher
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["center_camera"] = self.center_camera
        self.scene.rigid_objects["table"] = self._table
        self.scene.sensors["table_contact"] = self._table_contact

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        spawn_ground_plane(self.cfg.terrain.prim_path, GroundPlaneCfg(color=(0.96,0.96,0.8)))

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        self.actions["arm"] = actions["arm"].clone().clamp(-1.0,1.0)
        self.actions["hand"] = actions["hand"].clone().clamp(-1.0,1.0)

        # self.next_t_action["arm"] = actions["arm"].clone().clamp(-1.0,1.0)
        # self.next_t_action["hand"] = actions["hand"].clone().clamp(-1.0,1.0)

        current_step = self.episode_length_buf

        # release ���대컢�� �� env�ㅻ쭔 李얘린
        mask = (current_step >= self.object_release_delay) & (~self.object_released)

        if mask.any():
            env_ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)

            root_state = self._object.data.root_state_w.clone()
            selected = root_state[env_ids]

            # 1) release �쒓컙�� �ㅼ젣 �띾룄 �곸슜
            selected[:, 0:3] = self.object_spawn_pos[env_ids] + self.scene.env_origins[env_ids]
            selected[:, 3:7] = self.object_spawn_rot[env_ids]
            selected[:, 7:10] = self.object_throw_vel[env_ids]

            # 2) �곹깭 諛섏쁺
            self._object.write_root_state_to_sim(selected, env_ids)

            # 3) release �꾨즺 �쒖떆
            self.object_released[env_ids] = True

        pending_mask = ~self.object_released

        if pending_mask.any():
            env_ids = torch.nonzero(pending_mask, as_tuple=False).squeeze(-1)

            root_state = self._object.data.root_state_w.clone()
            selected = root_state[env_ids]

            g = 9.81
            decim = self.cfg.decimation
            physics_dt = self.dt / decim   # �먮뒗 cfg.sim.dt �ъ슜
            stay_vz = g * physics_dt * (decim + 1) / 2.0

            selected[:, 9] = stay_vz # root_state[:,7:10] = (vx, vy, vz)

            # 2) �곹깭 諛섏쁺
            self._object.write_root_state_to_sim(selected, env_ids)

    def _apply_action(self):
        # apply arm actions
        # self.latency_arm_action_buf = torch.roll(self.latency_arm_action_buf, shifts=1, dims=1)
        # self.latency_arm_action_buf[:,0] = self.actions["arm"]
        # delayed_arm_action = self.latency_arm_action_buf[
        #     torch.arange(self.num_envs, device=self.device),
        #     self.current_act_latency
        # ]
        # self.latency_hand_action_buf = torch.roll(self.latency_hand_action_buf, shifts=1, dims=1)
        # self.latency_hand_action_buf[:,0] = self.actions["hand"]
        # delayed_hand_action = self.latency_hand_action_buf[
        #     torch.arange(self.num_envs, device=self.device),
        #     self.current_act_latency
        # ]
        # self.cur_targets[:,self.dof_indices[:5]] = self.cur_joint_pos[:,:5]+self.actions["arm"]*0.15
        # self.cur_targets[:,self.dof_indices[:5]] = self.cur_joint_pos[:,:5] + self.prev_t_action["arm"]*0.13
        self.cur_targets[:,self.dof_indices[:6]] = self.prev_targets[:,self.dof_indices[:6]] + self.actions["arm"]*0.05
        # self.cur_targets[:,self.dof_indices[:6]] = self.prev_targets[:,self.dof_indices[:6]] + self.prev_t_action["arm"]*0.05
        # self.cur_targets[:,self.dof_indices[:6]] = self.prev_targets[:,self.dof_indices[:6]] + delayed_arm_action*0.05

        # apply hand actions
        # self.cur_targets[:,self.dof_indices[6:]] = scale(delayed_hand_action, self.dof_lower_limits[self.dof_indices[6:]], self.dof_upper_limits[self.dof_indices[6:]])
        # self.cur_targets[:,self.dof_indices[6:]] = scale(self.prev_t_action["hand"], self.dof_lower_limits[self.dof_indices[6:]], self.dof_upper_limits[self.dof_indices[6:]])
        self.cur_targets[:,self.dof_indices[6:]] = scale(self.actions["hand"], self.dof_lower_limits[self.dof_indices[6:]], self.dof_upper_limits[self.dof_indices[6:]])
        # apply act moving average
        self.cur_targets[:,self.dof_indices[6:]] = (
            self.cfg.act_moving_average * self.cur_targets[:,self.dof_indices[6:]]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:,self.dof_indices[6:]])
        self.cur_targets[:,self.dof_indices[:]] = torch.clamp(self.cur_targets[:,self.dof_indices[:]], self.dof_lower_limits[self.dof_indices[:]], self.dof_upper_limits[self.dof_indices[:]])

        self.Catcher.set_joint_position_target(self.cur_targets[:,self.dof_indices[:]], joint_ids=self.dof_indices[:])

        self.prev_targets[:,self.dof_indices[:]] = self.cur_targets[:,self.dof_indices[:]]

        # self.applied_actions["arm"] = delayed_arm_action.clone()
        # self.applied_actions["hand"] = delayed_hand_action.clone()

        # self.applied_actions["arm"] = self.prev_t_action["arm"].clone()
        # self.applied_actions["hand"] = self.prev_t_action["hand"].clone()
        self.applied_actions["arm"] = self.actions["arm"].clone()
        self.applied_actions["hand"] = self.actions["hand"].clone()

        # self.prev_t_action["arm"] = self.next_t_action["arm"].clone()
        # self.prev_t_action["hand"] = self.next_t_action["hand"].clone()

    # post-physics step calls

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: 
        self._compute_intermediate_values()       
        # get object position
        current_dist_hand_obj = torch.norm(self.hand_pos - self.object_pos, p=2, dim=-1)
        in_range = (current_dist_hand_obj < 0.08)
        
        self.catch_step = torch.where(in_range, self.catch_step + 1, torch.zeros_like(self.catch_step))
        self.track_step = torch.where(in_range, self.track_step + 1, self.track_step)

        # success condition
        K = 20 # hold steps
        success = (self.catch_step >= K)
        
        # success = (self.track_step >= K) & in_range
        track = (self.track_step >= 1)

        # drop condition
        drop_object = ((self.object_pos[:,2] < 0.45)) | (self.object_pos[:,0] < 0.03)
        
        # timeout condition
        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        # miss the object
        miss_object = torch.where((self.object_pos[:,0]<self.wrist3_pos[:,0]), torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
        self.miss_step = torch.where(miss_object==1, self.miss_step + 1, torch.zeros_like(self.catch_step))

        # collision with table
        table_collision = self.table_contact

        # terminated = {agent: (drop_object | time_out | success | table_collision) for agent in self.cfg.possible_agents}
        terminated = {agent: (drop_object | time_out | success) for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}

        # Store done_mask for use in _get_rewards()
        self.done_mask = terminated["arm"]
        self.success_mask = success
        self.track_mask = track

        # done_ids = torch.nonzero(self.done_mask, as_tuple=False).squeeze(-1)
        # for env_id in done_ids:
        #     s = float(success[env_id].item())
        #     t = float(track[env_id].item())
        #     self.success_window.append(s)
        #     self.track_window.append(t)
        # if len(self.success_window) > 0:
        #     self.sliding_success_rate[...] = sum(self.success_window) / len(self.success_window)
        # if len(self.track_window) > 0:
        #     self.sliding_track_rate[...] = sum(self.track_window) / len(self.track_window)

        # tracked_ep = (self.track_step >= 1) & self.done_mask
        # success_ep = success & self.done_mask

        # self.total_ep += self.done_mask.float()
        # self.track_ep += tracked_ep.float()
        # self.success_ep += success_ep.float()

        # self.avg_track = (self.track_ep.sum() / torch.clamp(self.total_ep.sum(), min=1.0))
        # self.avg_success = (self.success_ep.sum() / torch.clamp(self.total_ep.sum(), min=1.0))

        # ema stactics
        # num_done = self.done_mask.float().sum()
        # sum_tracked = tracked_ep.float().sum()
        # sum_success = success_ep.float().sum()
        # if num_done > 0:
        #     batch_track_rate = sum_tracked / num_done
        #     batch_success_rate = sum_success / num_done

        #     # EMA: new = avg_factor * batch + (1 - avg_factor) * old
        #     self.consecutive_tracked = (
        #         self.avg_factor * batch_track_rate
        #         + (1.0 - self.avg_factor) * self.consecutive_tracked
        #     )
        #     self.consecutive_successes = (
        #         self.avg_factor * batch_success_rate
        #         + (1.0 - self.avg_factor) * self.consecutive_successes
        #     )

        # calculate success rate
        # max_trial = 100

        # self.trial_num = torch.where(time_outs["arm"] | terminated["arm"], self.trial_num + 1, self.trial_num)
        # self.success_buf = torch.where(self.trial_num <= max_trial, torch.where((time_outs["arm"]), self.success_buf + 1, self.success_buf), self.success_buf)
        # self.track_buf = torch.where(self.trial_num <= max_trial, torch.where((self.track_step > 0)&(time_outs["arm"] | terminated["arm"]), self.track_buf + 1, self.track_buf), self.track_buf)
        
        # progress = torch.sum(torch.clamp(self.trial_num,max=max_trial))/(self.num_envs*max_trial)*(100.0)
        # print(self.success_buf.sum(), self.track_buf.sum())
        # if torch.all(self.trial_num >= max_trial):
        #     avg_success = torch.sum(self.success_buf).float() / (self.num_envs * max_trial)
        #     avg_track = torch.sum(self.track_buf).float() / (self.num_envs * max_trial)
        #     print(avg_success, avg_track)
        # else:
        #     print(f"Proceeding...{progress.cpu().item():.2f}%")

        return terminated, time_outs

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        (
            total_reward,
            # self.reset_buf,
            current_dist_hand_obj,
            current_dist_thumb_obj,
            current_dist_index_obj,
            current_dist_middle_obj,
            current_dist_ring_obj,
            # self.successes,
            # self.consecutive_successes,
            # self.tracked,
            # self.consecutive_tracked,
            reward_component
        ) = compute_rewards(
            self.reset_buf,
            self.episode_length_buf,
            self.max_episode_length,
            self.object_pos,
            self.hand_pos,
            self.thumb_tip_pos,
            self.index_tip_pos,
            self.middle_tip_pos,
            self.ring_tip_pos,
            self.prev_dist_hand_obj,
            self.prev_dist_thumb_obj,
            self.prev_dist_index_obj,
            self.prev_dist_middle_obj,
            self.prev_dist_ring_obj,
            self.actions,
            self.cur_joint_pos,
            self.cfg.drop_penalty,
            self.track_step,
            self.catch_step,
            self.miss_step,
            self.table_contact,
            # self.successes,
            # self.consecutive_successes,
            # self.tracked,
            # self.consecutive_tracked,
            self.avg_factor,
            # self.delta_object_pos,
            # self.delta_hand_pos,
            self.dt
        )

        self.prev_dist_hand_obj = current_dist_hand_obj.clone()
        self.prev_dist_thumb_obj = current_dist_thumb_obj.clone()
        self.prev_dist_index_obj = current_dist_index_obj.clone()
        self.prev_dist_middle_obj = current_dist_middle_obj.clone()
        self.prev_dist_ring_obj = current_dist_ring_obj.clone()
        self.prev_hand_pos = self.hand_pos.clone()
        self.prev_object_pos = self.object_pos.clone()

        # logging data
        self.cumulative_reward_arm.add_(total_reward["arm"].unsqueeze(-1))
        self.cumulative_reward_hand.add_(total_reward["hand"].unsqueeze(-1))

        # self.episode_success_buf = torch.maximum(self.episode_success_buf, self.success_mask.float().unsqueeze(-1))
        self.episode_success_buf = torch.where(self.success_mask.unsqueeze(-1), torch.ones_like(self.episode_success_buf), torch.zeros_like(self.episode_success_buf))
        # self.episode_track_buf = torch.maximum(self.episode_track_buf, self.track_mask.float().unsqueeze(-1))
        self.episode_track_buf = torch.where(self.track_mask.unsqueeze(-1), torch.ones_like(self.episode_track_buf), torch.zeros_like(self.episode_track_buf))

        finished_episodes = self.done_mask.nonzero(as_tuple=False)
        if finished_episodes.numel():
            # storage cumulative rewards
            self._track_rewards_arm.extend(self.cumulative_reward_arm[finished_episodes][:,0].reshape(-1).tolist())
            self._track_rewards_hand.extend(self.cumulative_reward_hand[finished_episodes][:,0].reshape(-1).tolist())

            self._track_success_rate.extend(self.episode_success_buf[finished_episodes][:,0].reshape(-1).tolist())
            self._track_track_rate.extend(self.episode_track_buf[finished_episodes][:,0].reshape(-1).tolist())

            # reset cumulative rewards
            # self.cumulative_reward_arm[finished_episodes] = 0.0
            # self.cumulative_reward_hand[finished_episodes] = 0.0
            # self.episode_success_buf[finished_episodes] = 0.0
            # self.episode_track_buf[finished_episodes] = 0.0
        if len(self._track_rewards_arm):
            track_rewards_arm = torch.tensor(self._track_rewards_arm, device=self.device)
            track_rewards_hand = torch.tensor(self._track_rewards_hand, device=self.device)
            track_success_rate = torch.tensor(self._track_success_rate, device=self.device)
            track_track_rate = torch.tensor(self._track_track_rate, device=self.device)

            self.extras["log"] = {
                "Episodic Return Arm Mean": track_rewards_arm.mean(),
                "Episodic Return Hand Mean": track_rewards_hand.mean(),
                "Success Rate": track_success_rate.mean(),
                "Track Rate": track_track_rate.mean(),
            }

        # print("episode success buf:", self.episode_success_buf)
        # print("episode track buf:", self.episode_track_buf)
        # print(self._track_rewards_arm)
        # print(track_rewards_arm.mean())

        # self.extras["log"] = {
        #     "Episodic Return Arm Mean": track_rewards_arm.mean(),
        #     "Episodic Return Hand Mean": track_rewards_hand.mean(),
        #     "Success Rate": track_success_rate.mean(),
        #     "Track Rate": track_track_rate.mean(),
        # }

        # self.extras["log"] = {
        #     # "success_rate_window": self.sliding_success_rate,
        #     # "track_rate_window": self.sliding_track_rate,
        #     # "track_rate_ema": self.consecutive_tracked,
        #     # "success_rate_ema": self.consecutive_successes,
        #     # "track_rate_total": self.avg_track,
        #     # "success_rate_total": self.avg_success,
        #     "arm_reward_mean": total_reward["arm"].mean(),
        #     "hand_reward_mean": total_reward["hand"].mean(),
        #     # "obj_dist_reward_arm_mean": reward_component["obj_dist_reward_arm"].mean(),
        #     # "obj_dist_reward_hand_mean": reward_component["obj_dist_reward_hand"].mean(),
        # }

        return {"arm": total_reward["arm"], "hand": total_reward["hand"]}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.Catcher._ALL_INDICES    

        super()._reset_idx(env_ids)
        # reset observation buffers
        self.arm_obs_buf[env_ids] = torch.zeros((self.cfg.num_obs_arm*(self.cfg.num_stacks)),device=self.device)
        self.hand_obs_buf[env_ids] = torch.zeros((self.cfg.num_obs_hand*(self.cfg.num_stacks)),device=self.device)

        # reset states buffers
        self.states_seq[env_ids] = torch.zeros((self.cfg.num_states_frame*(self.cfg.num_stacks)),device=self.device)

        # set right robot states
        joint_pos = self.Catcher.data.default_joint_pos[env_ids] + sample_uniform(
            -0.125,
            0.125,
            (len(env_ids), self.Catcher.num_joints),
            self.device,
        )

        joint_pos = torch.clamp(joint_pos, self.dof_lower_limits, self.dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.Catcher.set_joint_position_target(joint_pos, env_ids=env_ids)
        self.Catcher.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.dof_targets[env_ids,:] = joint_pos
        self.prev_targets[env_ids,:] = joint_pos
        self.cur_joint_pos[env_ids,:] = joint_pos

        # object state
        object_default_state = self._object.data.default_root_state.clone()[env_ids]
        random_pos, random_vel, release_delay = self.get_object_random_pose(env_ids=env_ids)

        self.object_spawn_pos[env_ids] = random_pos
        self.object_throw_vel[env_ids] = random_vel
        self.object_release_delay[env_ids] = release_delay

        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + random_pos + self.scene.env_origins[env_ids]
        )
        object_quat = quat_from_euler_xyz(torch.rand(len(env_ids))*math.pi*2,torch.rand(len(env_ids))*math.pi*2,torch.rand(len(env_ids))*math.pi*2)
        object_default_state[:, 3:7] = object_quat.to(self.device)
        self.object_spawn_rot[env_ids] = object_quat.to(self.device)
        # object_default_state[:, 7:10] = (
        #     object_default_state[:, 7:10] + random_vel
        # )
        self._object.write_root_state_to_sim(object_default_state, env_ids)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values()

        # define previous distance
        self.prev_dist_hand_obj[env_ids] = torch.norm(self.hand_pos[env_ids].clone() - self.object_pos[env_ids].clone(), p=2, dim=-1)
        self.prev_dist_thumb_obj[env_ids] = torch.norm(self.thumb_tip_pos[env_ids].clone() - self.object_pos[env_ids].clone(), p=2, dim=-1)
        self.prev_dist_index_obj[env_ids] = torch.norm(self.index_tip_pos[env_ids].clone() - self.object_pos[env_ids].clone(), p=2, dim=-1)
        self.prev_dist_middle_obj[env_ids] = torch.norm(self.middle_tip_pos[env_ids].clone() - self.object_pos[env_ids].clone(), p=2, dim=-1)
        self.prev_dist_ring_obj[env_ids] = torch.norm(self.ring_tip_pos[env_ids].clone() - self.object_pos[env_ids].clone(), p=2, dim=-1)
        self.prev_object_pos[env_ids] = self.object_pos[env_ids].clone()
        self.prev_hand_pos[env_ids] = self.hand_pos[env_ids].clone()

        # define buffers
        self.track_step[env_ids] = 0
        self.catch_step[env_ids] = 0
        self.miss_step[env_ids] = 0
        self.successes[env_ids] = 0
        self.tracked[env_ids] = 0

        self.object_released[env_ids] = torch.zeros((len(env_ids)), dtype=torch.bool, device=self.device)

        # reset action buffers
        self.actions["arm"][env_ids] = self.actions["arm"][env_ids].zero_()
        self.actions["hand"][env_ids] = self.actions["hand"][env_ids].zero_()
        self.applied_actions["arm"][env_ids] = self.applied_actions["arm"][env_ids].zero_()
        self.applied_actions["hand"][env_ids] = self.applied_actions["hand"][env_ids].zero_()
        self.pixel_cnt[env_ids] = self.pixel_cnt[env_ids].zero_()
        self.prev_cnt[env_ids] = self.prev_cnt[env_ids].zero_()
        
        # reset episodic return buffers
        self.cumulative_reward_arm[env_ids] = self.cumulative_reward_arm[env_ids].zero_()
        self.cumulative_reward_hand[env_ids] = self.cumulative_reward_hand[env_ids].zero_()

        # reset success buffer
        self.episode_success_buf[env_ids] = self.episode_success_buf[env_ids].zero_()
        self.episode_track_buf[env_ids] = self.episode_track_buf[env_ids].zero_()

        # reset latency buffers
        # self.latency_arm_buf[env_ids].zero_()
        # self.latency_hand_buf[env_ids].zero_()
        # self.latency_arm_action_buf[env_ids].zero_()
        # self.latency_hand_action_buf[env_ids].zero_()
        self.latency_pf_buf[env_ids] = self.latency_pf_buf[env_ids].zero_()
        # self.current_obs_latency[env_ids] = torch.randint(
        #     0, self.max_obs_latency_step, (len(env_ids),), device=self.device
        # )
        # self.current_act_latency[env_ids] = torch.randint(
        #     0, self.max_action_latency_step, (len(env_ids),), device=self.device
        # )
        self.current_pf_latency[env_ids] = torch.randint(
            1, self.max_pf_latency_step, (len(env_ids),), device=self.device
        )

    def get_object_random_pose(self, env_ids: torch.Tensor | None):
        g = 9.81
        # 1. Set Start Position
        Xs = torch.rand(len(env_ids), device=self.device) * 0.5 + 2.4 # 2.4 ~ 2.9m
        # Ys = torch.rand(len(env_ids), device=self.device) * 0.45 - 0.225
        Ys = torch.rand(len(env_ids), device=self.device) * 0.6 - 0.3 # -0.3 ~ 0.3m
        Zs = (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0) * 0.05 + 0.8

        # 2. Set Target XY Position (Robot is around x=0.5)
        # xT = torch.rand(len(env_ids), device=self.device) * 0.4 + 0.6 #0.6
        xT = torch.rand(len(env_ids), device=self.device) * 0.2 + 0.65 #0.6
        # yT = torch.rand(len(env_ids), device=self.device) * 0.45 - 0.225
        yT = torch.rand(len(env_ids), device=self.device) * 0.7 - 0.35 # -0.35 ~ 0.35m
        
        # [異붽�] 濡쒕큸�� 怨듭쓣 �≪쓣 �덉긽 �믪씠 (Target Z) �ㅼ젙
        # 濡쒕큸 �� �믪씠�� �묒뾽 怨듦컙 �믪씠瑜� 怨좊젮�� 0.8~1.0m �뺣룄濡� �ㅼ젙
        z_catch = torch.rand(len(env_ids), device=self.device) * 0.2 + 0.9 
        # -------------------------------------------------------------------
        # Logic: Fix Peak Height -> Calculate Time & Velocity based on Gravity
        # -------------------------------------------------------------------

        # 3. Set Max Peak Height (Randomly between 1.6m ~ 2.0m)
        z_peak_min = 1.6
        z_peak_max = 1.9
        target_peak = torch.rand(len(env_ids), device=self.device) * (z_peak_max - z_peak_min) + z_peak_min
        
        # Safety: Peak must be higher than start AND catch position (+ margin)
        max_start_catch = torch.maximum(Zs, z_catch)
        target_peak = torch.maximum(target_peak, max_start_catch + 0.2)

        # 4. Calculate Vertical Velocity (vz) to reach Peak
        # v_z0 = sqrt(2 * g * (H_peak - H_start))
        h_diff_up = target_peak - Zs
        v_lin_z = torch.sqrt(2.0 * g * h_diff_up)

        # 5. [�듭떖 �섏젙] Calculate Physics-based Flight Time (T)
        # 臾쇰━�곸쑝濡� �щ씪媛붾떎(t_up) �대젮�ㅻ뒗(t_down) �쒓컙�� 怨꾩궛�댁빞 ��
        # t_up = v_z0 / g
        # t_down = sqrt(2 * (H_peak - H_catch) / g)
        
        t_up = v_lin_z / g
        
        h_diff_down = target_peak - z_catch
        t_down = torch.sqrt(2.0 * h_diff_down / g)
        
        # �ㅼ젣 怨듭씠 紐⑺몴 吏���(z_catch)�� �꾨떖�섎뒗 珥� �쒓컙
        T_physics = t_up + t_down 

        # 6. Calculate Horizontal Velocity (vx, vy) using Physics Time
        # �댁젣 �� �띾룄濡� �섏�硫�, T_physics 珥� �ㅼ뿉 �뺥솗�� (xT, yT, z_catch)瑜� 吏��⑸땲��.
        dx = xT - Xs
        dy = yT - Ys
        
        v_lin_x = dx / T_physics
        v_lin_y = dy / T_physics

        # -------------------------------------------------------------------

        random_pos = torch.cat([Xs.unsqueeze(-1), Ys.unsqueeze(-1), Zs.unsqueeze(-1)], dim=-1)
        random_vel = torch.cat([v_lin_x.unsqueeze(-1), v_lin_y.unsqueeze(-1), v_lin_z.unsqueeze(-1)], dim=-1)
        release_delay = torch.randint(
            low=5, high=20, size=(len(env_ids),), device=self.device, dtype=torch.long
            # low=10, high=60, size=(len(env_ids),), device=self.device, dtype=torch.long
        )

        return random_pos, random_vel, release_delay

    def _get_observations(self) -> dict:
        # data_type = "semantic_segmentation"
        # center_camera_data = self.center_camera.data.output[data_type]
        # num_envs, H, W, C = center_camera_data.shape # (num_envs, 480, 640, 3)
        # # save_images_to_file(center_camera_data/255, "/home/kimsy/valid/img_%d.png"%self.episode_length_buf)

        # center_mask = self.mask_green(center_camera_data[...,:3])
        # self.center_bbox, self.center_bbox_info, center_bbox_for_vis = self.find_bounding_boxes(center_mask)
        # # self.save_bboxes_with_rgb(center_camera_data, center_bbox_for_vis, dark_factor=0.3)
        # # save_images_to_file(center_camera_data/255, "/home/kimsy/valid/img_%d.png"%self.episode_length_buf)

        # delta_center = self.center_bbox_info[:,:2].clone() - self.prev_bbox_info[:,:2].clone()
        # delta_len = self.center_bbox_info[:,2:].clone() - self.prev_bbox_info[:,2:].clone()

        # self.norm_delta_center[:,0] = delta_center[:,0].clone()
        # self.norm_delta_center[:,1] = delta_center[:,1].clone()
        # self.norm_delta_len[:,0] = delta_len[:,0].clone()
        # self.norm_delta_len[:,1] = delta_len[:,1].clone()
        # self.norm_bbox_center_x = self.center_bbox_info[:,0].clone()
        # self.norm_bbox_center_y = self.center_bbox_info[:,1].clone()

        # # update the previous bounding box info
        # self.prev_bbox_info = self.center_bbox_info.clone()
        # self.prev_bbox = self.center_bbox.clone()
        
        data_type = "rgb" if "rgb" in self.cfg.center_camera.data_types else "depth"
        if "rgb" in self.cfg.center_camera.data_types:
            center_camera_data = self.center_camera.data.output[data_type]
            num_envs, H, W, C = center_camera_data.shape # (num_envs, 480, 640, 3)

            center_mask = self.mask_green(center_camera_data)

            # self.pixel_cnt = center_mask.sum(dim=(1,2)).to(dtype=torch.float32)
            # noise = torch.randint(low=-50, high=50, size=(self.num_envs,), device=self.device, dtype=torch.float32)
            # self.pixel_cnt = torch.clamp(self.pixel_cnt + noise, min=0.0)
            self.center_bbox_info, center_bbox_for_vis = self.find_bounding_boxes(center_mask)

            self.delta_center = self.center_bbox_info[:,:2].clone() - self.prev_bbox_info[:,:2].clone()
            self.delta_len = self.center_bbox_info[:,2:].clone() - self.prev_bbox_info[:,2:].clone()

            self.norm_center_x = self.center_bbox_info[:,0].clone()/W
            self.norm_center_y = self.center_bbox_info[:,1].clone()/H
            self.norm_len_w = self.center_bbox_info[:,2].clone()/W
            self.norm_len_h = self.center_bbox_info[:,3].clone()/H

            # self.delta_size = self.pixel_cnt - self.prev_cnt
            # self.norm_pixel_cnt = self.pixel_cnt / (H*W)
            
            cur_pf = torch.cat([
                self.norm_center_x.unsqueeze(-1).clone(),
                self.norm_center_y.unsqueeze(-1).clone(),
                # self.norm_len_w.unsqueeze(-1).clone(),
                # self.norm_len_h.unsqueeze(-1).clone(),
                # self.norm_pixel_cnt.unsqueeze(-1).clone(),
                self.delta_center,
                self.delta_len,
                # self.delta_size.unsqueeze(-1).clone(),
            ],dim=-1).view(self.num_envs, -1)

            self.latency_pf_buf = torch.roll(self.latency_pf_buf, shifts=1, dims=1)
            self.latency_pf_buf[:,0] = cur_pf
            cur_pf = self.latency_pf_buf[
                torch.arange(self.num_envs, device=self.device),
                self.current_pf_latency
            ]

        self.obs_pf = cur_pf.clone()

        self.delta_hand_pos = self.hand_pos - self.prev_hand_pos
        self.delta_object_pos = self.object_pos - self.prev_object_pos

        self.compute_arm_obs()
        self.compute_hand_obs()

        # update the previous bounding box info
        self.prev_bbox_info = self.center_bbox_info.clone()
        self.prev_cnt = self.pixel_cnt.clone()
        self.prev_hand_pos = self.hand_pos.clone()
        self.prev_object_pos = self.object_pos.clone()

        observations = {"arm": self.arm_obs_buf,
                        "hand": self.hand_obs_buf}

        return observations

    def mask_green(self, camera_data):
        # lower_bound = torch.tensor([120, 200, 0], dtype=torch.uint8, device=self.device)
        # upper_bound = torch.tensor([160, 255, 60], dtype=torch.uint8, device=self.device)
        lower_bound = torch.tensor([0,120,0], dtype=torch.uint8, device=self.device)
        upper_bound = torch.tensor([100,244,100], dtype=torch.uint8, device=self.device)

        mask = (camera_data >= lower_bound) & (camera_data <= upper_bound)
        mask = mask.all(dim=-1)

        return mask.to(dtype=torch.uint8)

    def find_bounding_boxes(self, mask):
        """
        踰≫꽣��(Vectorized)�� Bounding Box 寃�異� �⑥닔
        - for 猷⑦봽 �쒓굅
        - 留덉뒪�� �ъ쁺(Projection) 諛⑹떇�� �ъ슜�섏뿬 GPU �곸뿉�� 怨좎냽 �곗궛
        """
        num_envs, H, W = mask.shape
        device = mask.device

        # 1. �쎌� 議댁옱 �щ� �뺤씤 (媛� �섍꼍蹂�)
        # (num_envs,) : �쎌��� �섎굹�쇰룄 �덉쑝硫� True
        # dim=(1,2)�� H, W 李⑥썝�� 紐⑤몢 �⑹퀜�� 怨꾩궛
        has_pixels = mask.any(dim=-1).any(dim=-1)

        # 2. �ъ쁺(Projection)�� �댁슜�� Min/Max 醫뚰몴 怨좎냽 �먯깋
        # 留덉뒪�щ� X異�, Y異뺤쑝濡� �뺤텞�섏뿬 1D濡� 留뚮벀
        y_proj = mask.any(dim=2).int() # (num_envs, H)
        x_proj = mask.any(dim=1).int() # (num_envs, W)

        # argmax�� '媛��� 癒쇱� �깆옣�섎뒗 理쒕�媛�(1)'�� �몃뜳�ㅻ� 諛섑솚 -> Min 醫뚰몴
        ymin = y_proj.argmax(dim=1)
        xmin = x_proj.argmax(dim=1)

        # �ㅼ쭛�댁꽌 argmax瑜� �섎㈃ '�ㅼ뿉�� 媛��� 癒쇱� �깆옣�섎뒗 1'�� 李얠쓬 -> Max 醫뚰몴
        # (H - 1) - index 濡� �먮옒 醫뚰몴怨꾨줈 蹂���
        ymax = (H - 1) - y_proj.flip(dims=[1]).argmax(dim=1)
        xmax = (W - 1) - x_proj.flip(dims=[1]).argmax(dim=1)

        # 3. �몄씠利� 異붽� (Batch 泥섎━)
        # (num_envs, 4) : [ymin_noise, xmin_noise, ymax_noise, xmax_noise]
        coord_noise = torch.randint(low=-5, high=5, size=(num_envs, 4), device=device)

        ymin = ymin + coord_noise[:, 0]
        xmin = xmin + coord_noise[:, 1]
        ymax = ymax + coord_noise[:, 2]
        xmax = xmax + coord_noise[:, 3]

        # 4. 醫뚰몴 �대옩�� (�대�吏� 踰붿쐞 踰쀬뼱�섏� �딅룄濡�)
        ymin = torch.clamp(ymin, 0, H - 1).float()
        xmin = torch.clamp(xmin, 0, W - 1).float()
        ymax = torch.clamp(ymax, 0, H - 1).float()
        xmax = torch.clamp(xmax, 0, W - 1).float()

        # 5. bbox �뺣낫 怨꾩궛 (Center X, Center Y, Width, Height)
        x_c = (xmin + xmax) / 2.0
        y_c = (ymin + ymax) / 2.0
        w = (xmax - xmin)
        h = (ymax - ymin)
        
        # (num_envs, 4)
        current_bbox_info = torch.stack([x_c, y_c, w, h], dim=1)

        # 6. �쒓컖�붿슜 bbox 醫뚰몴 援ъ꽦 (num_envs, 4, 2)
        # �쒖꽌: [xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]
        bboxes_for_vis = torch.stack([
            torch.stack([xmin, ymin], dim=-1),
            torch.stack([xmax, ymin], dim=-1),
            torch.stack([xmin, ymax], dim=-1),
            torch.stack([xmax, ymax], dim=-1)
        ], dim=1)

        # 7. �쎌��� �녿뒗 �섍꼍(has_pixels=False) 泥섎━
        # �쎌��� �놁쑝硫� �댁쟾 �ㅽ뀦�� bbox �뺣낫瑜� �좎� (torch.where �ъ슜)
        # has_pixels�� (N,) �대�濡� broadcasting�� �꾪빐 李⑥썝 �뺤옣 �꾩슂
        
        # bbox_info �낅뜲�댄듃
        final_bbox_info = torch.where(
            has_pixels.unsqueeze(-1), 
            current_bbox_info, 
            self.prev_bbox_info
        )
        
        # �쒓컖�붿슜 bbox�� �쎌��� �놁쑝硫� 0�쇰줈 �먭굅�� �댁쟾 寃껋쓣 �� �� �덉쓬.
        # 湲곗〈 濡쒖쭅: "else: bbox_info[i] = self.prev_bbox_info[i]" (vis�� �멸툒 �놁쑝�� 蹂댄넻 0 �먮뒗 �댁쟾媛�)
        # �ш린�쒕뒗 �덉쟾�섍쾶 0�쇰줈 泥섎━�섍굅��, �꾩슂�� prev_bbox_vis 蹂��섎� 留뚮뱾�� �좎��� �� �덉쓬.
        # �쇰떒 媛먯�媛� �덈릺硫� �쒓컖�� 諛뺤뒪�� (0,0,0,0)�쇰줈 蹂대궡�� 寃껋씠 �붾쾭源낆뿉 �좊━�섎�濡� 0�쇰줈 ��.
        # 留뚯빟 諛뺤뒪瑜� �좎��섍퀬 �띕떎硫� �대옒�� 硫ㅻ쾭蹂��� self.prev_bbox_vis媛� �꾩슂��.
        final_bboxes_for_vis = torch.where(
            has_pixels.unsqueeze(-1).unsqueeze(-1),
            bboxes_for_vis,
            torch.zeros_like(bboxes_for_vis) 
        )

        return final_bbox_info, final_bboxes_for_vis

    def _get_states(self) -> torch.Tensor:       
        states = torch.cat(
            (
                # hand position (3)
                self.hand_pos,
                # delta hand position (3)
                self.delta_hand_pos,
                # hand orientation (4)
                self.hand_rot,
                # DOF positions (19)
                self.cur_joint_pos[:,:],
                # current target DOF positions (19)
                self.cur_targets[:,:],
                # applied actions on arm (6)
                self.applied_actions["arm"],
                # applied actions on hand (13)
                self.applied_actions["hand"],
                # pixel-wise features (4)
                self.obs_pf,
                # object position (3)
                self.object_pos,
                # delat object position (3)
                self.delta_object_pos,
                # relative position between hand and object (3)
                self.relative_hand_obj_pos,
            ),
            dim=-1
        )
        # print("states: ", states.shape)
        self.states_seq = torch.cat((states,self.states_seq), dim=-1)
        if self.states_seq.shape[-1] > self.cfg.num_states_frame*(self.cfg.num_stacks):
            self.states_seq = self.states_seq[:,:self.cfg.num_states_frame*(self.cfg.num_stacks)]

        return self.states_seq

    def compute_arm_obs(self):
        current_obs_buf = torch.cat(
            (
                # eef position (3)
                self.hand_pos,
                # delta hand position (3)
                self.delta_hand_pos,
                # eef orientation (4)
                self.hand_rot,
                # DOF position (6)
                self.cur_joint_pos[:,:6],
                # current target DOF position (6)
                self.cur_targets[:,self.dof_indices[:6]],
                # # applied actions on arm (6)
                self.applied_actions["arm"],
                # pixel-wise features (6)
                self.obs_pf,
            ),
            dim=-1,
        )
        # print("current_arm_obs_buf: ", current_obs_buf.shape)

        # self.latency_arm_buf = torch.roll(self.latency_arm_buf, shifts=1, dims=1)
        # self.latency_arm_buf[:,0] = current_obs_buf
        # env_indices = torch.arange(self.num_envs, device=self.device)
        # delayed_obs = self.latency_arm_buf[env_indices, self.current_obs_latency]
        # self.arm_obs_buf = torch.cat((delayed_obs,self.arm_obs_buf[...,:-self.cfg.num_obs_arm]), dim=-1)

        self.arm_obs_buf = torch.cat((current_obs_buf,self.arm_obs_buf), dim=-1)
        if self.arm_obs_buf.shape[-1] > self.cfg.num_obs_arm*(self.cfg.num_stacks):
            self.arm_obs_buf = self.arm_obs_buf[:,:self.cfg.num_obs_arm*(self.cfg.num_stacks)]

    def compute_hand_obs(self):
        current_obs_buf = torch.cat(
            (
                # eef possition (3)
                self.hand_pos,
                # delta hand position (3)
                self.delta_hand_pos,
                # eef orientation (4)
                self.hand_rot,
                # DOF position (13)
                self.cur_joint_pos[:,6:],
                # current target DOF position (13)
                self.cur_targets[:,self.dof_indices[6:]],
                # applied actions on hand (13)
                self.applied_actions["hand"],
                # pixel-wise features (4)
                self.obs_pf,
            ),
            dim=-1,
        )
        # print("current_hand_obs_buf: ", current_obs_buf.shape)
        
        # self.latency_hand_buf = torch.roll(self.latency_hand_buf, shifts=1, dims=1)
        # self.latency_hand_buf[:,0] = current_obs_buf
        # env_indices = torch.arange(self.num_envs, device=self.device)
        # delayed_obs = self.latency_hand_buf[env_indices, self.current_obs_latency]
        # self.hand_obs_buf = torch.cat((delayed_obs,self.hand_obs_buf[...,:-self.cfg.num_obs_hand]), dim=-1)
        
        self.hand_obs_buf = torch.cat((current_obs_buf,self.hand_obs_buf), dim=-1)
        if self.hand_obs_buf.shape[-1] > self.cfg.num_obs_hand*(self.cfg.num_stacks):
            self.hand_obs_buf = self.hand_obs_buf[:,:self.cfg.num_obs_hand*(self.cfg.num_stacks)]

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self.Catcher._ALL_INDICES

        # get right robot states
        self.hand_pos = self.Catcher.data.body_pos_w[:, self.palm_link_idx] - self.scene.env_origins
        self.hand_rot = self.Catcher.data.body_quat_w[:, self.palm_link_idx]
        self.hand_rot = self.hand_rot * torch.where(self.hand_rot[:, 0] < 0.0, -1.0, 1.0).unsqueeze(1)
        self.cur_joint_pos = self.Catcher.data.joint_pos
        self.cur_joint_vel = self.Catcher.data.joint_vel

        self.wrist3_pos = self.Catcher.data.body_pos_w[:, self.wrist_3_link_idx] - self.scene.env_origins
        
        # get object position
        self.object_pos = self._object.data.root_pos_w - self.scene.env_origins

        self.relative_hand_obj_pos = self.object_pos - self.hand_pos

        # marker_hand_pos = self.Catcher.data.body_pos_w[:, self.index_tip_idx]
        # marker_hand_rot = self.Catcher.data.body_quat_w[:, self.index_tip_idx]
        # self.palm_marker.visualize(marker_hand_pos, marker_hand_rot)

        table_f = self.scene["table_contact"].data.net_forces_w
        self.table_contact = (torch.norm(table_f[:,0,:2],dim=-1) > 1.0)

        # get hand position
        self.thumb_tip_pos = self.Catcher.data.body_pos_w[:, self.thumb_tip_idx] - self.scene.env_origins
        self.index_tip_pos = self.Catcher.data.body_pos_w[:, self.index_tip_idx] - self.scene.env_origins
        self.middle_tip_pos = self.Catcher.data.body_pos_w[:, self.middle_tip_idx] - self.scene.env_origins
        self.ring_tip_pos = self.Catcher.data.body_pos_w[:, self.ring_tip_idx] - self.scene.env_origins

@torch.jit.script
def compute_rewards(
    reset_buf: Tensor,
    episode_length: Tensor,
    max_episode_length: int,
    object_pos: Tensor,
    hand_pos: Tensor,
    thumb_tip_pos: Tensor,
    index_finger_tip_pos: Tensor,
    middle_tip_pos: Tensor,
    ring_tip_pos: Tensor,
    prev_dist_hand_obj: Tensor,
    prev_dist_thumb_obj: Tensor,
    prev_dist_index_obj: Tensor,
    prev_dist_middle_obj: Tensor,
    prev_dist_ring_obj: Tensor,
    actions: dict[str, Tensor],
    dof_pos: Tensor,
    drop_penalty_score: float,
    track_step: Tensor,
    catch_step: Tensor,
    miss_step: Tensor,
    table_contact: Tensor,
    # successes: Tensor,
    # consecutive_success: Tensor,
    # tracked: Tensor,
    # consecutive_tracked: Tensor,
    av_factor: float,
    # delta_object_pos: Tensor,  # �� 異붽�
    # delta_hand_pos: Tensor,     # �� 異붽�
    dt: float                   # �� 異붽� (���꾩뒪��)
):
    # ========== 1. Distance Rewards ==========
    # xy_current_dist_hand_obj = torch.norm(hand_pos[:,:2] - object_pos[:,:2], p=2, dim=-1)
    # z_current_dist_hand_obj = torch.abs(hand_pos[:,2] - object_pos[:,2])
    # current_dist_hand_obj = 0.8*xy_current_dist_hand_obj + 0.2*z_current_dist_hand_obj
    current_dist_hand_obj = torch.norm(hand_pos - object_pos, p=2, dim=-1)
    # yz_current_dist_hand_obj = torch.norm(hand_pos[:,1:] - object_pos[:,1:], p=2, dim=-1)
    # x_current_dist_hand_obj = torch.abs(hand_pos[:,0] - object_pos[:,0])
    # current_dist_hand_obj = 0.7*yz_current_dist_hand_obj + 0.3*x_current_dist_hand_obj
    obj_dist_reward = prev_dist_hand_obj - current_dist_hand_obj

    # Fingertip distances
    current_dist_thumb_obj = torch.norm(thumb_tip_pos - object_pos, p=2, dim=-1)
    current_dist_index_obj = torch.norm(index_finger_tip_pos - object_pos, p=2, dim=-1)
    current_dist_middle_obj = torch.norm(middle_tip_pos - object_pos, p=2, dim=-1)
    current_dist_ring_obj = torch.norm(ring_tip_pos - object_pos, p=2, dim=-1)

    obj_fingertip_dist_reward = (
        (prev_dist_thumb_obj - current_dist_thumb_obj) + 
        (prev_dist_index_obj - current_dist_index_obj) + 
        (prev_dist_middle_obj - current_dist_middle_obj) + 
        (prev_dist_ring_obj - current_dist_ring_obj)
    )
    
    # ========== 5. Existing Rewards ==========
    is_approached = (current_dist_hand_obj < 0.08)
    app_bonus = torch.where(is_approached, torch.ones_like(is_approached, device=hand_pos.device), torch.zeros_like(is_approached, device=hand_pos.device))
    
    obj_dist_reward_arm = obj_dist_reward
    # obj_dist_reward_hand = 0.5 * obj_dist_reward + 0.5 * (obj_fingertip_dist_reward/4.0)
    obj_dist_reward_hand = (obj_dist_reward + obj_fingertip_dist_reward) / 5.0

    arm_action_penalty = torch.sum(actions["arm"]**2, dim=-1)
    hand_action_penalty = torch.sum(actions["hand"]**2, dim=-1)

    drop_penalty = torch.where(
        (object_pos[:,2] < 0.45)|(object_pos[:,0] < 0.03), 
        drop_penalty_score, 
        torch.tensor(0.0, device=reset_buf.device)
    )

    is_tracked = torch.where(track_step >= 1, torch.ones_like(track_step), torch.zeros_like(track_step))
    track_success = torch.where((is_tracked > 1) & (current_dist_hand_obj < 0.1), torch.ones_like(is_tracked), torch.zeros_like(track_step))

    is_catched = torch.where(catch_step >= 20, torch.ones_like(catch_step), torch.zeros_like(catch_step))
    # is_catched = torch.where((track_step >= 20)&(current_dist_hand_obj < 0.06), torch.ones_like(track_step), torch.zeros_like(catch_step))

    # arm_action_penalty = torch.where(is_catched > 0, arm_action_penalty * 50.0, arm_action_penalty)
    arm_action_penalty = torch.where(catch_step >= 5, arm_action_penalty * 50.0, arm_action_penalty)

    # goal_resets = torch.where((episode_length >= (max_episode_length - 1))&(current_dist_hand_obj < 0.06), torch.ones_like(is_catched), torch.zeros_like(is_catched))
    goal_resets = torch.where((is_catched > 0), torch.ones_like(is_catched), torch.zeros_like(is_catched))

    success_bonus = goal_resets * 10.0

    # ========== 6. Combined Rewards with Velocity Matching ==========
    r_time = -0.01
    r_dist = obj_dist_reward_arm
    r_app = app_bonus
    p_current_dist = current_dist_hand_obj
    p_act_arm = arm_action_penalty
    r_success = goal_resets
    p_drop = torch.where(
        (object_pos[:,2] < 0.45)|(object_pos[:,0] < 0.03), 
        1.0, 
        0.0
    )
    p_collision = torch.where(
        table_contact, 
        1.0, 
        0.0
    )
    p_miss = torch.where(
        miss_step >= 1,
        1.0,
        0.0
    )

    arm_reward = (
        r_time
        + r_dist
        - p_act_arm *0.01
        + r_success * 10.0
        + r_app * 0.1
        - p_drop * 5.0
        - p_collision * 5.0
        # - p_miss * 0.1
    )

    r_dist = obj_dist_reward_hand
    p_act_hand = hand_action_penalty

    hand_reward = (
        r_dist
        - p_act_hand *0.01
        # + r_app * 0.5
        + r_success * 10.0
        - p_drop * 5.0
        - p_collision * 5.0
    )

    total_reward = {
        "arm": arm_reward,
        "hand": hand_reward,
    }

    reward_component = {
        "obj_dist_reward_arm": obj_dist_reward_arm * 5.0,
        "obj_dist_reward_hand": obj_dist_reward_hand * 5.0,
        # "velocity_match_reward": velocity_match_reward,        # �� NEW
        # "direction_alignment_reward": direction_alignment_reward,  # �� NEW
        # "soft_impact_bonus": soft_impact_bonus,                # �� NEW
    }

    # drop_mask = (object_pos[:,2] < 0.45)|(object_pos[:,0] < 0.03)
    # timeout_mask = (episode_length >= (max_episode_length - 1))
    # success_mask = (is_catched > 0)&(current_dist_hand_obj < 0.06)
    # done_mask = drop_mask | timeout_mask | success_mask

    # episode_success = goal_resets
    # episode_success_f = episode_success.float()
    # episode_tracked_f = is_tracked.float() * done_mask.float()

    # successes = torch.where(done_mask, episode_success_f, successes)
    # tracked = torch.where(done_mask, episode_tracked_f, tracked)

    # num_resets = done_mask.float().sum()
    # sum_success = episode_success_f.sum()
    # sum_tracked = episode_tracked_f.sum()

    # cons_successes = torch.where(
    #     num_resets > 0,
    #     av_factor * (sum_success / num_resets) + (1.0 - av_factor) * consecutive_success,
    #     consecutive_success
    # )

    # cons_tracked = torch.where(
    #     num_resets > 0,
    #     av_factor * (sum_tracked / num_resets) + (1.0 - av_factor) * consecutive_tracked,
    #     consecutive_tracked
    # )

    return (
        total_reward,
        current_dist_hand_obj,
        current_dist_thumb_obj,
        current_dist_index_obj,
        current_dist_middle_obj,
        current_dist_ring_obj,
        # successes,
        # cons_successes,
        # tracked,
        # cons_tracked,
        reward_component
    )

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )