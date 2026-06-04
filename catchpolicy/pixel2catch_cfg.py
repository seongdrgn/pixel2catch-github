import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectMARLEnvCfg, mdp
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file, ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.envs.common import ViewerCfg

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import NoiseModelWithAdditiveBiasCfg, GaussianNoiseCfg

import os
import torch
from typing import Sequence, Dict
import math
import numpy as np

# Directory of this package; used to resolve bundled asset paths in a
# location-independent way (works regardless of where the repo is cloned).
CATCHPOLICY_DIR = os.path.dirname(os.path.abspath(__file__))

##
# Pre-defined configs
##
from torch._tensor import Tensor

@configclass
class EventCfg:
    robot_joint_stiffness_and_damping = EventTerm(
        func = mdp.randomize_actuator_gains,
        mode = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("Catcher", joint_names=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]),
            "stiffness_distribution_params": (0.8,1.2),
            "damping_distribution_params": (0.8,1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    hand_joint_stiffness_and_damping = EventTerm(
        func = mdp.randomize_actuator_gains,
        mode = "reset",
        params = {
            "asset_cfg": SceneEntityCfg("Catcher", joint_names=["joint00","joint01","joint02","joint03",
                                                                "joint11","joint12","joint13",
                                                                "joint21","joint22","joint23",
                                                                "joint31","joint32","joint33",]),
            "stiffness_distribution_params": (0.7,1.3),
            "damping_distribution_params": (0.7,1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    object_mass = EventTerm(
        func = mdp.randomize_rigid_body_mass,
        mode="reset",
        params = {
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        }
    )
    
    object_restitution = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.5,1.0),
            "dynamic_friction_range": (1.0,1.5),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
      },
  )

@configclass
class DynamicCatchEnvCfg(DirectMARLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 3.0 #3.0
    # obs stack frames
    num_stacks = 2
    possible_agents = ["arm","hand"]
    num_obs_arm = 36-2
    num_obs_hand = 57-2
    num_states_frame = 84-2
    state_space = num_states_frame * num_stacks
    action_spaces = {"arm": 6, "hand": 13}
    observation_spaces = {"arm": num_obs_arm*num_stacks, "hand": num_obs_hand*num_stacks}

    # viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(3.3, 1.8, 2.5),
        lookat=(0.5, -0.5, 0.7),
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.5,
        )
    )

    # domain randomization
    events: EventCfg = EventCfg()

    # camera
    intrinsic_matrix = [618.163, 0, 328.451, 0, 618.468, 246.513, 0, 0, 1]
    width=640
    height=480

    center_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/center_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-0.39, 0.01, 2.2), rot=(0.96814764037, 0, 0.25038000405, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=width,
            height=height,
            clipping_range=(0.1, 10.0),
            focus_distance=1.0,
            f_stop=0.0,
            projection_type="pinhole"
        ),
        width=640,
        height=480,
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    arm_action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.03, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    )
    hand_action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.02, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    action_noise_model = {"arm": arm_action_noise_model, "hand": hand_action_noise_model}

    arm_observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    )

    hand_observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
      bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.001, operation="abs"),
    )

    observation_noise_model = {"arm": arm_observation_noise_model, "hand": hand_observation_noise_model}

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=8.0, replicate_physics=False)

    Catcher = ArticulationCfg(
        prim_path="/World/envs/env_.*/Catcher",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{CATCHPOLICY_DIR}/assets/allegroUR5e/ur5e/ur5e_allegro.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=10,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": math.pi,
                "shoulder_lift_joint": (2/5)*math.pi + math.pi - math.pi*2,
                "elbow_joint": (4/6)*math.pi,
                "wrist_1_joint": math.pi,
                "wrist_2_joint": -0.5*math.pi,
                "wrist_3_joint": math.pi,

                "joint11": 0.0,
                "joint12": 0.0,
                "joint13": 0.0,

                "joint21": 0.0,
                "joint22": 0.0,
                "joint23": 0.0,

                "joint31":0.0,
                "joint32":0.0,
                "joint33":0.0,

                "joint00":0.2630,
                "joint01":0.0,
                "joint02":0.0,
                "joint03":0.0,
            },
            pos=(0.0,0.0,0.821),
            rot=(0.0,0.0,0.0,1.0),
        ),
        actuators={
            "shoulder_pan_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 2504.0,
                damping= 342.0,
                friction= 0.51,
                armature=0.16
            ),
            "shoulder_lift_joint": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_lift_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 2848.0,
                damping= 353.0,
                friction= 0.42,
                armature=0.15
            ),
            "elbow_joint": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 2536.0,
                damping= 343.0,
                friction= 0.40,
                armature=0.14
            ),
            "wrist_1_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_1_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 1681.0,
                damping= 234.0,
                friction= 0.30,
                armature=0.23
            ),
            "wrist_2_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_2_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 2243.0,
                damping= 279.0,
                friction= 0.30,
                armature=0.20
            ),
            "wrist_3_joint": ImplicitActuatorCfg(
                joint_names_expr=["wrist_3_joint"],
                velocity_limit_sim=2.79253,
                effort_limit_sim=200.0,
                stiffness= 2066.0,
                damping= 202.0,
                friction= 0.25,
                armature=0.21
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["joint00","joint01","joint02","joint03",
                                  "joint11","joint12","joint13",
                                  "joint21","joint22","joint23",
                                  "joint31","joint32","joint33",],
                effort_limit_sim=0.5,
                stiffness=3.0,
                damping=0.1,
                friction=0.01,
            ),
        },
    )

    actuated_joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",

        "joint00",
        "joint01",
        "joint02",
        "joint03",

        "joint11",
        "joint12",
        "joint13",

        "joint21",
        "joint22",
        "joint23",

        "joint31",
        "joint32",
        "joint33",
    ]

    # markers
    palm_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/palm_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
            ),
        },
    )

    # Random Object
    '''for validation
    '''

    objects = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # normal cone
                sim_utils.ConeCfg(
                    radius=0.04,
                    height=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # large cone
                sim_utils.ConeCfg(
                    radius=0.045,
                    height=0.09,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # normal cylinder
                sim_utils.CylinderCfg(
                    radius=0.03,
                    height=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # large cylinder
                sim_utils.CylinderCfg(
                    radius=0.035,
                    height=0.07,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # normal cube
                sim_utils.CuboidCfg(
                    size=(0.05,0.05,0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # large cube
                sim_utils.CuboidCfg(
                    size=(0.06,0.06,0.06),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # normal sphere
                sim_utils.SphereCfg(
                    radius=0.035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # large sphere
                sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # normal capsule
                sim_utils.CapsuleCfg(
                    radius=0.025,
                    height=0.04,
                    axis='Z',
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
                # large capsule
                sim_utils.CapsuleCfg(
                    radius=0.03,
                    height=0.05,
                    axis='Z',
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,1.0,0.0), metallic=0.2),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="max",
                       restitution_combine_mode="max",
                    )
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32, 
                solver_velocity_iteration_count=4, 
                max_depenetration_velocity=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0001,
            ),
            semantic_tags= [("class", "object"),("color", "green")]
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0,0.0,0.0),
            rot=(0, 0, 0.7071068, 0.7071068)),
    )

    # table
    table_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{CATCHPOLICY_DIR}/assets/table.usd",
            scale=(1.0,1.0,1.1),
            mass_props=sim_utils.MassPropertiesCfg(mass=500000.0),
            activate_contact_sensors=True
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.05, 0.0, 0.81),
            rot=(0.0, 0.0, 0.0, -1.0)
        ),
    )
    table_contact = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Table/.*",
        update_period=0,
        history_length=1,
    )

    # ground plane
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
        )
    )

    # reward scales
    drop_penalty = 1.0
    object_height_threshold = 0.835
    success_tolerance = 0.05
    act_moving_average = 1.0
    avg_factor = 0.1 #0.01

    pf_queue_len = 2