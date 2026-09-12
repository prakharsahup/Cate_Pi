from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

TACTILE_SENSOR_CFG = VisuoTactileSensorCfg(
    # Prim path — one sensor per fingertip; use regex for both fingers
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/panda_leftfinger|{ENV_REGEX_NS}/Robot/panda_hand/panda_rightfinger",
    
    # Sensor output modes
    enable_camera_tactile=True,   # visuotactile RGB image → optional, for richer X_t
    enable_force_field=True,      # distributed normal + shear → primary X_t for CATE-π

    # Tactile array resolution — determines spatial token count in §4.3
    # 20×25 = 500 taxels per fingertip; comparable to TacVLA's 36-token setup
    # (your tactile encoder will pool this down to ~36 tokens)
    tactile_array_size=(20, 25),
    tactile_margin=0.003,         # elastomer boundary margin in metres

    # Contact object — randomised objects in the scene
    contact_object_prim_path_expr="{ENV_REGEX_NS}/object",

    # Force field physics parameters — tune against your sensor spec
    normal_contact_stiffness=1.0,
    friction_coefficient=2.0,
    tangential_stiffness=0.1,

    # Optional: camera for visuotactile RGB (can disable if only using force field)
    camera_cfg=TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/panda_leftfinger_cam|{ENV_REGEX_NS}/Robot/panda_hand/panda_rightfinger_cam",
        update_period=1 / 60,     # 60 Hz
        height=320,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=None,               # camera already spawned in USD
    ),
)

CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/panda_.*finger",
    track_air_time=False,
    max_contact_data_count=10,    # only need aggregate force, not full contact report
)
