from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Using Franka Panda with high PD gains for robust hybrid control
ROBOT_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
