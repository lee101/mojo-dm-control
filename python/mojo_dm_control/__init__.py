"""Mojo port of the compute-bound parts of google-deepmind/dm_control.

Module layout mirrors dm_control's own package layout, so the two can be read
side by side:

    mojo_dm_control.mujoco.math                     <- dm_control.mujoco.math
    mojo_dm_control.utils.transformations           <- dm_control.utils.transformations
    mojo_dm_control.utils.rewards                   <- dm_control.utils.rewards
    mojo_dm_control.locomotion.tasks.reference_pose.rewards
    mojo_dm_control.utils.inverse_kinematics        <- dm_control.utils.inverse_kinematics
    mojo_dm_control.suite.lqr_solver                <- dm_control.suite.lqr_solver
    mojo_dm_control.rl.control                      <- dm_control.rl.control
    mojo_dm_control.composer.variation.rotations    <- dm_control.composer.variation.rotations
"""

__version__ = "0.1.0"
