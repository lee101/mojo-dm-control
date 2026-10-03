"""Parity of mojo_dm_control.utils.inverse_kinematics against upstream."""

import numpy as np
import pytest

from dm_control import suite
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as upstream
from mojo_dm_control.utils import inverse_kinematics

mjlib = mjbindings.mjlib

TOL = dict(rtol=1e-10, atol=1e-12)

SITE = 'touch_toe'


@pytest.fixture(scope='module')
def hopper():
    return suite.load('hopper', 'stand')


@pytest.fixture
def jacobians():
    rng = np.random.default_rng(6)
    jac = rng.normal(size=(6, 9))
    return jac, rng.normal(size=6)


def test_nullspace_method(jacobians):
    jac, delta = jacobians
    np.testing.assert_allclose(
        inverse_kinematics.nullspace_method(jac, delta),
        upstream.nullspace_method(jac, delta), **TOL)


def test_nullspace_method_positional_only(jacobians):
    jac, delta = jacobians
    np.testing.assert_allclose(
        inverse_kinematics.nullspace_method(jac[:3], delta[:3]),
        upstream.nullspace_method(jac[:3], delta[:3]), **TOL)


@pytest.mark.parametrize('strength', [1e-4, 3e-2, 1.0])
def test_nullspace_method_regularized(jacobians, strength):
    jac, delta = jacobians
    np.testing.assert_allclose(
        inverse_kinematics.nullspace_method(jac, delta,
                                            regularization_strength=strength),
        upstream.nullspace_method(jac, delta, regularization_strength=strength),
        **TOL)


def test_nullspace_method_uses_the_gram_matrix(jacobians):
    """The kernel must reproduce `jac.T @ jac` and `jac.T @ delta`."""
    jac, delta = jacobians
    hess = jac.T @ jac
    joint_delta = jac.T @ delta
    # `lstsq(hess, joint_delta)` is the unregularized branch upstream takes.
    np.testing.assert_allclose(
        inverse_kinematics.nullspace_method(jac, delta),
        np.linalg.lstsq(hess, joint_delta, rcond=-1)[0], **TOL)
    # And `solve(hess + s I, joint_delta)` the regularized one.
    np.testing.assert_allclose(
        inverse_kinematics.nullspace_method(jac, delta,
                                            regularization_strength=0.25),
        np.linalg.solve(hess + 0.25 * np.eye(9), joint_delta), **TOL)


def test_qpos_from_site_pose_reaches_target(hopper):
    physics = hopper.physics
    start = physics.named.data.site_xpos[SITE].copy()
    target = start + np.array([0.01, 0.0, 0.02])

    result = inverse_kinematics.qpos_from_site_pose(
        physics, SITE, target_pos=target, tol=1e-8, max_steps=200)

    assert result.success
    assert result.err_norm < 1e-8
    # `inplace=False` must not have disturbed the caller's physics.
    np.testing.assert_allclose(physics.named.data.site_xpos[SITE], start,
                               atol=1e-12)

    achieved = physics.copy(share_model=True)
    achieved.data.qpos[:] = result.qpos
    mjlib.mj_fwdPosition(achieved.model.ptr, achieved.data.ptr)
    np.testing.assert_allclose(achieved.named.data.site_xpos[SITE], target,
                               atol=1e-6)


def test_qpos_from_site_pose_matches_upstream(hopper):
    physics = hopper.physics
    start = physics.named.data.site_xpos[SITE].copy()
    target = start + np.array([0.015, 0.0, 0.01])
    kwargs = dict(target_pos=target, tol=1e-8, max_steps=200)

    ours = inverse_kinematics.qpos_from_site_pose(physics, SITE, **kwargs)
    theirs = upstream.qpos_from_site_pose(physics, SITE, **kwargs)
    np.testing.assert_allclose(ours.qpos, theirs.qpos, **TOL)
    assert ours.success == theirs.success
    assert ours.steps == theirs.steps
    np.testing.assert_allclose(ours.err_norm, theirs.err_norm, **TOL)


def test_qpos_from_site_pose_in_place(hopper):
    physics = hopper.physics.copy(share_model=True)
    before = physics.data.qpos.copy()
    target = physics.named.data.site_xpos[SITE].copy() + np.array([0.01, 0, 0.01])
    result = inverse_kinematics.qpos_from_site_pose(
        physics, SITE, target_pos=target, tol=1e-8, max_steps=200, inplace=True)
    assert result.success
    # `inplace=True` must have written the solution back into `physics`.
    np.testing.assert_allclose(physics.data.qpos, result.qpos)
    assert not np.allclose(physics.data.qpos, before)


def test_qpos_from_site_pose_with_orientation(hopper):
    """Rotation-only IK; this checks parity, upstream halts early here."""
    physics = hopper.physics
    site_xquat = np.empty(4)
    mjlib.mju_mat2Quat(site_xquat, physics.named.data.site_xmat[SITE])
    rng = np.random.default_rng(7)
    nudge = rng.normal(size=4)
    nudge /= np.linalg.norm(nudge)
    target_quat = site_xquat + 0.02 * nudge

    ours = inverse_kinematics.qpos_from_site_pose(
        physics, SITE, target_quat=target_quat, tol=1e-6, max_steps=200)
    theirs = upstream.qpos_from_site_pose(
        physics, SITE, target_quat=target_quat, tol=1e-6, max_steps=200)
    np.testing.assert_allclose(ours.qpos, theirs.qpos, **TOL)
    assert ours.success == theirs.success
    assert ours.steps == theirs.steps
    np.testing.assert_allclose(ours.err_norm, theirs.err_norm, **TOL)


def test_qpos_from_site_pose_with_joint_subset(hopper):
    physics = hopper.physics
    target = physics.named.data.site_xpos[SITE].copy() + np.array([0.01, 0, 0.01])
    kwargs = dict(target_pos=target, joint_names=['knee'], tol=1e-6,
                  max_steps=200)
    ours = inverse_kinematics.qpos_from_site_pose(physics, SITE, **kwargs)
    theirs = upstream.qpos_from_site_pose(physics, SITE, **kwargs)
    np.testing.assert_allclose(ours.qpos, theirs.qpos, **TOL)
    # Only the knee may have moved.
    adr = int(np.asarray(physics.named.model.jnt_qposadr['knee']))
    moved = np.nonzero(np.abs(ours.qpos - physics.data.qpos))[0]
    assert set(moved) <= set(range(adr, adr + 1))


def test_qpos_from_site_pose_requires_a_target(hopper):
    with pytest.raises(ValueError):
        inverse_kinematics.qpos_from_site_pose(hopper.physics, SITE)


def test_qpos_from_site_pose_rejects_bad_joint_names(hopper):
    with pytest.raises(ValueError):
        inverse_kinematics.qpos_from_site_pose(
            hopper.physics, SITE, target_pos=np.zeros(3), joint_names=SITE)


def test_nullspace_method_rejects_delta_of_the_wrong_length():
    """The kernel reads `delta` with the row count of `jac_joints`."""
    with pytest.raises(ValueError):
        inverse_kinematics.nullspace_method(np.zeros((6, 3)), np.zeros(5))


def test_ikresult_fields_match_upstream():
    ours = inverse_kinematics.IKResult(np.zeros(3), 0.0, 1, True)
    theirs = upstream.IKResult(np.zeros(3), 0.0, 1, True)
    assert ours._fields == theirs._fields
    np.testing.assert_array_equal(ours.qpos, theirs.qpos)
    assert (ours.err_norm, ours.steps, ours.success) == (
        theirs.err_norm, theirs.steps, theirs.success)
