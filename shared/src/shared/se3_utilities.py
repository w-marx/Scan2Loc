import numpy as np
from scipy.spatial.transform import Rotation

from shared.assertion_helpers import assert_homogeneous_mat, assert_homogeneous_mat_batch


def rotational_difference(hom1:np.ndarray, hom2:np.ndarray)->float:
    """
    Computes the rotational difference between two homogeneous 4x4 matrices
    :param hom1: homogeneous 4x4 matrix
    :param hom2: homogeneous 4x4 matrix
    :return rotational difference as float
    """
    assert assert_homogeneous_mat(hom1, size=4)
    assert assert_homogeneous_mat(hom2, size=4)

    return np.arccos(np.clip((np.trace(hom1[:3, :3] @ hom2[:3, :3].T) - 1) / 2, -1.0, 1.0))


def translational_difference(hom1:np.ndarray, hom2:np.ndarray)->float:
    """
    Computes the translational difference between two homogeneous 4x4 matrices
    :param hom1: homogeneous 4x4 matrix
    :param hom2: homogeneous 4x4 matrix
    :return translational euclidian difference as float
    """
    assert assert_homogeneous_mat(hom1, size=4)
    assert assert_homogeneous_mat(hom2, size=4)

    return float(np.linalg.norm(hom1[:3, 3] - hom2[:3, 3]))


def compute_pose_pseudo_median(poses:list[np.ndarray])->np.ndarray | None:
    """
    Takes a numpy array of poses and computes the median pose.
    To compute the median pose the median rotation and the geometric median of the translation are combined.
    Therefore, the returned pose may not be in poses
    :param poses: Nx4x4 numpy array of poses
    :return: median pose, as a 4x4 numpy array
    """
    if len(poses) == 0:
        return None
    assert all([assert_homogeneous_mat(m, size=4) for m in poses])

    median_pose = np.eye(4)
    median_pose[:3,3] = min(poses, key = lambda x: sum([np.linalg.norm(x[:3,3]-y[:3,3]) for y in poses]))[:3,3]
    median_pose[:3,:3] = min(poses, key = lambda x: sum([rotational_difference(x, y) for y in poses]))[:3, :3]
    return median_pose


def r_t_to_hom(r:np.ndarray, t:np.ndarray) -> np.ndarray:
    """
    Takes a rotation and translation vector and returns the homogeneous transformation matrix
    :param r: rotation matrix size: NxN
    :param t: translation vector size: N
    :return: N+1xN+1 numpy array
    """
    n = t.shape[0]
    assert r.shape == (n, n), f"R not {n}x{n}: {r.shape}"
    assert t.shape == (n,), f"T not {n}: {t.shape}"

    m = np.eye(n+1)
    m[0:n, 0:n] = r
    m[0:n, n] = t
    return m


def t_quat_to_hom(t:np.ndarray, quat:list[float]):
    """
    :param t: translation vector of the form: [tx, ty, tz]
    :param quat: quaternion of the form: [qx, qy, qz, qw]
    :return: 4x4 homogeneous transformation matrix
    """
    rotation = Rotation.from_quat(quat)
    return r_t_to_hom(r = rotation.as_matrix(), t = t)

def ate_rmse(predicted:np.ndarray, actual:np.ndarray) -> tuple[float, float]:
    """
    Computes the Root mean square error - Absolute Trajectory Error (RMSE-ATE)
    :param predicted: The predicted poses (Nx4x4) with each in SE3
    :param actual: The actual poses (Nx4x4), so that actual[i]~predicted[i]
    :return: The translational RMSE and the rotational RMSE
    """
    assert assert_homogeneous_mat_batch(predicted, size=4) and assert_homogeneous_mat_batch(actual, size=4)
    assert predicted.shape == actual.shape, f"Cant compute ATE between different pose sizes: {predicted.shape} != {actual.shape}"

    t_rmse = np.sqrt(np.mean(np.linalg.norm(predicted[:,:3, 3]- actual[:,:3, 3], axis = -1) **2))
    r_rmse = np.sqrt(np.mean(np.asarray([rotational_difference(m1, m2) for m1, m2 in zip(predicted, actual)])**2))
    return t_rmse, r_rmse

def rte_error_matrice_s(timestamps:list[int], predicted:np.ndarray, actual:np.ndarray) -> list[None | np.ndarray]:
    """
    Computes the homogeneous relative error for each consecutive prediction (needed for RTE)
    Uses timestamp_indices, assuming fixed Hz so if timestamp[i] != timestamp[i+1]-1, returns None
    :param timestamps: the timestamp index where predicted and actual were taken
    :param predicted: The predicted poses (Nx4x4) with each in SE3
    :param actual: The actual poses (Nx4x4), so that actual[i]~predicted[i]
    :return: A list of relative error matrices
    """
    assert assert_homogeneous_mat_batch(predicted, size=4) and assert_homogeneous_mat_batch(actual, size=4)
    assert len(timestamps) == predicted.shape[0] == actual.shape[0]

    if len(timestamps) < 2:
        return []

    error_matrices = []
    for i, (t_now, p_now, a_now) in enumerate(zip(timestamps[:-1], predicted[:-1], actual[:-1])):
        if t_now + 1 != timestamps[i+1]:
            error_matrices.append(None)
            continue

        error_matrices.append(
            np.linalg.inv((np.linalg.inv(p_now) @ predicted[i+1])) @ (np.linalg.inv(a_now) @ actual[i+1])
        )
    return error_matrices

def rte_translational_errors_rmse(
        timestamps:list[int],
        predicted:np.ndarray,
        actual:np.ndarray
    )->tuple[list[float], float]:
    """
    Computes translational RTEs (Relative Trajectory Error) and RMSE(RTEs)
    Uses timestamp_indices, assuming fixed Hz so if timestamp[i] != timestamp[i+1]-1, returns None
    :param timestamps: the timestamp index where predicted and actual were taken
    :param predicted: The predicted poses (Nx4x4) with each in SE3
    :param actual: The actual poses (Nx4x4), so that actual[i]~predicted[i]
    :return 1. a list of the translational errors (nan if not computable) 2. their RMSE
    """
    error_matrices = rte_error_matrice_s(timestamps, predicted, actual)
    translational_errors = [
        np.nan if m is None else np.linalg.norm(m[:3,3])
        for m in error_matrices
    ]
    rmse = np.sqrt(np.nanmean(np.asarray(translational_errors)**2))
    return translational_errors, rmse

def rte_rotational_errors_rmse(
        timestamps:list[int],
        predicted:np.ndarray,
        actual:np.ndarray
    )->tuple[list[float], float]:
    """
    Computes rotational RTEs (Relative Trajectory Error) and RMSE(RTEs)
    Uses timestamp_indices, assuming fixed Hz so if timestamp[i] != timestamp[i+1]-1, returns None
    :param timestamps: the timestamp index where predicted and actual were taken
    :param predicted: The predicted poses (Nx4x4) with each in SE3
    :param actual: The actual poses (Nx4x4), so that actual[i]~predicted[i]
    :return 1. a list of the rotational errors (nan if not computable) 2. their RMSE
    """
    error_matrices = rte_error_matrice_s(timestamps, predicted, actual)
    rotational_errors = [
        np.nan if m is None else rotational_difference(m, np.eye(4))
        for m in error_matrices
    ]
    rmse =  np.sqrt(np.nanmean(np.asarray(rotational_errors)**2))
    return rotational_errors, rmse