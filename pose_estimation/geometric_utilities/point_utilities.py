import numpy as np

from shared.assertion_helpers import assert_homogeneous_mat, assert_intrinsic_mat

def project_visible_points(base_points:np.ndarray, cam_t_base:np.ndarray, intrinsic_mat:np.ndarray):

    assert assert_homogeneous_mat(cam_t_base, size = 4)
    assert assert_intrinsic_mat(intrinsic_mat)

    base_points = np.column_stack([base_points, np.ones(base_points.shape[0])])
    P = intrinsic_mat @ cam_t_base[:3, :]
    projected = (P @ base_points.T).T    
    Zc = projected[:, 2:3]
    return projected[:, :2] / np.clip(Zc, a_min=1e-10, a_max=None)


def remove_outliers_from_point_cloud(points:np.ndarray, contamination:float = 0.05, eps = 1e-8)->np.ndarray:
    """
    Uses I-Forest to remove points deemed as outliers
    :param contamination: The percentage of points to remove
    :param points: A Nx3-float numpy array of x,y,z points
    :return: A Mx3-float numpy array of x,y,z points with M <= N
    """
    assert 0 <= contamination <= 1.0
    assert points.ndim == 2 and points.shape[1] == 3

    if contamination <= eps:
        return points
    if contamination >= 1.0-eps:
        return np.empty((0,3))

    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(contamination=contamination)
    forest.fit(points)
    prediction = forest.predict(points)
    return points[prediction==1]


def set_outliers_to_nan(points:np.ndarray, contamination:float = 0.05, eps = 1e-8)->np.ndarray:
    """
    Uses I-Forest to set points deemed as outliers to NaN
    :param contamination: The percentage of points to set to Nan
    :param points: A Nx3-float numpy array of x,y,z points
    :return: A Mx3-float numpy array of x,y,z points with M <= N
    """
    assert 0 <= contamination <= 1.0
    assert points.ndim == 2 and points.shape[1] == 3

    if contamination >= 1.0-eps:
        return np.full_like(points, np.nan)

    c_points = points.copy()

    if contamination <= eps:
        return c_points
    
    valid_mask = np.isfinite(c_points).all(axis=1)


    from sklearn.ensemble import IsolationForest

    forest = IsolationForest(contamination=contamination)
    forest.fit(c_points[valid_mask])

    prediction = np.ones(len(c_points), dtype=int)
    prediction[valid_mask] = forest.predict(c_points[valid_mask])

    c_points[prediction !=1 ] = np.nan

    return c_points

