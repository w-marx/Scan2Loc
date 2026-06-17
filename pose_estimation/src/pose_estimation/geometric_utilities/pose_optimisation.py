import torch

def rot_vec_to_so3(w:torch.Tensor) -> torch.Tensor:
    """
    Computes hat(w) from w and doesnt break gradient flow
    :param w: A rotation vector of size 3
    """
    wx, wy, wz = w
    z = torch.zeros((), dtype=w.dtype, device=w.device)
    return torch.stack([
        torch.stack([z,  -wz,  wy]),
        torch.stack([wz,  z,  -wx]),
        torch.stack([-wy, wx,  z]),
    ])

def compute_exp_what(w:torch.Tensor) -> torch.Tensor:
    """
    Compute exp(w_hat * theta), ! divides by ||w||
    :param w: A rotation vector of size 3
    """
    theta = torch.norm(w)
    w_hat_norm = rot_vec_to_so3(w/theta)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + w_hat_norm * torch.sin(theta) + (w_hat_norm @ w_hat_norm) * (1-torch.cos(theta))

def exp_se3(s:torch.Tensor, eps = 1e-8):
    """
    Takes an array: S = (w, v)
    And returns exp([S])
    :param s: An array: [w1, w2, w3, s1, s2, s3]
    :param eps: The theta threshold for simplified calculation
    :return: 4x4 hom. matrix: exp([S]*theta)
    """
    w, v = s[:3], s[3:]

    # theta = |Sw|
    theta = torch.norm(w)

    if theta < eps:
        exp_what = torch.eye(3, device=s.device, dtype=s.dtype) + rot_vec_to_so3(w)
        t = v + 0.5 * torch.cross(w, v, dim = 0)
    else:
        w_norm = w / theta
        exp_what = compute_exp_what(w)
        eye_minus_exp_what = torch.eye(3, device=s.device, dtype=s.dtype) - exp_what
        wxv = torch.cross(w_norm, v, dim = 0)
        t = eye_minus_exp_what @ wxv + (w_norm[:, None] @ w_norm[:, None].T) @ v * theta

    T = torch.eye(4, device=s.device, dtype=s.dtype)
    T[:3, :3] = exp_what
    T[:3, 3] = t
    return T


def euler_to_matrix(angles):
    """
    angles: (3,) tensor with requires_grad=True
    return: (3,3) rotation matrix, fully differentiable w.r.t. angles
    """
    rx, ry, rz = angles
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    one = torch.ones((), device=angles.device)
    zero = torch.zeros((), device=angles.device)

    Rx = torch.stack([
        torch.stack([one,  zero,  zero]),
        torch.stack([zero,  cx,  -sx]),
        torch.stack([zero,  sx,   cx])
    ])
    Ry = torch.stack([
        torch.stack([ cy,  zero,  sy]),
        torch.stack([zero,  one,  zero]),
        torch.stack([-sy,  zero,  cy])
    ])
    Rz = torch.stack([
        torch.stack([ cz, -sz, zero]),
        torch.stack([ sz,  cz, zero]),
        torch.stack([zero, zero, one])
    ])

    return Rz @ Ry @ Rx


def compute_pose_exp_se3(x_i:torch.Tensor, cam_t_base:torch.Tensor)->torch.Tensor:
    return exp_se3(x_i) @ cam_t_base


def compute_pose_euler(x_i:torch.Tensor, cam_t_base:torch.Tensor)->torch.Tensor:
    angles = x_i[:3]
    t = x_i[3:]

    r_delta = euler_to_matrix(angles)

    delta_pose = torch.eye(4, dtype=x_i.dtype, device=x_i.device)
    delta_pose[:3, :3] = r_delta
    delta_pose[:3, 3] = t

    return delta_pose @ cam_t_base