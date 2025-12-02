import torch
import torch.nn.functional as F
import math

# from pytorch3d.transforms import matrix_to_axis_angle


def process_input_and_kinematics(input_tensor, inward, num_people):

    parent_dict = {child: parent for child, parent in inward}

    # input_tensor shape: (N, 6, T, V)
    N, C, T, V = input_tensor.shape
    # num_people = 2

    # (N, 2, 3, T, V) -> (N*2, 3, T, V)
    if num_people > 1:
        output_tensor = input_tensor.view(N, num_people, 3, T, V)
        output_tensor = output_tensor.permute(0, 1, 3, 4, 2).contiguous()  # -> (N, 2, T, V, 3)
        output_tensor = output_tensor.view(N * num_people, T, V, 3)  # -> (N*2, T, V, 3)
    else:
        output_tensor = input_tensor.permute(0, 2, 3, 1).contiguous()

    return output_tensor, parent_dict


def get_root_orientation(rel_pos, joint_map):
    """
    rel_pos shape: (N_m, T, V, 3)
    joint_map
    """
    # 假设 joint_map = {'SpineBase': 0, 'SpineMid': 1, 'RightHip': 12, 'LeftHip': 16, ...}
    p_root = rel_pos[:, :, joint_map['SpineBase'], :]  # (0,0,0)
    p_spine_mid = rel_pos[:, :, joint_map['SpineMid'], :]
    p_rhip = rel_pos[:, :, joint_map['RightHip'], :]
    p_lhip = rel_pos[:, :, joint_map['LeftHip'], :]

    y_axis = F.normalize(p_spine_mid - p_root, dim=-1)
    x_axis_raw = p_rhip - p_lhip

    # Gram-Schmidt orthogonalization
    z_axis = F.normalize(torch.cross(x_axis_raw, y_axis, dim=-1), dim=-1)
    x_axis = F.normalize(torch.cross(y_axis, z_axis, dim=-1), dim=-1)

    # rotation matrix R = [x_axis, y_axis, z_axis]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (N_m, T, 3, 3)

    # --- SO3 ratation matrix to direction vector ---
    root_rotation_vector = manual_matrix_to_axis_angle(R)  # (N_m, T, 3)

    return root_rotation_vector


def get_local_joint_rotations(rel_pos, parent_dict):
    """
    rel_pos shape: (N_m, T, V, 3)
    """
    joint_rotations = []

    for j in range(rel_pos.shape[2]):  # V
        if j not in parent_dict: continue
        p = parent_dict[j]
        if p not in parent_dict: continue
        gp = parent_dict[p]

        vec_parent = F.normalize(rel_pos[:, :, p, :] - rel_pos[:, :, gp, :], dim=-1)
        vec_child = F.normalize(rel_pos[:, :, j, :] - rel_pos[:, :, p, :], dim=-1)

        # rotate axis
        axis = F.normalize(torch.cross(vec_parent, vec_child, dim=-1), dim=-1)

        # ratate angle
        dot_product = torch.sum(vec_parent * vec_child, dim=-1)
        angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0)).unsqueeze(-1)  # (N_m, T, 1)

        # vector = angle * axis
        rotation_vector = angle * axis  # (N_m, T, 3)
        joint_rotations.append(rotation_vector)

    # cat all
    return torch.cat(joint_rotations, dim=-1)  # (N_m, T, (V-k)*3)


def get_root_orientation_2d(rel_pos, joint_map):
    """
    rel_pos shape: (N, T, V, 2)
    joint_map
    """
    # joint_map = {'SpineBase': 0, 'SpineMid': 1, ...}
    p_root = rel_pos[:, :, joint_map['SpineBase'], :]
    p_spine_mid = rel_pos[:, :, joint_map['SpineMid'], :]

    body_y_axis = p_spine_mid - p_root  # Shape: (N, T, 2)

    angle = torch.atan2(body_y_axis[:, :, 1], body_y_axis[:, :, 0])  # Shape: (N, T)

    return angle.unsqueeze(-1)  # Shape: (N, T, 1)


def get_local_joint_rotations_2d(rel_pos, parent_dict):
    """
    rel_pos shape: (N, T, V, 2)
    """
    joint_angles = []

    for j in range(rel_pos.shape[2]):  # V
        if j not in parent_dict: continue
        p = parent_dict[j]

        if p not in parent_dict: continue
        gp = parent_dict[p]

        vec_parent = F.normalize(rel_pos[:, :, p, :] - rel_pos[:, :, gp, :], dim=-1)
        vec_child = F.normalize(rel_pos[:, :, j, :] - rel_pos[:, :, p, :], dim=-1)

        # vec_parent: (N, T, 2), vec_child: (N, T, 2)
        p_x, p_y = vec_parent[:, :, 0], vec_parent[:, :, 1]
        c_x, c_y = vec_child[:, :, 0], vec_child[:, :, 1]

        dot_product = p_x * c_x + p_y * c_y

        cross_product_mag = p_x * c_y - p_y * c_x

        # atan2 get the angle with direction between vec_child and vec_parent
        angle = torch.atan2(cross_product_mag, dot_product)  # Shape: (N, T)

        joint_angles.append(angle.unsqueeze(-1))  # Shape: (N, T, 1)

    # 将所有关节的角度拼接起来
    return torch.cat(joint_angles, dim=-1)  # (N, T, V-k)


def manual_matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
        matrix (torch.Tensor): (..., 3, 3)
        torch.Tensor:  (..., 3)
    """
    if not torch.is_tensor(matrix):
        raise TypeError("the input must be a PyTorch tensor")
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("the last 2 dimension of the tensor (3, 3)")

    matrix = matrix.float()

    batch_dim = matrix.shape[:-2]
    num_matrices = matrix.dim() - 2

    #  (N, 3, 3)
    R = matrix.reshape(-1, 3, 3)

    axis_angle = torch.zeros(R.shape[0], 3, device=R.device, dtype=R.dtype)

    # trace(R) = R_11 + R_22 + R_33
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # theta
    # cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.acos(cos_theta)


    # 1.  angle ->0  (trace -> 3)
    near_zero_mask = angle < 1e-6

    # 2.  angle ->pi  (trace -> -1)
    #  cos(angle) -> -1, sin(angle) -> 0
    near_pi_mask = torch.isclose(angle, torch.tensor(torch.pi, device=R.device, dtype=R.dtype), atol=1e-5)

    # 3.  (0 < angle < pi)
    general_mask = ~(near_zero_mask | near_pi_mask)

    if general_mask.any():
        R_general = R[general_mask]
        angle_general = angle[general_mask]

        # sin(angle)  is not 0
        sin_angle = torch.sin(angle_general)

        # caculate axis
        axis_unnormalized = torch.stack([
            R_general[:, 2, 1] - R_general[:, 1, 2],
            R_general[:, 0, 2] - R_general[:, 2, 0],
            R_general[:, 1, 0] - R_general[:, 0, 1]
        ], dim=-1)

        axis = axis_unnormalized / (2 * sin_angle[:, None])

        axis_angle[general_mask] = axis * angle_general[:, None]

    # --- angle -> pi ---
    if near_pi_mask.any():
        R_pi = R[near_pi_mask]

        #  R + I = 2 * u * u^T
        # diag(R + I) = [2*ux^2, 2*uy^2, 2*uz^2]
        diag_R_plus_I = torch.diagonal(R_pi, dim1=-2, dim2=-1) + 1.0

        # ux^2 = (R_00 + 1) / 2, ...
        diag_R_plus_I[diag_R_plus_I < 0] = 0

        # 计算轴的平方
        axis_sq = diag_R_plus_I / 2.0

        # off_diag = [R_12, R_02, R_01]
        off_diag = torch.stack([R_pi[:, 1, 2], R_pi[:, 0, 2], R_pi[:, 0, 1]], dim=-1)

        # axis = sqrt(axis_sq)
        axis = torch.sqrt(axis_sq)

        _, max_idx = torch.max(axis_sq, dim=-1)

        # sgn(ux*uy) = sgn(R_01), sgn(ux*uz) = sgn(R_02), sgn(uy*uz) = sgn(R_12)

        signs = torch.sign(off_diag)

        for i, idx in enumerate(max_idx):
            if idx == 0:  # ux is the max
                axis[i, 1] *= torch.sign(R_pi[i, 0, 1])
                axis[i, 2] *= torch.sign(R_pi[i, 0, 2])
            elif idx == 1:  # uy is the max
                axis[i, 0] *= torch.sign(R_pi[i, 1, 0])
                axis[i, 2] *= torch.sign(R_pi[i, 1, 2])
            else:  # uz is the max
                axis[i, 0] *= torch.sign(R_pi[i, 2, 0])
                axis[i, 1] *= torch.sign(R_pi[i, 2, 1])

        # * pi
        axis_angle[near_pi_mask] = axis * angle[near_pi_mask].unsqueeze(-1)

    # for near_zero_mask，the result is the initial zero vector, so no need to handle

    return axis_angle.reshape(*batch_dim, 3)


def generate_generalized_coordinates(input_tensor, dataset):

    input_tensor = input_tensor.permute(0, 1, 3, 2).contiguous()

    if dataset == 'PKU-subject' or dataset == 'PKU-view':
        inward_connections = [(12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16),
                              (18, 17), (19, 18), (1, 0), (20, 1), (2, 20), (3, 2), (4, 20),
                              (5, 4), (6, 5), (7, 6), (21, 7), (22, 6), (8, 20), (9, 8),
                              (10, 9), (11, 10), (24, 10), (23, 11)]
        joint_map = {'SpineBase': 0, 'SpineMid': 1, 'RightHip': 12, 'LeftHip': 16}
        num_people = 2
        indices = [3,4,5,9,10,11]
        input_tensor = input_tensor[:,indices,:,:]

    elif dataset == 'LARA':
        inward_connections = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 0), (6, 5), (7, 6), (8, 7), (9, 0), (10, 9), (11, 9),
                         (12, 11), (13, 12), (14, 13), (15, 9), (16, 15), (17, 16), (18, 17)]
        joint_map = {'SpineBase': 0, 'SpineMid': 9, 'RightHip': 1, 'LeftHip': 5}
        num_people = 1
        input_tensor = input_tensor[:,-3:,:,:]

    elif dataset == 'MCFS-130' or dataset == 'MCFS-22':
        inward_connections = [(1,8), (0,1), (15,0), (17,15), (16,0), (18,16), (5,1),
                          (6,5), (7,6), (2,1), (3,2), (4,3), (9,8),
                          (10, 9), (11, 10), (24, 11), (22, 11), (23, 22), (12,8),
                          (13, 12), (14, 13), (21, 14), (19, 14), (20, 19)]
        joint_map = {'SpineBase': 8, 'SpineMid': 1, 'RightHip': 9, 'LeftHip': 12}
        num_people = 1
        # N, C, T, V = input_tensor.shape
        # z_axis = torch.zeros((N, 1, T, V), dtype=input_tensor.dtype, device=input_tensor.device)
        # input_tensor = torch.cat([input_tensor, z_axis], dim=1)
        # input_tensor = input_tensor[:,:,:,:]

    elif dataset == 'TCG-15':
        inward_connections = [(16, 2), (0, 16), (9, 0), (1, 9), (3, 9), (5, 3), (6, 5), (10, 9), (12, 10), (13, 12),
                             (8, 2), (7, 8), (4, 7), (15, 2), (14, 15), (11, 14)]
        joint_map = {'SpineBase': 2, 'SpineMid': 16, 'RightHip': 15, 'LeftHip': 8}
        num_people = 1
        input_tensor = input_tensor[:,-3:,:,:]

    # 步骤1
    rel_pos, parent_dict = process_input_and_kinematics(input_tensor, inward_connections, num_people)

    if dataset == 'MCFS-130' or dataset == 'MCFS-22':
        root_orientation_q = get_root_orientation_2d(rel_pos, joint_map)
        local_rotations_q = get_local_joint_rotations_2d(rel_pos, parent_dict)
    else:
        root_orientation_q = get_root_orientation(rel_pos, joint_map)
        local_rotations_q = get_local_joint_rotations(rel_pos, parent_dict)

    # 步骤4：最终拼接
    q = torch.cat([root_orientation_q, local_rotations_q], dim=-1)  # (N*2, T, d)

    return q


def get_derivatives(q: torch.Tensor, T_dim=1):
    """
    from q to get q_dot and q_ddot

    Args:
        q (torch.Tensor):  (..., T, d)。
        T_dim (int)

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: q, q_dot, q_ddot。
    """

    if T_dim != 1:
        dims = list(range(q.dim()))
        dims[1], dims[T_dim] = dims[T_dim], dims[1]
        q = q.permute(*dims)

    q_dot = torch.cat([q[:, 0:1, :], q[:, 1:, :] - q[:, :-1, :]], dim=1)
    q_ddot = torch.cat([q_dot[:, 0:1, :], q_dot[:, 1:, :] - q_dot[:, :-1, :]], dim=1)

    if T_dim != 1:
        dims = list(range(q.dim()))
        dims[1], dims[T_dim] = dims[T_dim], dims[1]
        q = q.permute(*dims)
        q_dot = q_dot.permute(*dims)
        q_ddot = q_ddot.permute(*dims)

    return q, q_dot, q_ddot