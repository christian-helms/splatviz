# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""
import math
import torch


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(
            horizontal_mean,
            vertical_mean,
            lookat_position,
            radius=1,
            up_vector=torch.tensor([0, -1, 0]),
            device=torch.device("cuda:0")
    ):
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius)
        forward_vectors = get_forward_vector(lookat_position, horizontal_mean, vertical_mean, radius, camera_origins=camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins, up_vector=up_vector).to(device)


def get_origin(horizontal_mean, vertical_mean, radius):
    h = torch.tensor(horizontal_mean)
    v = torch.tensor(vertical_mean)
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = torch.arccos(1 - 2 * v)

    camera_origins = torch.zeros((3))
    camera_origins[0:1] = radius * torch.sin(phi) * torch.cos(math.pi - theta)
    camera_origins[2:3] = radius * torch.sin(phi) * torch.sin(math.pi - theta)
    camera_origins[1:2] = radius * torch.cos(phi)
    return camera_origins


def get_forward_vector(lookat_position, horizontal_mean, vertical_mean, radius, camera_origins=None):
    if camera_origins is None:
        camera_origins = get_origin(horizontal_mean, vertical_mean, radius)
    return normalize_vecs(lookat_position.to(camera_origins.device) - camera_origins)


def create_cam2world_matrix(forward_vector, origin, up_vector=torch.tensor([0, -1, 0])):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """
    forward_vector = normalize_vecs(forward_vector)
    up_vector = up_vector.float().to(origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)
    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert cam2world.shape[1:] == (4, 4)
    return cam2world


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

