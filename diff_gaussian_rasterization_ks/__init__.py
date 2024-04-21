#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from dataclasses import dataclass
from typing import Tuple, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from . import _C


def cpu_deep_copy_tuple(input_tuple):
    return tuple([item.detach().clone().cpu() if isinstance(item, Tensor) else item for item in input_tuple])


class ImageState:

    def __init__(self, buffer:Tensor, size:Tuple[int, int], align:int=128):
        H, W = size
        N = H * W
        offset = 0
        buffer = buffer.cpu().numpy()

        def next_offset() -> int:
            nonlocal offset
            while offset % align:
                offset += 1

        next_offset()
        final_T = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.float32).reshape((H, W))
        next_offset()
        n_contrib = torch.frombuffer(memoryview(buffer[offset:offset+4*N]), dtype=torch.int32).reshape((H, W))
        next_offset()
        ranges = torch.frombuffer(memoryview(buffer[offset:offset+8*N]), dtype=torch.int32).reshape((H, W, 2))

        self._final_T = final_T      # float, 4 bytes
        self._n_contrib = n_contrib  # uint32_t, 4 bytes
        self._ranges = ranges        # uint2, 8 bytes

    @property
    def final_T(self): return self._final_T
    @property
    def n_contrib(self): return self._n_contrib
    @property
    def ranges(self): return self._ranges


@dataclass
class GaussianRasterizationSettings:
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    bg: Tensor
    scale_modifier: float
    viewmatrix: Tensor
    projmatrix: Tensor
    sh_degree: int
    campos: Tensor
    prefiltered: bool = False
    debug: bool = False
    limit_n_contrib: bool = -1


class CudaRasterizer(Function):

    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities, importances, scales, rotations, cov3Ds_precomp, raster_settings:GaussianRasterizationSettings):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg,             # [C=3]
            means3D,                        # [P=182686, pos=3]
            colors_precomp,                 # no use
            opacities,                      # [P=182686, sigma=1]
            importances,                    # [P=182686, beta=1]
            scales,                         # [P=182686, pos=3]
            rotations,                      # [P=182686, Q=4]
            raster_settings.scale_modifier, # 1.0
            raster_settings.limit_n_contrib, # -1
            cov3Ds_precomp,                 # no use
            raster_settings.viewmatrix,     # [4, 4]
            raster_settings.projmatrix,     # [4, 4]
            raster_settings.tanfovx,        # float
            raster_settings.tanfovy,        # float
            raster_settings.image_height,   # int
            raster_settings.image_width,    # int
            sh,                             # [P=182686, D=16, pos=3]
            raster_settings.sh_degree,      # int
            raster_settings.campos,         # [pos=3]
            raster_settings.prefiltered,    # False
            raster_settings.debug           # False
        )

        # Invoke C++/CUDA rasterizer
        cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
        try:
            # int, [C=3, H=545, W=980], [H, W], [P=182686], [14433784], [56993151], [8545824]
            num_rendered, color, importance_map, n_contrib, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_fw.dump")
            print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
            raise ex

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, importances, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, importance_map, n_contrib, imgBuffer

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_importance_map, grad_n_contrib, grad_imgBuffer):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings: GaussianRasterizationSettings = ctx.raster_settings
        colors_precomp, importances, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D, 
            radii, 
            colors_precomp, 
            importances, 
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            raster_settings.limit_n_contrib, 
            cov3Ds_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_color, 
            grad_out_importance_map, 
            sh, 
            raster_settings.sh_degree, 
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        cpu_args = cpu_deep_copy_tuple(args)    # Copy them before they can be corrupted
        try:
            # [P=182686, pos=3], [P=182686, rgb=3], [P=182686, 1], [P=182686, pos=3], [P=182686, 6], [P=182686, D=16, pos=3], [P=182686, pos=3], [P=182686, Q=4]
            grad_means2D, grad_colors_precomp, grad_opacities, grad_importances, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
        except Exception as ex:
            torch.save(cpu_args, "snapshot_bw.dump")
            print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
            raise ex

        return (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_importances,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,   # raster_settings
        )


class GaussianRasterizer(nn.Module):

    def __init__(self, raster_settings:GaussianRasterizationSettings):
        super().__init__()

        self.raster_settings = raster_settings

    @torch.no_grad
    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        raster_settings = self.raster_settings
        visible = _C.mark_visible(
            positions,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
        )
        return visible

    def forward(self, means3D, means2D, opacities, importances=None, shs=None, colors_precomp=None, scales=None, rotations=None, cov3D_precomp=None):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = Tensor([])
        if colors_precomp is None:
            colors_precomp = Tensor([])
        if importances is None:
            importances = torch.ones_like(opacities)
        if scales is None:
            scales = Tensor([])
        if rotations is None:
            rotations = Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = Tensor([])

        # Invoke C++/CUDA rasterization routine
        color, radii, importance_map, n_contrib, imgBuffer = CudaRasterizer.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            importances,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
        img_state = ImageState(imgBuffer, (raster_settings.image_height, raster_settings.image_width))
        return color, radii, importance_map, n_contrib, img_state
