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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


class RasterizationOutputs(NamedTuple):
    """Outputs from the Gaussian rasterizer.
    
    Core material outputs (same level as color):
        color: [3, H, W] - Rendered albedo color
        roughness: [H, W] - Rendered roughness
        metallic: [H, W] - Rendered metallic
        
    Auxiliary outputs (geometry/regularization):
        depth: [1, H, W] - Rendered depth
        alpha: [1, H, W] - Accumulated alpha (opacity)  
        normal: [3, H, W] - Rendered surface normals
        middepth: [1, H, W] - Median depth
        distortion: [1, H, W] - Distortion loss auxiliary
        
    Other:
        radii: [N] - Screen-space radii of Gaussians
    """
    color: torch.Tensor        # [3, H, W] - Rendered albedo color
    roughness: torch.Tensor    # [H, W] - Rendered roughness
    metallic: torch.Tensor     # [H, W] - Rendered metallic
    depth: torch.Tensor        # [1, H, W] - Rendered depth
    alpha: torch.Tensor        # [1, H, W] - Accumulated alpha (opacity)
    normal: torch.Tensor       # [3, H, W] - Rendered surface normals
    middepth: torch.Tensor     # [1, H, W] - Median depth
    distortion: torch.Tensor   # [1, H, W] - Distortion loss auxiliary
    radii: torch.Tensor        # [N] - Screen-space radii of Gaussians


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    albedo,
    roughness,
    metallic,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
) -> RasterizationOutputs:
    color, out_roughness, out_metallic, depth, alpha, normal, middepth, distortion, radii = _RasterizeGaussians.apply(
        means3D,
        means2D,
        albedo,
        roughness,
        metallic,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    return RasterizationOutputs(
        color=color,
        roughness=out_roughness,
        metallic=out_metallic,
        depth=depth,
        alpha=alpha,
        normal=normal,
        middepth=middepth,
        distortion=distortion,
        radii=radii,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        albedo,
        roughness,
        metallic,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            means3D,
            albedo,
            roughness,
            metallic,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                num_rendered, color, out_roughness, out_metallic, auxiliary, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, out_roughness, out_metallic, auxiliary, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(albedo, roughness, metallic, means3D, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer)
        
        # Unpack auxiliary [7, H, W] into individual components
        out_depth = auxiliary[0:1]        # [1, H, W]
        out_alpha = auxiliary[1:2]        # [1, H, W]
        out_normal = auxiliary[2:5]       # [3, H, W]
        out_middepth = auxiliary[5:6]     # [1, H, W]
        out_distortion = auxiliary[6:7]   # [1, H, W]
        
        return color, out_roughness, out_metallic, out_depth, out_alpha, out_normal, out_middepth, out_distortion, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_roughness, grad_metallic, grad_depth, grad_alpha, grad_normal, grad_middepth, grad_distortion, grad_radii):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        albedo, roughness, metallic, means3D, scales, rotations, cov3Ds_precomp, radii, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Repack auxiliary gradients into single tensor [7, H, W] for C++ backend
        grad_auxiliary = torch.cat([
            grad_depth,       # [1, H, W] -> channel 0
            grad_alpha,       # [1, H, W] -> channel 1
            grad_normal,      # [3, H, W] -> channels 2-4
            grad_middepth,    # [1, H, W] -> channel 5
            grad_distortion,  # [1, H, W] -> channel 6
        ], dim=0)

        # Restructure args as C++ method expects them
        args = (
            means3D, 
            radii, 
            albedo,
            roughness,
            metallic,
            scales, 
            rotations, 
            raster_settings.scale_modifier, 
            cov3Ds_precomp, 
            raster_settings.viewmatrix, 
            raster_settings.projmatrix, 
            raster_settings.tanfovx, 
            raster_settings.tanfovy, 
            grad_out_color,
            grad_roughness,
            grad_metallic,
            grad_auxiliary,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args)
            try:
                grad_means2D, grad_albedo, grad_roughness, grad_metallic, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_albedo, grad_roughness, grad_metallic, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_albedo,
            grad_roughness,
            grad_metallic,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,  # raster_settings
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx: float
    tanfovy: float
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    prefiltered: bool
    debug: bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def forward(self, means3D, means2D, opacities, albedo, roughness, metallic, scales=None, rotations=None, cov3D_precomp=None) -> RasterizationOutputs:
        raster_settings = self.raster_settings

        if (scales is None or rotations is None) and cov3D_precomp is None:
            raise Exception('Please provide either scale/rotation pair or precomputed 3D covariance!')
        if (scales is not None or rotations is not None) and cov3D_precomp is not None:
            raise Exception('Please provide either scale/rotation pair or precomputed 3D covariance, not both!')

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            albedo,
            roughness,
            metallic,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

def compute_relocation(opacity_old, scale_old, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation(opacity_old, scale_old, N.int(), binoms, n_max)
    return new_opacity, new_scale 