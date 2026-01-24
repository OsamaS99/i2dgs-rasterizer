/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		float focal_x, float focal_y,
		const float background,
		const float2* means2D,
		const float4* normal_opacity,
		const float* albedo,
		const float* roughness,
		const float* metallic,
		const float* transMats,
		const float* depths,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dpix_roughness,
		const float* dL_dpix_metallic,
		const float* dL_daux,
		// Intersection gradients
		int max_intersections,
		const float* dL_dintersection_depths,
		const float* dL_dintersection_weights,
		const int* num_intersections,
		float* dL_dtransMat,
		float4* dL_dmean2D,
		float* dL_dnormal3D,
		float* dL_dopacity,
		float* dL_dalbedo,
		float* dL_droughness,
		float* dL_dmetallic);

	void preprocess(
		int P,
		const float3* means,
		const int* radii,
		const glm::vec2* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* transMats,
		const float* view,
		const float* proj,
		const float focal_x, const float focal_y,
		const float tan_fovx, const float tan_fovy,
		const glm::vec3* campos,
		float4* dL_dmean2D,
		const float* dL_dnormal3D,
		float* dL_dtransMat,
		float* dL_dalbedo,
		glm::vec3* dL_dmeans,
		glm::vec2* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
