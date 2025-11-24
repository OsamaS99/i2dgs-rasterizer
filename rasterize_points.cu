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

#include <math.h>
#include <torch/extension.h>
#include <tuple>
#include <cuda_runtime_api.h>
#include <functional>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
	auto lambda = [&t](size_t N) {
		t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
	};
	return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& albedo,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
	if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
		AT_ERROR("means3D must have dimensions (num_points, 3)");
	}

	const int P = means3D.size(0);
	const int H = image_height;
	const int W = image_width;

	CHECK_CUDA(means3D);
	CHECK_CUDA(albedo);
	CHECK_CUDA(roughness);
	CHECK_CUDA(metallic);
	CHECK_CUDA(opacity);
	CHECK_CUDA(scales);
	CHECK_CUDA(rotations);
	CHECK_CUDA(transMat_precomp);
	CHECK_CUDA(viewmatrix);
	CHECK_CUDA(projmatrix);
	CHECK_CUDA(campos);

	auto int_opts = means3D.options().dtype(torch::kInt32);
	auto float_opts = means3D.options().dtype(torch::kFloat32);

	// Core material outputs (same level as color)
	torch::Tensor out_color = torch::zeros({NUM_CHANNELS, H, W}, float_opts);
	torch::Tensor out_roughness = torch::zeros({H, W}, float_opts);
	torch::Tensor out_metallic = torch::zeros({H, W}, float_opts);
	// Auxiliary outputs (depth, alpha, normal, middepth, distortion)
	torch::Tensor out_auxiliary = torch::zeros({7, H, W}, float_opts);
	torch::Tensor radii = torch::zeros({P}, int_opts);

	torch::Device device(torch::kCUDA);
	torch::TensorOptions options(torch::kByte);
	torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
	torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
	torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

	int rendered = 0;
	if (P != 0)
	{
		rendered = CudaRasterizer::Rasterizer::forward(
			geomFunc,
			binningFunc,
			imgFunc,
			P,
			W, H,
			means3D.contiguous().data<float>(),
			albedo.contiguous().data<float>(),
			roughness.contiguous().data<float>(),
			metallic.contiguous().data<float>(),
			opacity.contiguous().data<float>(), 
			scales.contiguous().data_ptr<float>(),
			scale_modifier,
			rotations.contiguous().data_ptr<float>(),
			transMat_precomp.contiguous().data<float>(), 
			viewmatrix.contiguous().data<float>(), 
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			prefiltered,
			out_color.contiguous().data<float>(),
			out_roughness.contiguous().data<float>(),
			out_metallic.contiguous().data<float>(),
			out_auxiliary.contiguous().data<float>(),
			radii.contiguous().data<int>(),
			debug);
	}
	return std::make_tuple(rendered, out_color, out_roughness, out_metallic, out_auxiliary, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& albedo,
	const torch::Tensor& roughness,
	const torch::Tensor& metallic,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& transMat_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_roughness,
	const torch::Tensor& dL_dout_metallic,
	const torch::Tensor& dL_dout_auxiliary,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug) 
{
	CHECK_CUDA(means3D);
	CHECK_CUDA(radii);
	CHECK_CUDA(albedo);
	CHECK_CUDA(roughness);
	CHECK_CUDA(metallic);
	CHECK_CUDA(scales);
	CHECK_CUDA(rotations);
	CHECK_CUDA(transMat_precomp);
	CHECK_CUDA(viewmatrix);
	CHECK_CUDA(projmatrix);
	CHECK_CUDA(campos);
	CHECK_CUDA(binningBuffer);
	CHECK_CUDA(imageBuffer);
	CHECK_CUDA(geomBuffer);

	const int P = means3D.size(0);
	const int H = dL_dout_color.size(1);
	const int W = dL_dout_color.size(2);

	torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dalbedo = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_droughness = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dmetallic = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dnormal = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
	torch::Tensor dL_dtransMat = torch::zeros({P, 9}, means3D.options());
	torch::Tensor dL_dscales = torch::zeros({P, 2}, means3D.options());
	torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

	if (P != 0)
	{  
		CudaRasterizer::Rasterizer::backward(
			P, R,
			W, H, 
			means3D.contiguous().data<float>(),
			albedo.contiguous().data<float>(),
			roughness.contiguous().data<float>(),
			metallic.contiguous().data<float>(),
			scales.data_ptr<float>(),
			scale_modifier,
			rotations.data_ptr<float>(),
			transMat_precomp.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			campos.contiguous().data<float>(),
			tan_fovx,
			tan_fovy,
			radii.contiguous().data<int>(),
			reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
			reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
			dL_dout_color.contiguous().data<float>(),
			dL_dout_roughness.contiguous().data<float>(),
			dL_dout_metallic.contiguous().data<float>(),
			dL_dout_auxiliary.contiguous().data<float>(),
			dL_dmeans2D.contiguous().data<float>(),
			dL_dnormal.contiguous().data<float>(),  
			dL_dopacity.contiguous().data<float>(),
			dL_dalbedo.contiguous().data<float>(),
			dL_droughness.contiguous().data<float>(),
			dL_dmetallic.contiguous().data<float>(),
			dL_dmeans3D.contiguous().data<float>(),
			dL_dtransMat.contiguous().data<float>(),
			dL_dscales.contiguous().data<float>(),
			dL_drotations.contiguous().data<float>(),
			debug);
	}

	return std::make_tuple(dL_dmeans2D, dL_dalbedo, dL_droughness, dL_dmetallic, dL_dopacity, dL_dmeans3D, dL_dtransMat, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
	torch::Tensor& means3D,
	torch::Tensor& viewmatrix,
	torch::Tensor& projmatrix)
{ 
	const int P = means3D.size(0);
	torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));

	if (P != 0)
	{
		CudaRasterizer::Rasterizer::markVisible(
			P,
			means3D.contiguous().data<float>(),
			viewmatrix.contiguous().data<float>(),
			projmatrix.contiguous().data<float>(),
			present.contiguous().data<bool>());
	}

	return present;
}
