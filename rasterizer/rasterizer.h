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

#ifndef RASTERIZER_H
#define RASTERIZER_H

#include <iostream>
#include <vector>
#include <functional>
#include <cuda_runtime_api.h>

namespace RasterizerState
{
	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};
}

namespace Rasterizer
{
	int forward(
		std::function<char* (size_t)> geometryBuffer,
		std::function<char* (size_t)> binningBuffer,
		std::function<char* (size_t)> imageBuffer,
		const int P, int D, int M,
		const float* background,
		const int width, int height,
		const float* means3D,
		const float* shs,
		const float* colors_precomp,
		const float* opacities,
		const float* importances,
		const float* scales,
		const float scale_modifier,
		const int limit_n_contrib,
		const float* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float* cam_pos,
		const float tan_fovx, float tan_fovy,
		const bool prefiltered,
		float* out_color,
		float* out_importance_map,
		int* out_n_contrib,
		int* radii = nullptr,
		bool debug = false
	);

	void backward(
		const int P, int D, int M, int R,
		const float* background,
		const int width, int height,
		const float* means3D,
		const float* shs,
		const float* colors_precomp,
		const float* importances,
		const float* scales,
		const float scale_modifier,
		const int limit_n_contrib,
		const float* rotations,
		const float* cov3D_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float* campos,
		const float tan_fovx, float tan_fovy,
		const int* radii,
		char* geom_buffer,
		char* binning_buffer,
		char* image_buffer,
		const float* dL_dpix,
		const float* dL_dpix_imp,
		float* dL_dmean2D,
		float* dL_dconic,
		float* dL_dopacity,
		float* dL_dimportance,
		float* dL_dcolor,
		float* dL_dmean3D,
		float* dL_dcov3D,
		float* dL_dsh,
		float* dL_dscale,
		float* dL_drot,
		bool debug = false
	);

	void markVisible(
		int P,
		float* means3D,
		float* viewmatrix,
		float* projmatrix,
		bool* present
	);
};

#endif
