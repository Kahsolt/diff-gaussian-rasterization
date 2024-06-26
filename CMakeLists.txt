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

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_library(CudaRasterizer
	rasterizer/config.h
	rasterizer/auxiliary.h
	rasterizer/backward.h
	rasterizer/backward.cu
	rasterizer/forward.h
	rasterizer/forward.cu
	rasterizer/rasterizer.h
	rasterizer/rasterizer.cu
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
