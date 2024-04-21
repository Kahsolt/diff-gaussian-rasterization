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

import os
base_path = os.path.dirname(os.path.abspath(__file__))

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='diff_gaussian_rasterization_ks',
    packages=['diff_gaussian_rasterization_ks'],
    ext_modules=[
        CUDAExtension(
            name='diff_gaussian_rasterization_ks._C',
            sources=[
                'rasterizer/rasterizer.cu',
                'rasterizer/forward.cu',
                'rasterizer/backward.cu',
                'rasterizer_api.cu',
                'extension.cpp'
            ],
            extra_compile_args={
              'nvcc': ['-I' + os.path.join(base_path, 'third_party', 'glm')]
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
