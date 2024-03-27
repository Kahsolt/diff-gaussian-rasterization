# Differential Gaussian Rasterization with Importances

This is a fork of the original rasterization engine [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for the paper ["3D Gaussian Splatting for Real-Time Rendering of Radiance Fields"](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf). We add a new salar attribute namely "importance" to each guassian, and implement both forward & backward pass for it.


##### Modeling formula:

⚪ Forward

Following the definition of rendered image color $ C $, which derives from the accumulation of guassian color multiplied by a learnable $ opacity_i $ both exponentially decayed w.r.t. the depth from viewpoint and zig-zagly decayed w.r.t. the sorted depth order:

$$ C = \sum_{i \in \mathcal{N}} c_i \alpha_i \sum_{j=1}^{i-1} (1 - \alpha_j) $$

where $ \alpha_i = opacity_i * e^{power} $, and $ power <= 0 $ interprets as the viewpoint depth.

We define here the importance map $ O $ in pixel space as:

$$ O = \sum_{i \in \mathcal{N}} o_i \alpha_i \sum_{j=1}^{i-1} (1 - \alpha_j) $$

⚪ Backward

Since the per-pixel importance $ O_i $ is derived from the accumulation per-Gaussian importance $ o_i $, conducting the chaining rule to obtain the gradients:

$$
\frac{\partial L}{\partial o_i} = \frac{\partial L}{\partial O} * \frac{\partial O}{\partial o_i} = \frac{\partial L}{\partial O} * \left[ \alpha_i \sum_{j=1}^{i-1} (1 - \alpha_j) \right]
$$

noted that the right part $ \left[ \cdot \right] $ has already been computed in the forward pass, we might simply cache it and reuse; but the original repo code recomputes it again (like pytorch [checkpointing](https://pytorch.org/docs/stable/checkpoint.html) mechanism), trading time for less VRAM usage :(

Additionally, the importance $ o_i $ interacts with opacity $ \alpha_i $, hence the gradients of opacity will add this new term:

$$
\frac{\partial L}{\partial \alpha_i} += \frac{\partial L}{\partial O} * \frac{\partial O}{\partial \alpha_i} = \frac{\partial L}{\partial O} * \left[ o_i \sum_{j=1}^{i-1} (1 - \alpha_j) \right]
$$

again the right part $ \left[ \cdot \right] $ is implemented in a recursive way following the original code, by making good use of the forward formula, here we show the rough idea:

$$ o_i \sum_{j=1}^{i-1} (1 - \alpha_j) = \frac{O - \sum_{k=1}^{j-1} O_k}{\alpha_i} $$


##### Code usage:

ℹ We provide source code only, for compiling how-tos follow the original [guide](https://github.com/graphdeco-inria/gaussian-splatting#Optimizer) :)

```python
from diff_gaussian_rasterization_ks import GaussianRasterizationSettings, GaussianRasterizer

# modified version with importance attribute
pc: GaussianModel
# [num_points, D=1], alike opacity
importance = pc.get_importance

rasterizer = GaussianRasterizer(raster_settings=...)
rendered_image, importance_map, radii = rasterizer(
  means3D = means3D,
  means2D = means2D,
  shs = shs,
  colors_precomp = colors_precomp,
  opacities = opacity,
  importances = importance,   # <- new pointcloud attributes
  scales = scales,
  rotations = rotations,
  cov3D_precomp = cov3D_precomp,
)

# [H, W], alike rendered_image but without channel dim C
importance_map  # <- rendered in pixel space
```


#### Acknowledgements

- gaussian-splatting: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- diff-gaussian-rasterization: [https://github.com/graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
  - ashawkey's: [https://github.com/ashawkey/diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)
  - leo-frank's: [https://github.com/leo-frank/diff-gaussian-rasterization-depth](https://github.com/leo-frank/diff-gaussian-rasterization-depth)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <!-- <h2 class="title">BibTeX</h2> -->
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

----
by Armit
2024/03/27
