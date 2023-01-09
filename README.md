![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/Banner.png)

# FEATURES
- ## BRDFs

  ### Glossy Specular 
  
  This path tracer implementes the microfacet model for the glossy specular, as described by [cook-torrance](), i.e.
  
  $$f(l, v) = {D(h)F(v, h)G(l,v,h)\over{4(n\cdot l)(n\cdot v)}}$$
  
  Since the normal distribution function ($NDF$) is the dominant term, we sample it to determin how the exitence radiance is reflected with respect to the surface normal. The probability distribution function ( $PDF$ ) is thus derived from the $NDF$.
  
  The term, such that the $NDF$ term is normalized is $$\int_\Omega D(h)(n\cdot h)d\omega_h = 1$$
  
  The marginal density function ( $MDF$ ) for $p(\theta)$ is 

  $$p(\theta) = \int_\Omega D(m)\cos{\theta_m}d\omega_h$$
  $$p(\theta) = \int_0^{2\pi}\int_0^{\pi\over{2}} D(m)\cos{\theta_m}\sin{\theta_m}d\theta_m d\phi_m = 2\pi D(m)\cos{\theta_m}\sin{\theta_m}$$
  
  The conditional density ( $CDF$ ) for $\phi$ is 
  
  $$p(\phi|\theta) = {p(\theta, \phi)\over{p(\theta)}} = {D(m)\cos{\theta_m}\sin{\theta_m} \over{2\pi D(m)\cos{\theta_m}\sin{\theta_m}}} = {1\over{2\pi}}$$
  
  Given the $MDF$ and the $CDF$ we can now compute the functions needed to sample our $PDF$ for $D(h)$.
  
  | $\theta$ | $\phi$ |
  | :---: | :---: |
  | $$p(\theta) = {\int_0^\theta 2\pi D(m)\cos{\theta_m'}}\sin{\theta_m'}d\theta_m' = {-\sin^2{x}\over{(\alpha^2-1)\sin^2{x}-\alpha^2}}$$ $$x = {-\sin^2{\theta}\over {(\alpha^2-1)\sin^2{\theta}-\alpha^2}}$$ $$\sin^2(\theta) = {\alpha^2x\over{x(\alpha^2-1)+1}}$$ $$\cos^2(\theta) = {1-x\over{x(\alpha^2-1)+1}}$$ $$\theta = cos^{-1}{\sqrt{1-x\over{x(\alpha^2-1)+1}}} $$ | 
  
  $D(m) = {\alpha^2\over{\pi((n\cdot m)^2(\alpha^2 - 1) + 1)^2 }}$
  $G(v) = {2(n\cdot v) \over {(n\cdot v) + \sqrt{\alpha^2 + (1 - \alpha^2)(n\cdot v)^2}}}$
  $F(v, h) = F0 + (1 - F0)(1 - (v\cdot h))^5$
  
  $f(l, v) = {D(h)F(v, h)G(l,v,h)\over{4(n\cdot l)(n\cdot v)}}$  $pdf = {D(m)(n\cdot h)\over{4(\omega_o\cdot h)}}$

  ### Lambertian Diffuse
  
  $f(l, v) = {albedo\over{\pi}}$
  $pdf = {1 \over{2\pi}}$
  
  ### Implementation
  - CUDA-RayTracer/dMaterial.cu
  
  ### Sources:
  - [A Reflectance Model for Computer Graphics](https://graphics.pixar.com/library/ReflectanceModel/paper.pdf)
  - [PBR: Importance Sampling](https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling)
  - [Specular BRDF Reference](http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html)
- ## Multiple Importance Sampling

    ### Results
    | BRDF Sample | Light Sample | MIS |
    | :---: | :---: | :---: |
    | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/brdf_samp.png?raw=true) | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/light_samp.png?raw=true) | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/ground_t.png?raw=true) |
    
    *250 samples per pixel with a path depth of 5*
    
    ### Implementation
    - CUDA-RayTracer/wavefront_kernels.cu
    - CUDA-RayTracer/dMaterial.cu
    
    ### Sources:
    - [PBR: Direct Lighting](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Direct_Lighting)
    
- ## Importance Sampled Environment Lights

    For an arbitrary HRDI image, probabilities are assigned to each pixel based on the total flux that passes through the solid angle.
   
    | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/heat_map.png?raw=true) | 
    |:--| 
    | *heat map of where samples are being drawn.* |
    
    The strategy used for sampling is as follows
    
      1. Sample a row of the environment map using the marginal distribution p(y)

      2. Sample a pixel within that row using the condition distribution p(x|y)

      3. Convert that (x,y) to a direction vector and return the appropriate radiance and pdf values.

    ### Results
    | Pre-Filtering HRDI | Uniform Sampling| Importance Sampling|
    | :---: | :---: | :---: |
    | On | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/ENV_importance_sampling_off_easy.png?raw=true) | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/ground_t.png?raw=true) |
    | Off | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/ENV_importance_sampling_off_hard.png?raw=true) | ![alt text](https://github.com/JakeKurtz/MC-Path-Tracer/blob/main/images/ENV_importance_sampling_on_hard.png?raw=true) |
    
    *250 samples per pixel with a path depth of 5*
  
    ### Implementation
    - CUDA-RayTracer/light_initialization_kernels.cu
    - CUDA-RayTracer/EnvironmentLight.cu
  
    ### Sources:
  - [PBR: Sampling Lights](https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Light_Sources)
  - [Berkeley: Environment Map Lights](https://cs184.eecs.berkeley.edu/sp18/article/25)
- ## CUDA Wavefront
  Wavefront pathtracing avoids thread divergence by splitting up the render kernel into several smaller kernels, which are each responsible for a specific task e.g.
    1. logic
    2. generate ray
    3. material ray
    4. extend ray
    5. shadow ray
  
    ### Implementation
    - CUDA-RayTracer/wavefront_kernels.cu
    - CUDA-RayTracer/Wavefront.cuh
  
  ### Sources:
  - [Megakernels Considered Harmful](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
