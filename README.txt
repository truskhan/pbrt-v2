Welcome to the ray hierarchy project. The project enriches the PBRT renderer (Physical Based Renderer from Mat Pharr and Greg Humphreys) with an accelerator - ray hierarchy and breadth-first renderer. The accelerator takes advantage of OpenCL parallel capability.

The OpenCL code could be run on CPU (thanks to ATI OpenCL SDK) or on GPU (use ATI or NVIDIA OpenCL SDK), in case you have compatible graphic card. 

The project aim is in exploring ray-hierarchy possibility in ray-object test speed up and comparing the time spend in ray tracing the scene with combine approach - using object space and ray space hierarchies. 

The idea and inspiration comes from several sources:
David Roger, Ulf Assarsson, and Nicolas Holzschuch. Whitted Ray-Tracing for Dynamic Scenes using a Ray-Space Hierarchy on the GPU,2007
http://artis.imag.fr/Publications/2007/RAH07/
A. J. Chung and A.J. Field. Ray space for hierarchical ray casting, 1999
ftp://ftp.computer.org/MAGS/CG&A/mms/111581.pdf
Laszlo Szecsi. The Hierarchical Ray Eengine, 2006
http://wscg.zcu.cz/wscg2006/papers_2006/full/g89-full.pdf
Kirill Garanzha and Charles Loop. Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing, 2010 
http://research.microsoft.com/en-us/um/people/cloop/garanzhaloop2010.pdf

Windows compilation quick guide:
- install CUDA toolkit (contains necessary OpenCL libs) from NVIDIA's web page (developer.nvidia.com/object/cuda_3_2_downloads.html)
- install CUDA SDK and compile enclosed examples, run oclDeviceQuery to verify you have CUDA enabled GPU
- open MS Visual Studio Express 2008 and the project file located in ./src/pbrt.vs2008
- use preprocessor directives to get code insight 
  GPU_TIMES - prints OpenCL kernels total execution times at the end of rendering
  STAT_RAY_TRIANGLE - renders image visualisation primary ray-triangle test counts in rainbow colour mapping
  STAT_PRAY_TRIANGLE - renders image visualisation shadow ray-triangle test counts in rainbow colour mapping

Ubuntu compilation quick guide:
- install CUDA toolkit (contains necessary OpenCL libs) from NVIDIA's web page (developer.nvidia.com/object/cuda_3_2_downloads.html)
- install CUDA SDK and compile enclosed examples, run oclDeviceQuery to verify you have CUDA enabled GPU
- use enclosed Makefile or CodeBlocks project to compile the PBRT, ensure Makefile variable NVIDIASDKROOT or ATISTREAMSDKROOT points to right direction, where you installed the SDKs (if you left the defaults values it should be OK and won't need any modifications) 

Tested platforms:
Windows 7, Visual Studio Express 2008 and NVIDIA CUDA enabled GPU GTX 275
Ubuntu 10.04, g++ and NVIDIA CUDA enabled GPU GTX 275
Ubuntu 10.04, g++ and AMD SDK with CPU (only naive accelerator as the others needs texture support)

New and modified PBRT input file parameters:
Renderer "breadth-first" - to explore more rays at once and to be able to build the ray hierarchy
SurfaceIntegrator "directlighting" or "path" - modified surface renderers which can cope with multiple rays at once
Accelerator "naive" - naive ray-triangle method on OpenCL device
Accelerator "rayhierarchy" - ray-hierarchy build and used on OpenCL device
Accelerator "ray-bvh" - ray-hierarchy build and used on OpenCL device enriched with BVH build on CPU

For example see modified scene prt-teapot.pbrt.

Running PBRT:
run PBRT as usually - pbrt input_file.pbrt
