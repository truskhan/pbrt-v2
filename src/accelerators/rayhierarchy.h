#ifndef PBRT_ACCELERATORS_RAYHIERARCHY_H
#define PBRT_ACCELERATORS_RAYHIERARCHY_H
/**
 * @file rayhierarchy.h
 * @author: Hana Truskova hana.truskova@seznam.cz
**/

#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

/**
 * RayHierarchy is class for holding necessery accelerator data and to provide functions
 * to build a rayhierarchy on GPU via OpenCL and to test mulitple rays against scene
 * primitives at once.
**/
class RayHieararchy : public Aggregate {
public:
  /**
   * The class constructor
   * @param[in] p vector holding all scene geometry
   * @param[in] onG indicates whether the OpenCL code should be run on GPU or CPU
   * @param[in] chunkX defines how many rays should be gathered in leaves in x axis direction
   * @param[in] chunkY defines how many rays should be gathered in leaves in y axis direction
   * @param[in] height defines how many ray hierarchy levels should be build
   * @param[in] node defines which node type should be used for ray hierarchy
   * @param[in] sortVert indicates whether the primitives should be sorted via BVH
   * @param[in] sm indicates which strategy to use when building BVH for sorting scene primitives
   * @param[in] maxBVHPrim indicates how many primitives can be left in BVH leaf
   * @param[in] repairRun indicates how many times should be run intersection method to
                eradicate algorithm failures
   * @param[in] scale indicates maximum scale factor when visualisating ray-triangle test figures
  **/
  RayHieararchy(const vector<Reference<Primitive> > &p,bool onG, int chunkX, int chunkY,
    int height, string node, bool sortVert, const string &sm, const int &maxBVHPrim,
    const unsigned int &repairRun
  #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
  , int scale
  #endif
  );
  /**
   * Class desctructor
  **/
  ~RayHieararchy();
  /**
   * The function returns a class variable, which value is precomputed.
   * @return bounding box containing the whole scene
  **/
  BBox WorldBound() const;
  /**
   * The function determines if the primitive can be intersected, remains from the parent - Primitive
   * @return says if the primitive contained in ray hierarchy can be tested for intersection
  **/
  bool CanIntersect() const { return true; }
  /**
   * The function is used for primary ray-triangle testes
   * @param[in] r primary ray's array to test against scene geometry
   * @param[out] in intersection's array to fill with the information needed for shading
   * @param[out] hit array to fill with the information which ray missed the whole scene
   * @param[in] count number of rays to test
   * @param[out] Ls result coulour used for ray-triangle intersection test's count (used rainbow colour mapping)
  **/
  void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const unsigned int count
  #ifdef STAT_RAY_TRIANGLE
  , Spectrum *Ls
  #endif
  );
  /**
   * The function is used for rays after the first bounce
   * @param[in] r ray's array to test against scene geometry
   * @param[out] in intersection's array to fill with the information needed for shading
   * @param[out] hit array to fill with the information which ray missed the whole scene
   * @param[in] count number of rays to test
   * @param[in] bounce says how deep the input rays are (crutial for GPU_TIMES statistics)
  **/
  void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const unsigned int count, const int bounce
  );
  /**
   * The function is used for only 1 ray-triangle intersection and
   * it computes the intersection naively
   * @param[in] ray ray to test against scene geometry
   * @param[out] isect intersection structure to fill with the information needed for shading
  **/
  bool Intersect(const Ray &ray, Intersection *isect) const;
  /**
    * The function is used for computing 1 shadow ray-triangle instersection
    * @param[in] ray shadow ray to test against scene geometry
    * @return indicates whether the ray reached the light (false) or hit some geometry (true)
  **/
  bool IntersectP(const Ray &ray) const;
  /**
   * The function is used for computing multiple shadow ray-triangle intersection
   * @param[in] ray shadow ray's array to test against scene geometry
   * @param[out] occluded array to indicate which shadow ray reached the light ('0')
                    and which hit some geometry ('1')
   * @param[in] count number of shadow rays to test
   * @param[in] hit array which indicates which shadow ray is valid
  **/
  void IntersectP(const Ray* ray, char * occluded, const size_t count, const bool* hit
  #ifdef STAT_PRAY_TRIANGLE
  , Spectrum *Ls
  #endif
  );
  /**
   * The function inspectes OpenCL device capability and says how many rays at once can trace
   * (tested only on GTX 275)
   * @return how many rays at once can trace on available OpenCL device
  **/
  unsigned int MaxRaysPerCall();
  /**
   * The function tells ray hierarchy the generated image dimension and samples per pixel,
   * so that the accelerator can allocate enough space on OpenCL device for processing the rays
  **/
    void Preprocess(const Camera* camera, const unsigned samplesPerPixel, const int nx, const int ny);
private:
  /**
   * Auxiliary function for constructing ray hierarchy on OpenCL device
   * @param[in] rayDir array holding ray directions
   * @param[in] rayO array holding ray origins
   * @param[out] roffsetX indicates where the ray hierarchy's top level starts in x dimension
   * @param[out] xWidth indicates width of the ray hierarchy's top level
   * @param[out] yWidth indicates height of the ray hierarchy's top level
  **/
  size_t ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, int *roffsetX, int *xWidth, int *yWidth);
  /**
   * Auxiliary function for constructing shadow ray hierarchy on OpenCL device
   * @param[in] rayDir array holding ray directions
   * @param[in] rayO array holding ray origins
   * @param[out] roffsetX indicates where the ray hierarchy's top level starts in x dimension
   * @param[out] xWidth indicates width of the ray hierarchy's top level
   * @param[out] yWidth indicates height of the ray hierarchy's top level
  **/
  size_t ConstructRayHierarchyP(cl_float* rayDir, cl_float* rayO,int *roffsetX, int *xWidth, int *yWidth );
  /**
   * Auxiliary function for computing ray-triangle intersection and parameters needed for shading,
   * used only when 1 ray is called for ray-triangle intersection (naive method)
  **/
  bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;

  /** vector for holding scene geometry **/
  vector<Reference<Primitive> > primitives;
  /** box enclosing all the scene primitives **/
  BBox bbox;
  /** total triangle count of scene geometry **/
  size_t triangleCount;
  /** defines into how many parts where scene geometry triangles devided  **/
  unsigned int parts;
  /** triangle part count of scene geometry - parted to fit in OpenCL device memory **/
  size_t trianglePartCount;
  /** last triangle part count of scene geometry **/
  size_t triangleLastPartCount;
  /** maximum value in rainbow colour mapping **/
  #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    int scale;
  #endif
  /** how many times should be run intersection kernel to repair algorithm errors
   * due to race condition
   **/
  unsigned int repairRun;
  /** pointer to linear array holding all triangle vertices **/
  cl_float* vertices;
  /** pointer to linear array holding all UVs **/
  cl_float* uvs;
  /** says if the code is run on GPU or CPU **/
  bool onGPU;
  /** says how many levels has the ray hierarchy **/
  cl_uint height;
  /** says how many rays are grouped in leaves in x direction **/
  cl_uint chunkX;
  /** says how many rays are grouped in leaves in y direction **/
  cl_uint chunkY;
  /** pointer to OpenCL auxiliary class, simplifying the OpenCL C API **/
  OpenCL* ocl;
  /** indicates which OpenCL command queue to use **/
  size_t cmd;
  /** indicates how many samples per pixel is used**/
  unsigned samplesPerPixel;
  /** generated picture width resolution (or width resolution of picutre part) **/
  mutable unsigned int xResolution;
  /** generated picture height resolution (or height resolution of picutre part) **/
  unsigned int yResolution;
  /** indicates whether the vertices should be sorted via BVH build **/
  bool sortVert;
  /** semaphore to run only one kernel at time on OpenCL device **/
  Semaphore *workerSemaphore;
  /** node type to use for ray hierarchy**/
  string node;
  #ifdef GPU_TIMES
    /** primary ray's count for the whole image **/
    size_t primaryRays;
    /** shadow ray's count for the whole image **/
    size_t shadowRays;
    /** total deeper ray's count **/
    size_t secondaryRays;
    /** kernel execution times for 0 .. 9 bounce rays **/
    double intersectTimes[10];
    /** number of rays in individual levels of ray-tracing recursion **/
    size_t bounceRays[10];
  #endif
    #ifdef STAT_RAY_TRIANGLE
    /** total number of ray-triangle test's count **/
    size_t intersectionCount;
    #endif
};

/** Method for creating ray-hierarchy accelerator.
 * @param[in] prims vector of all scene primitives
 * @param[in] ps parameteres from input PBRT file
 * return constructed ray hierarchy
 **/
RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_RAYHIERARCHY_H
