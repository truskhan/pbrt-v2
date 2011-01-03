
#ifndef PBRT_ACCELERATORS_NAIVE_H
#define PBRT_ACCELERATORS_NAIVE_H
/**
 * @file naive.h
 * @author: Hana Truskova hana.truskova@seznam.cz
 * @description: OpenCL naive accelerator class
**/
// accelerators/Naive.h*
#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

// NaiveAccel Declarations
class NaiveAccel : public Aggregate {
public:
  /**
   * The function returns a class variable, which value is precomputed.
   * @return bounding box containing the whole scene
  **/
  BBox WorldBound() const;
  /**
   * The class constructor
   * @param[in] p vector holding all scene geometry
   * @param[in] onG indicates whether the OpenCL code should be run on GPU or CPU
   * @param[in] method defines which GPU naive method should be used
  **/
  NaiveAccel(const vector<Reference<Primitive> > &p, bool onG, const string& method);
  /**
   * The function determines if the primitive can be intersected, remains from the parent - Primitive
   * @return says if the primitive contained in ray hierarchy can be tested for intersection
  **/
  bool CanIntersect() const { return true; }
  /**
   * Class desctructor
  **/
  ~NaiveAccel();
  /**
   * The function inspectes OpenCL device capability and says how many rays at once can trace
   * (tested only on GTX 275)
   * @return how many rays at once can trace on available OpenCL device
  **/
  unsigned int MaxRaysPerCall();
  /**
   * The function is used for rays after the first bounce
   * @param[in] r ray's array to test against scene geometry
   * @param[out] in intersection's array to fill with the information needed for shading
   * @param[out] hit array to fill with the information which ray missed the whole scene
   * @param[in] count number of rays to test
   * @param[in] bounce says how deep the input rays are (crutial for GPU_TIMES statistics)
  **/
  void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const int count, const int bounce = 0);
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
  void IntersectP(const Ray* ray, char* occluded, const size_t count, const bool* hit);

private:
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
    /** pointers to linear arrays holding vertices **/
    cl_float** vertices;
    /** pointers to linear arrays holding uvs **/
    cl_float** uvs;
    /** indicates wheter the algorithm should run on GPU **/
    bool onGPU;
    /** string defining which GPU naive version should be used **/
    string method;
    /** pointer to OpenCL auxiliary class, simplifying the OpenCL C API **/
    OpenCL* ocl;
    /** indicates which OpenCL command queue to use **/
    size_t cmd;
};


NaiveAccel *CreateNaiveAccelerator(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_Naive_H
