
#ifndef PBRT_ACCELERATORS_RAYBVH_H
#define PBRT_ACCELERATORS_RAYBVH_H

#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "accelerators/bvh.h"
#include "GPUparallel.h"

// RayBVH Declarations
class RayBVH : public Aggregate {
public:
    // RayBVH Public Methods
    BBox WorldBound() const;
    RayBVH(const vector<Reference<Primitive> > &p,bool onG, int chunk, int height, int BVHheight, string node,
        int MaxBVHPrim,  const string &sm
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , int scale
    #endif
    );
    bool CanIntersect() const { return true; }
    ~RayBVH();
    void Intersect(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, const unsigned int count
    #ifdef STAT_RAY_TRIANGLE
    , Spectrum *Ls
    #endif
    );
    bool Intersect(const Ray &ray, Intersection *isect) const;
    bool IntersectP(const Ray &ray) const;
    void IntersectP(const Ray* ray, char * occluded, const size_t count, const bool* hit
    #ifdef STAT_PRAY_TRIANGLE
    , Spectrum *Ls
    #endif
    );
    unsigned int MaxRaysPerCall();
    void PreprocessP(const int rays);
    void Preprocess();
    void Preprocess(const Camera* camera, const unsigned samplesPerPixel);
    void Preprocess(const Camera* camera, const unsigned samplesPerPixel, const int nx, const int ny);
private:
    size_t ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, int *roffsetX, int *xWidth, int *yWidth);
    size_t ConstructRayHierarchyP(cl_float* rayDir, cl_float* rayO,int *roffsetX, int *xWidth, int *yWidth );
    bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;

    // RayBVH Private Methods
    BVHAccel* bvh;
    string splitMethod;
    size_t triangleCount;
    size_t trianglePartCount;
    size_t triangleLastPartCount;
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    int scale;
    #endif
    unsigned int parts;
    cl_float* vertices; cl_float* uvs;
    bool onGPU;
    cl_uint height;
    cl_uint BVHheight;
    cl_uint maxBVHPrim;
    cl_uint chunk;
    OpenCL* ocl; //pointer to OpenCL auxiliary functions
    size_t cmd; //index to command queue
    unsigned a,b, global_a,global_b;
    unsigned samplesPerPixel;
    cl_uint threadsCount;
    cl_uint worgGroupSize;
    unsigned int rest_x, rest_y;
    mutable unsigned int xResolution;
    unsigned int yResolution;
    Semaphore *workerSemaphore;

    size_t topLevelCount;
    string node;
    size_t nodeSize;
};


RayBVH *CreateRayBVH(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_RAYHIERARCHY_H

