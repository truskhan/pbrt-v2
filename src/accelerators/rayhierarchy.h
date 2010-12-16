
#ifndef PBRT_ACCELERATORS_RAYHIERARCHY_H
#define PBRT_ACCELERATORS_RAYHIERARCHY_H

#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

// RayHieararchy Declarations
class RayHieararchy : public Aggregate {
public:
    // RayHieararchy Public Methods
    BBox WorldBound() const;
    RayHieararchy(const vector<Reference<Primitive> > &p,bool onG, int chunkX, int chunkY,
      int height, string node, bool sortVert, const string &sm, const int &maxBVHPrim, const unsigned int &repairRun
    #ifdef TRIANGLES_PER_THREAD
    , const int & trianlgesPerThread
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , int scale
    #endif
    );
    bool CanIntersect() const { return true; }
    ~RayHieararchy();
    void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const unsigned int count
    #ifdef STAT_RAY_TRIANGLE
    , Spectrum *Ls
    #endif
    );
    void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const unsigned int count, const int bounce
    );
    bool Intersect(const Ray &ray, Intersection *isect) const;
    bool IntersectP(const Ray &ray) const;
    void IntersectP(const Ray* ray, char * occluded, const size_t count, const bool* hit
    #ifdef STAT_PRAY_TRIANGLE
    , Spectrum *Ls
    #endif
    );
    unsigned int MaxRaysPerCall();
    void Preprocess(const Camera* camera, const unsigned samplesPerPixel);
    void Preprocess(const Camera* camera, const unsigned samplesPerPixel, const int nx, const int ny);
private:
    size_t ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, int *roffsetX, int *xWidth, int *yWidth);
    size_t ConstructRayHierarchyP(cl_float* rayDir, cl_float* rayO,int *roffsetX, int *xWidth, int *yWidth );
    bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;

    // RayHieararchy Private Methods
    vector<Reference<Primitive> > primitives;
    BBox bbox;
    size_t triangleCount;
    size_t trianglePartCount;
    size_t triangleLastPartCount;
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    int scale;
    #endif
    unsigned int repairRun;
    unsigned int parts;
    cl_float* vertices; cl_float* uvs;
    bool onGPU;
    cl_uint height;
    cl_uint chunkX, chunkY;
    OpenCL* ocl; //pointer to OpenCL auxiliary functions
    size_t cmd; //index to command queue
    unsigned a,b, global_a,global_b;
    unsigned samplesPerPixel;
    cl_uint threadsCount;
    cl_uint worgGroupSize;
    unsigned int rest_x, rest_y;
    mutable unsigned int xResolution;
    unsigned int yResolution;
    bool sortVert;
    Semaphore *workerSemaphore;
    #ifdef TRIANGLES_PER_THREAD
    int trianlgesPerThread;
    #endif
    size_t topLevelCount;
    string node;
    size_t nodeSize;
    #ifdef GPU_TIMES
    size_t primaryRays;
    size_t shadowRays;
    size_t secondaryRays;
    double intersectTimes[10];
    size_t bounceRays[10];
    #endif
    #ifdef STAT_RAY_TRIANGLE
    size_t intersectionCount;
    #endif
};


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_RAYHIERARCHY_H

