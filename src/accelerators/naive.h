
#ifndef PBRT_ACCELERATORS_NAIVE_H
#define PBRT_ACCELERATORS_NAIVE_H

// accelerators/Naive.h*
#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

// NaiveAccel Declarations
class NaiveAccel : public Aggregate {
public:
    // NaiveAccel Public Methods
    BBox WorldBound() const;
    NaiveAccel(const vector<Reference<Primitive> > &p, bool onG);
    bool CanIntersect() const { return true; }
    ~NaiveAccel();
    unsigned int MaxRaysPerCall();
    void Intersect(const RayDifferential *r, Intersection *in, bool* hit, const int counter, const unsigned int & samplesPerPixel);
    void Intersect(const RayDifferential *r, Intersection *in, bool* hit,
                   const int counter, const unsigned int & samplesPerPixel, const int bounce);
    bool Intersect(const Ray &ray, Intersection *isect) const;
    bool IntersectP(const Ray &ray) const;
    void IntersectP(const Ray* ray, char* occluded, const size_t count, const bool* hit);

private:
    bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;
    // NaiveAccel Private Methods
    vector<Reference<Primitive> > primitives;
    BBox bbox;
    size_t triangleCount;
    size_t trianglePartCount;
    size_t triangleLastPartCount;
    unsigned int parts;
    float* vertices; float* uvs;
    bool onGPU;
    OpenCL* ocl;
    size_t cmd; //index to command queue
    unsigned int samplesPerPixel;
};


NaiveAccel *CreateNaiveAccelerator(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_Naive_H
