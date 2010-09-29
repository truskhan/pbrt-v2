//TODO: refactor the code and try tu run it!
// accelerators/Naive.cpp*
#include "accelerators/naive.h"
#include "probes.h"
#include "paramset.h"
#include "intersection.h"
#include "GPUparallel.h"
#include <iostream>

#define KERNEL_INTERSECTION 0
#define KERNEL_INTERSECTIONP 1

using namespace std;
Semaphore *workerSem;

// NaiveAccel Method Definitions
NaiveAccel::NaiveAccel(const vector<Reference<Primitive> > &p, bool onG) {
    triangleCount = 0;
    onGPU = onG;

    ocl = new OpenCL(onGPU,2);
    //precompile OpenCL kernels
    size_t lenP = strlen(PbrtOptions.pbrtPath);
    size_t *lenF = new size_t[2];
    char** names = new char*[2];
    char** file = new char*[2];

    names[0] = "cl/naive.cl";
    names[1] = "cl/naive.cl";

    for ( int i = 0; i < 2; i++){
      lenF[i] = strlen(names[i]);
      file[i] = new char[lenP + lenF[i] + 1];
      strncpy(file[i], PbrtOptions.pbrtPath, lenP);
      strncpy(file[i]+lenP, names[i], lenF[i]+1);
    }

    ocl->CompileProgram(file[0], "Intersection", "oclIntersection.ptx", KERNEL_INTERSECTION);
    ocl->CompileProgram(file[1], "IntersectionP", "oclIntersectionP.ptx", KERNEL_INTERSECTIONP);

    cmd = ocl->CreateCmdQueue();

    delete [] lenF;
    for (int i = 0; i < 2; i++){
      delete [] file[i];
    }
    delete [] names;
    delete [] file;

    for (uint32_t i = 0; i < p.size(); ++i)
        p[i]->FullyRefine(primitives);

    //store vertices and uvs in linear order
    vertices = new cl_float[3*3*primitives.size()];
    uvs = new cl_float[6*primitives.size()];
    for (uint32_t i = 0; i < primitives.size(); ++i) {
        const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (primitives[i].GetPtr()));
        if ( gp == 0 ) continue;
        const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
        const TriangleMesh* mesh = shape->GetMeshPtr();
        const Point &p1 = mesh->p[shape->v[0]];
        const Point &p2 = mesh->p[shape->v[1]];
        const Point &p3 = mesh->p[shape->v[2]];
         vertices[9*triangleCount+0] = p1.x;
         vertices[9*triangleCount+1] = p1.y;
         vertices[9*triangleCount+2] = p1.z;
         vertices[9*triangleCount+3] = p2.x;
         vertices[9*triangleCount+4] = p2.y;
         vertices[9*triangleCount+5] = p2.z;
         vertices[9*triangleCount+6] = p3[0];
         vertices[9*triangleCount+7] = p3[1];
         vertices[9*triangleCount+8] = p3[2];

        if (mesh->uvs) {
            uvs[6*triangleCount] = mesh->uvs[2*shape->v[0]];
            uvs[6*triangleCount+1] = mesh->uvs[2*shape->v[0]+1];
            uvs[6*triangleCount+2] = mesh->uvs[2*shape->v[1]];
            uvs[6*triangleCount+3] = mesh->uvs[2*shape->v[1]+1];
            uvs[6*triangleCount+4] = mesh->uvs[2*shape->v[2]];
            uvs[6*triangleCount+5] = mesh->uvs[2*shape->v[2]+1];
        } else { //todo - indicate this and compute at GPU
            uvs[6*triangleCount] = 0.f;
            uvs[6*triangleCount+1] = 0.f;
            uvs[6*triangleCount+2] = 1.f;
            uvs[6*triangleCount+3] = 0.f;
            uvs[6*triangleCount+4] = 1.f;
            uvs[6*triangleCount+5] = 1.f;
        }

        ++triangleCount;
    }

    // Compute bounds of all primitives
    for (uint32_t i = 0; i < p.size(); ++i) {

        bbox = Union(bbox, p[i]->WorldBound());
    }

    workerSem = new Semaphore(1);
}

NaiveAccel::~NaiveAccel() {
  ocl->DeleteCmdQueue(cmd);
  #ifdef GPU_TIMES
  ocl->PrintTimes();
  #endif
  delete ocl;
  delete [] vertices;
  delete [] uvs;
  delete workerSem;
}

BBox NaiveAccel::WorldBound() const {
    return bbox;
}

unsigned int NaiveAccel::MaxRaysPerCall(){
    //TODO: check the OpenCL device and decide, how many rays can be processed at one thread
    // check how many threads can be proccessed at once
    return 100000;
}

bool NaiveAccel::Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                           Vector &dpdu, Vector &dpdv, float& tu, float &tv,
                           float uvs[3][2], const Point p[3], float* coord) const {
    // Get triangle vertices in _p1_, _p2_, and _p3_
    const TriangleMesh* mesh = shape->GetMeshPtr();
    const Point &p1 = mesh->p[shape->v[0]];
    const Point &p2 = mesh->p[shape->v[1]];
    const Point &p3 = mesh->p[shape->v[2]];
    Vector e1 = p2 - p1;
    Vector e2 = p3 - p1;
    Vector s1 = Cross(ray.d, e2);
    float divisor = Dot(s1, e1);
    if (divisor == 0.)
        return false;
    float invDivisor = 1.f / divisor;

    // Compute first barycentric coordinate
    Vector d = ray.o - p1;
    float b1 = Dot(d, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        return false;

    // Compute second barycentric coordinate
    Vector s2 = Cross(d, e1);
    float b2 = Dot(ray.d, s2) * invDivisor;
    if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Compute _t_ to intersection point
    float t = Dot(e2, s2) * invDivisor;
    if (t < ray.mint || t > ray.maxt)
        return false;

    // Compute deltas for triangle partial derivatives
    float du1 = uvs[0][0] - uvs[2][0];
    float du2 = uvs[1][0] - uvs[2][0];
    float dv1 = uvs[0][1] - uvs[2][1];
    float dv2 = uvs[1][1] - uvs[2][1];
    Vector dp1 = p1 - p3, dp2 = p2 - p3;
    float determinant = du1 * dv2 - dv1 * du2;
    if (determinant == 0.f) {
        // Handle zero determinant for triangle partial derivative matrix
        CoordinateSystem(Normalize(Cross(e2, e1)), &dpdu, &dpdv);
    } else {
        float invdet = 1.f / determinant;
        dpdu = ( dv2 * dp1 - dv1 * dp2) * invdet;
        dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    }

    // Interpolate $(u,v)$ triangle parametric coordinates
    float b0 = 1 - b1 - b2;
    tu = b0*uvs[0][0] + b1*uvs[1][0] + b2*uvs[2][0];
    tv = b0*uvs[0][1] + b1*uvs[1][1] + b2*uvs[2][1];

    *tHit = t;
    *coord = du1;
    PBRT_RAY_TRIANGLE_INTERSECTION_HIT(const_cast<Ray *>(&ray), t);
    return true;
}

void NaiveAccel::Intersect(const RayDifferential *r, Intersection *in,
                               float* rayWeight, bool* hit, const int count, const unsigned int & samplesPerPixel)  {

    this->samplesPerPixel = samplesPerPixel;
   // count *= samplesPerPixel;
    size_t tn = ocl->CreateTask(KERNEL_INTERSECTION , triangleCount, cmd, 32);
    OpenCLTask* gput = ocl->getTask(tn);
    gput->InitBuffers(9);

    gput->CreateBuffer(0,sizeof(cl_float)*3*3*triangleCount, CL_MEM_READ_ONLY ); //for vertices
    gput->CreateBuffer(1,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY); //for ray directions
    gput->CreateBuffer(2,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY); //for ray origins
    gput->CreateBuffer(3,sizeof(cl_float)*2*count, CL_MEM_READ_ONLY); //for ray bounds
    gput->CreateBuffer(4,sizeof(cl_float)*6*triangleCount, CL_MEM_READ_ONLY); //for uvs
    gput->CreateBuffer(5,sizeof(cl_float)*count, CL_MEM_READ_WRITE); //for Thit
    gput->CreateBuffer(6,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY); //for tu,tv
    gput->CreateBuffer(7,sizeof(cl_float)*2*3*count, CL_MEM_WRITE_ONLY); //for dpdu, dpdv
    gput->CreateBuffer(8,sizeof(cl_uint)*count, CL_MEM_READ_WRITE); //index to shape

    gput->SetIntArgument(9, count);
    gput->SetIntArgument(10,(int&)triangleCount);

    cl_float* rayDirArray = new cl_float[count*3];//ray directions
    cl_float* rayOArray = new cl_float[count*3];//ray origins
    cl_float* rayBoundsArray = new cl_float[count*2];//ray bounds
    cl_float* tHitArray = new cl_float[count];
    cl_uint* indexArray = new cl_uint[count];//for indexs to shape
    rayDirArray = new cl_float[count*3];
    for (int k = 0; k < count; ++k) {
        rayDirArray[3*k] = r[k].d[0];
        rayDirArray[3*k+1] = r[k].d[1];
        rayDirArray[3*k+2] = r[k].d[2];

        rayOArray[3*k] = r[k].o[0];
        rayOArray[3*k+1] = r[k].o[1];
        rayOArray[3*k+2] = r[k].o[2];

        rayBoundsArray[2*k] = r[k].mint;
        rayBoundsArray[2*k+1] = INFINITY;

        tHitArray[k] = INFINITY;

        indexArray[k] = 0;
    }

    gput->EnqueueWriteBuffer( 0, vertices );
    gput->EnqueueWriteBuffer( 1, rayDirArray);
    gput->EnqueueWriteBuffer( 2, rayOArray);
    gput->EnqueueWriteBuffer( 3, rayBoundsArray);
    gput->EnqueueWriteBuffer( 4, uvs);
    gput->EnqueueWriteBuffer( 5, tHitArray);
    gput->EnqueueWriteBuffer( 8, indexArray);

    if (!gput->Run())exit(EXIT_FAILURE);

    data[6] = new float[2*count];
    data[7] = new float[2*3*count];
    gput->EnqueueReadBuffer(6, data[6]); //tu,tv
    gput->EnqueueReadBuffer(7, data[7]); //dpdu, dpdv
    gput->EnqueueReadBuffer(8, indexArray);
    gput->EnqueueReadBuffer(5, tHitArray);
    gput->WaitForRead();

    int index;

    Vector dpdu, dpdv;

    for ( int i = 0; i < count; i++) {
        index = indexArray[i];
        hit[i] = false;
        if ( !index ) continue;
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(((float*)data[7])[6*i],((float*)data[7])[6*i+1],((float*)data[7])[6*i+2]);
        dpdv = Vector(((float*)data[7])[6*i+3],((float*)data[7])[6*i+4],((float*)data[7])[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i], ((float*)data[6])[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i],((float*)data[6])[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * tHitArray[i]; //thit
        r[i].maxt = tHitArray[i];
        hit[i] = true;
    }

    ocl->delTask(tn,cmd);
    delete [] ((float*)data[6]);
    delete [] ((float*)data[7]);
    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] tHitArray;
    delete [] indexArray;
}

bool NaiveAccel::Intersect(const Ray &ray, Intersection *isect) const {
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->Intersect(ray,isect)) {
            hit = true;
        }
    }
    return hit;
}


bool NaiveAccel::IntersectP(const Ray &ray) const {
    Intersection isect;
    return Intersect(ray, &isect);
}

void NaiveAccel::IntersectP(const Ray* r, unsigned char* occluded, const size_t count, const bool* hit) {
   // size_t count = co * samplesPerPixel;
    size_t tn = ocl->CreateTask (KERNEL_INTERSECTIONP, count, cmd, 32);
    OpenCLTask* gput = ocl->getTask(tn);
    gput->InitBuffers(5);

    cl_float* rayDirArray = new cl_float[count*3];//ray directions
    cl_float* rayOArray = new cl_float[count*3];//ray origins
    cl_float* rayBoundsArray = new cl_float[count*2];//ray bounds
    unsigned char* temp = new unsigned char[count];
    unsigned int elem_counter = 0;
    for (int k = 0; k < count; ++k) {
        if ( !hit[k] ) continue; //not a valid ray
        rayDirArray[3*elem_counter] = r[k].d[0];
        rayDirArray[3*elem_counter+1] = r[k].d[1];
        rayDirArray[3*elem_counter+2] = r[k].d[2];

        rayOArray[3*elem_counter] = r[k].o[0];
        rayOArray[3*elem_counter+1] = r[k].o[1];
        rayOArray[3*elem_counter+2] = r[k].o[2];

        rayBoundsArray[2*elem_counter] = r[k].mint;
        rayBoundsArray[2*elem_counter+1] = INFINITY;

        ((unsigned char*)temp)[elem_counter] = '0';
        ++elem_counter;
    }

    gput->CreateBuffer(0,sizeof(cl_float)*3*3*triangleCount, CL_MEM_READ_ONLY ); //for vertices
    gput->CreateBuffer(1,sizeof(cl_float)*3*elem_counter, CL_MEM_READ_ONLY); //for ray directions
    gput->CreateBuffer(2,sizeof(cl_float)*3*elem_counter, CL_MEM_READ_ONLY); //for ray origins
    gput->CreateBuffer(3,sizeof(cl_float)*2*elem_counter, CL_MEM_READ_ONLY); //for ray bounds
    gput->CreateBuffer(4,sizeof(cl_uchar)*elem_counter, CL_MEM_READ_WRITE); //for Thit

    if (!gput->SetIntArgument(5,(int&)elem_counter)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument(6,(int&)triangleCount)) exit(EXIT_FAILURE);

    gput->EnqueueWriteBuffer( 0, vertices );
    gput->EnqueueWriteBuffer( 1, rayDirArray);
    gput->EnqueueWriteBuffer( 2, rayOArray);
    gput->EnqueueWriteBuffer( 3, rayBoundsArray);
    gput->EnqueueWriteBuffer( 4, temp);

    if (!gput->Run())exit(EXIT_FAILURE);

    gput->EnqueueReadBuffer( 4, occluded);
    gput->WaitForRead();

    elem_counter = 0;
    for (int k = 0; k < count; ++k) {
        if ( !hit[k] ) continue; //not a valid ray
        occluded[k] = temp[elem_counter];
        ++elem_counter;
    }

    ocl->delTask(tn,cmd);
    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] temp;
}


NaiveAccel *CreateNaiveAccelerator(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    return new NaiveAccel(prims,onGPU);
}

