#include "accelerators/rayhierarchy.h"
#include "core/probes.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/renderer.h"
#include "core/paramset.h"
#include "core/intersection.h"
#include "core/GPUparallel.h"
#include "accelerators/bvh.h"
#include <iostream>
#if ( defined STAT_RAY_TRIANGLE || defined STAT_TRIANGLE_CONE)
#include "imageio.h"
#endif

#define KERNEL_COUNT 9
#define HEADER_COUNT 4
#define KERNEL_COMPUTEDPTUTV 0
#define KERNEL_RAYLEVELCONSTRUCT 1
#define KERNEL_RAYCONSTRUCT 2
#define KERNEL_INTERSECTIONP 4
#define KERNEL_INTERSECTIONR 3
#define KERNEL_YETANOTHERINTERSECTION 5
#define KERNEL_RAYLEVELCONSTRUCTP 6
#define KERNEL_RAYCONSTRUCTP 7
#define KERNEL_INTERSECTION2 8

using namespace std;


// RayHieararchy Method Definitions
RayHieararchy::RayHieararchy(const vector<Reference<Primitive> > &p, bool onG, int chunkX, int chunkY,
  int height, string node, bool sortVert, const string &sm, const int &maxBVHPrim, const unsigned int& repairRun
    #ifdef TRIANGLES_PER_THREAD
    , const int & trianlgesPerThread
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , int scale
    #endif
  ) {
    this->chunkX = chunkX;
    this->chunkY = chunkY;
    this->height = height;
    this->sortVert = sortVert;
    this->repairRun = repairRun;
    #ifdef TRIANGLES_PER_THREAD
    this->trianlgesPerThread = trianlgesPerThread;
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    this->scale = scale;
    #endif
    triangleCount = 0;
    onGPU = onG;

    Info("Created OpenCL context");
    //precompile OpenCL kernels
    size_t lenP = strlen(PbrtOptions.pbrtPath);
    size_t *lenF = new size_t[KERNEL_COUNT];
    char** names = new char*[KERNEL_COUNT];
    char** file = new char*[KERNEL_COUNT];
    char** headerName = new char*[HEADER_COUNT];
    char** headerFile = new char*[HEADER_COUNT];
    int headerNum = 0;

  nodeSize = 0;
  headerName[headerNum++] = "cl/intersectAllLeaves.cl";
  if ( node == "ia"){
    names[0] = "cl/intersectionIA.cl";
    names[1] = "cl/intersectionP6DB.cl";
    names[2] = "cl/rayhconstruct6DB.cl";
    names[3] = "cl/levelConstruct6DB.cl";
    names[4] = "cl/yetAnotherIntersectionIA.cl";
    names[6] = "cl/levelConstructP6DB.cl";
    names[7] = "cl/rayhconstructP6DB.cl";
    names[8] = "cl/intersection6DB2.cl";
    headerName[headerNum++] = "cl/intersectsNodeIA.cl";
    cout << "accel nodes : IA" << endl;
    nodeSize = 13;
  }
  if ( node == "sphere_uv"){
    names[0] = "cl/intersection5D.cl";
    names[1] = "cl/intersectionP5D.cl";
    names[2] = "cl/rayhconstruct5D.cl";
    names[3] = "cl/levelConstruct5D.cl";
    names[4] = "cl/yetAnotherIntersection5D.cl";
    names[6] = "cl/levelConstructP5D.cl";
    names[7] = "cl/rayhconstructP5D.cl";
    names[8] = "cl/intersection5D2.cl";
    headerName[headerNum++] = "cl/intersectsNode5DS.cl";
    cout << "accel nodes : 5D nodes with spheres" << endl;
    nodeSize = 9;
  }
  if ( node == "box_uv") {
    names[0] = "cl/intersection5DB.cl";
    names[1] = "cl/intersectionP5DB.cl";
    names[2] = "cl/rayhconstruct5DB.cl";
    names[3] = "cl/levelConstruct5DB.cl";
    names[4] = "cl/yetAnotherIntersection5DB.cl";
    names[6] = "cl/levelConstructP5DB.cl";
    names[7] = "cl/rayhconstructP5DB.cl";
    names[8] = "cl/intersection5DB2.cl";
    headerName[headerNum++] = "cl/intersectsNode5DB.cl";
    cout << "accel nodes : 5D nodes with boxes" << endl;
    nodeSize = 11;
  }
  if ( node == "box_dir") {
    names[0] = "cl/intersection6DB.cl";
    names[1] = "cl/intersectionP6DB.cl";
    names[2] = "cl/rayhconstruct6DB.cl";
    names[3] = "cl/levelConstruct6DB.cl";
    names[4] = "cl/yetAnotherIntersection6DB.cl";
    names[6] = "cl/levelConstructP6DB.cl";
    names[7] = "cl/rayhconstructP6DB.cl";
    names[8] = "cl/intersection6DB2.cl";
    headerName[headerNum++] = "cl/intersectsNode6DB.cl";
    cout << "accel nodes : 6D nodes with boxes" << endl;
    nodeSize = 13;
  }
  if ( nodeSize == 0){
    cout << "Unknown accelarator node type " << node << endl;
    abort();
  }
    names[5] = "cl/computeDpTuTv.cl";

    for ( int i = 0; i < KERNEL_COUNT; i++){
      lenF[i] = strlen(names[i]);
      file[i] = new char[lenP + lenF[i] + 1];
      strncpy(file[i], PbrtOptions.pbrtPath, lenP);
      strncpy(file[i]+lenP, names[i], lenF[i]+1);
    }

    for ( int i = 0; i < headerNum; i++){
      headerFile[i] = new char[lenP + strlen(headerName[i]) + 1];
      strncpy(headerFile[i], PbrtOptions.pbrtPath, lenP);
      strncpy(headerFile[i] + lenP, headerName[i], strlen(headerName[i]) + 1);
    }

    ocl = new OpenCL(onGPU,KERNEL_COUNT);
    cmd = ocl->CreateCmdQueue();
    ocl->CompileProgram(file[0], "IntersectionR", "oclIntersection.ptx", KERNEL_INTERSECTIONR, headerFile, headerNum);
    ocl->CompileProgram(file[2], "rayhconstruct", "oclRayhconstruct.ptx",KERNEL_RAYCONSTRUCT);
    ocl->CompileProgram(file[1], "IntersectionP", "oclIntersectionP.ptx", KERNEL_INTERSECTIONP, headerFile, headerNum);
    ocl->CompileProgram(file[3], "levelConstruct", "oclLevelConstruct.ptx",KERNEL_RAYLEVELCONSTRUCT);
    ocl->CompileProgram(file[4], "YetAnotherIntersection", "oclYetAnotherIntersection.ptx", KERNEL_YETANOTHERINTERSECTION, headerFile, headerNum);
    ocl->CompileProgram(file[5], "computeDpTuTv", "oclcomputeDpTuTv.ptx", KERNEL_COMPUTEDPTUTV);
    ocl->CompileProgram(file[6], "levelConstructP", "oclLevelConstructP.ptx",KERNEL_RAYLEVELCONSTRUCTP);
    ocl->CompileProgram(file[7], "rayhconstructP", "oclRayhconstructP.ptx",KERNEL_RAYCONSTRUCTP);
    ocl->CompileProgram(file[8], "IntersectionR2", "oclIntersection2.ptx", KERNEL_INTERSECTION2, headerFile, headerNum);

    delete [] lenF;
    for (int i = 0; i < KERNEL_COUNT; i++){
      //delete [] names[i];
      delete [] file[i];
    }
    delete [] names;
    delete [] file;
    for ( int i = 0; i < headerNum; i++)
      delete [] headerFile[i];
    delete [] headerFile;
    delete [] headerName;

    if ( sortVert){
      BVHAccel* bvh = new BVHAccel(p, maxBVHPrim, sm);
      primitives.swap(bvh->primitives);
      delete bvh;
    }
    else{
    for (uint32_t i = 0; i < p.size(); ++i)
        p[i]->FullyRefine(primitives);
    }
    //store vertices and uvs in linear order
    vertices = new cl_float[3*3*primitives.size()];
    uvs = new cl_float[6*primitives.size()];
    for (uint32_t i = 0; i < primitives.size(); ++i) {
        const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (primitives[i].GetPtr()));
        if ( gp == 0 ) continue;
        const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
        if ( shape == 0) continue;
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
    //TODO: check how many threads can be proccessed at once (depends on MaxRaysPerCall)
    cout << "Scene contains " << triangleCount << " triangles"<< endl;
    workerSemaphore = new Semaphore(1);
    a = chunkX;
    b = chunkY;

    #ifdef GPU_TIMES
    primaryRays = 0;
    shadowRays = 0;
    secondaryRays = 0;
    #endif
}

RayHieararchy::~RayHieararchy() {
  #ifdef GPU_TIMES
  ocl->PrintTimes();
  cout << "primary rays: " << primaryRays << endl;
  cout << "shadow rays: " << shadowRays << endl;
  cout << "secondary rays: " << secondaryRays << endl;
  #endif
  delete ocl;
  delete [] vertices;
  delete [] uvs;
  delete workerSemaphore;
}

BBox RayHieararchy::WorldBound() const {
    return bbox;
}

void RayHieararchy::Preprocess(const Camera* camera, const unsigned samplesPerPixel,
  const int nx, const int ny){
  xResolution /= nx;
  yResolution /= ny;

  //number of rectangles in x axis
  global_a = (xResolution + a - 1) / a; //round up -> +a-1
  //number of rectangles in y axis
  global_b = (yResolution + b - 1) / b;
  //x and y sizes of overlapping area
  rest_x = global_a*a - xResolution;
  rest_y = global_b*b - yResolution;
  threadsCount = global_a * global_b;
}

void RayHieararchy::Preprocess(const Camera* camera, const unsigned samplesPerPixel){
  this->xResolution = camera->film->xResolution;
  this->yResolution = camera->film->yResolution;
  this->samplesPerPixel = samplesPerPixel;
  //number of rectangles in x axis
  global_a = (xResolution + a - 1) / a; //round up -> +a-1
  //number of rectangles in y axis
  global_b = (yResolution + b - 1) / b;
  //x and y sizes of overlapping area
  rest_x = global_a*a - xResolution;
  rest_y = global_b*b - yResolution;
  threadsCount = global_a * global_b;
}

unsigned int RayHieararchy::MaxRaysPerCall(){
    worgGroupSize = 64;

    //TODO: check the OpenCL device and decide, how many rays can be processed at one thread
    // check how many threads can be proccessed at once
    cl_ulong gms = ocl->getGlobalMemSize();
    //ray dir,o, bounds, tHit
    unsigned int x;

    //vertices
    #define MAX_VERTICES 40000
    if ( triangleCount > MAX_VERTICES) {
      parts = (triangleCount + MAX_VERTICES -1 )/MAX_VERTICES;
      trianglePartCount = (triangleCount + parts - 1)/ parts;
      triangleLastPartCount = triangleCount - (parts-1)*trianglePartCount;
    } else {
      parts = 1;
      trianglePartCount = triangleCount;
      triangleLastPartCount = triangleCount;
    }

    //(GlobalMemorySize - vertices - pointersToChildren - counts)
	x = (gms - sizeof(cl_float)*(9+6)*trianglePartCount //vertices + uvs
             - sizeof(cl_uint)*threadsCount //counts
             ) / (
             sizeof(cl_uint) //index
             + 9*sizeof(cl_float) //ray dir, o, bounds, tHit
             );

    //local mem - stack:
    cl_ulong lms = ocl->getLocalMemSize();
    cl_ulong localSize = sizeof(cl_int)*(2 + (height+1)*(height+2)/2)*worgGroupSize;
    if ( lms < localSize){
      Severe("Need local memory size at least %i Bytes, present %i B. Try to decrease hierarchy's height.", localSize, lms);
    }


    cout << "Global memory size on OpenCL device (in B): " << gms << endl;
    cout << "Maximum memory allocation size at once: " << ocl->getMaxMemAllocSize() << endl;
    cout << "Local memory size on OpenCL device: " << ocl->getLocalMemSize() << " needed: " << localSize << endl;
    cout << "Constant memory size: " << ocl->getMaxConstantBufferSize() << endl;
    cout << "Max work group size: " << ocl->getMaxWorkGroupSize() << endl;

    cout << "Needed rays " << xResolution*yResolution*samplesPerPixel << " device maximu rays " << x << endl;
    x = 60000;
    return min(x, xResolution*yResolution*samplesPerPixel);
}

//classical method for testing one ray
bool RayHieararchy::Intersect(const Triangle* shape, const Ray &ray, float *tHit,
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

//constructs ray hieararchy on GPU -> creates array of cones
size_t RayHieararchy::ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, int *roffsetX, int *xWidth, int *yWidth ){
  Assert(height > 0);
  size_t globalSize[2] = {(xResolution*samplesPerPixel + a -1)/a, (yResolution+b-1)/b};
  size_t localSize[2] = {8,8};
  size_t tn = ocl->CreateTask(KERNEL_RAYCONSTRUCT, 2,  globalSize , localSize, cmd);
  OpenCLTask* gpuray = ocl->getTask(tn,cmd);

  gpuray->InitBuffers(3);

  cl_image_format imageFormat;
  imageFormat.image_channel_data_type = CL_FLOAT;
  imageFormat.image_channel_order = CL_RGBA;

  gpuray->CreateImage2D(0, CL_MEM_READ_ONLY , &imageFormat, xResolution*samplesPerPixel, yResolution, 0);
  gpuray->CreateImage2D(1, CL_MEM_READ_ONLY , &imageFormat, xResolution*samplesPerPixel, yResolution, 0);
  gpuray->CreateImage2D(2, CL_MEM_WRITE_ONLY, &imageFormat, 2*globalSize[0], 2*globalSize[1], 0); //for hierarchy nodes
  gpuray->SetIntArgument(3, globalSize[0]);
  gpuray->SetIntArgument(4, globalSize[1]);
  gpuray->SetIntArgument(5, a);
  gpuray->SetIntArgument(6, b);

  gpuray->EnqueueWrite2DImage(0, rayDir);
  gpuray->EnqueueWrite2DImage(1, rayO);

  gpuray->Run();
  //cl_float* nodes = new cl_float[4*xResolution*samplesPerPixel*yResolution];
  //gpuray->EnqueueRead2DImage(2, nodes);
  //gpuray->WaitForRead();
  gpuray->WaitForKernel();

  gpuray->SetPersistentBuff(0);
  gpuray->SetPersistentBuff(1);
  gpuray->SetPersistentBuff(2);

  *roffsetX = 0;
  int woffsetX = globalSize[0];

  size_t gws[2] = {globalSize[0]/2, globalSize[1]/2};
  size_t tasknum = ocl->CreateTask(KERNEL_RAYLEVELCONSTRUCT, 2, gws, localSize, cmd);
  OpenCLTask* gpurayl = ocl->getTask(tasknum,cmd);
  gpurayl->InitBuffers(2);
  gpurayl->CopyBuffer(2,0,gpuray);
  gpurayl->CreateImage2D(1, CL_MEM_READ_ONLY, &imageFormat, 2*globalSize[0], 2*globalSize[1], 0);

  for ( cl_uint i = 1; i < height; i++){
    gpurayl->CopyImage2D(0,1,gpurayl);
    gpurayl->SetIntArgument(2, *roffsetX);
    gpurayl->SetIntArgument(3, woffsetX);
    gpurayl->SetIntArgument(4, globalSize[0]); //width
    gpurayl->SetIntArgument(5, globalSize[1]); //height
    ocl->Finish();

    gpurayl->Run();

    globalSize[0] /= 2;
    globalSize[1] /= 2;
    *roffsetX = woffsetX;
    woffsetX += globalSize[0];

    gpurayl->WaitForKernel();
    //gpurayl->EnqueueRead2DImage(0, nodes);
    //gpuray->WaitForRead();

    //if one size is even or 1 end with building the hierarchy
    if ( globalSize[0] & 0x1 || globalSize[1] & 0x1 ){
      cout << "Hierarchy height only " << i << endl;
      break;
    }
  }

  /*for ( int i = 0; i < yResolution; i++){
    for ( int j = 0; j < xResolution*samplesPerPixel; j++){
      cout << nodes[4*i*xResolution*samplesPerPixel + 4*j] << " "
           << nodes[4*i*xResolution*samplesPerPixel + 4*j + 1] << " "
           << nodes[4*i*xResolution*samplesPerPixel + 4*j + 2] << " "
           << nodes[4*i*xResolution*samplesPerPixel + 4*j + 3] << " ";
    }
    cout << endl;
  }*/

  *xWidth = globalSize[0];
  *yWidth = globalSize[1];
  //*roffsetX -= *xWidth;

  gpurayl->SetPersistentBuff(0);
  ocl->delTask(tasknum, cmd);

  return tn; //return index to first task - so that buffers can be copied
}

//constructs ray hieararchy on GPU -> creates array of cones
size_t RayHieararchy::ConstructRayHierarchyP(cl_float* rayDir, cl_float* rayO,
  int *roffsetX, int *xWidth, int *yWidth ){
  Assert(height > 0);
  size_t globalSize[2] = {(xResolution*samplesPerPixel + a -1)/a, (yResolution+b-1)/b};
  size_t localSize[2] = {8,8};
  size_t tn = ocl->CreateTask(KERNEL_RAYCONSTRUCTP, 2,  globalSize , localSize, cmd);
  OpenCLTask* gpuray = ocl->getTask(tn,cmd);

  gpuray->InitBuffers(4);

  cl_image_format imageFormat;
  imageFormat.image_channel_data_type = CL_FLOAT;
  imageFormat.image_channel_order = CL_RGBA;

  cl_image_format hitFormat;
  hitFormat.image_channel_data_type = CL_SIGNED_INT32;
  hitFormat.image_channel_order = CL_R;

  gpuray->CreateImage2D(0, CL_MEM_READ_ONLY , &imageFormat, xResolution*samplesPerPixel, yResolution, 0);
  gpuray->CreateImage2D(1, CL_MEM_READ_ONLY , &imageFormat, xResolution*samplesPerPixel, yResolution, 0);
  gpuray->CreateImage2D(2, CL_MEM_WRITE_ONLY, &imageFormat, 2*globalSize[0], 2*globalSize[1], 0); //for hierarchy nodes
  gpuray->CreateImage2D(3, CL_MEM_WRITE_ONLY, &hitFormat, 2*globalSize[0], globalSize[1],0); //for storing info about validity
  gpuray->SetIntArgument(4, globalSize[0]);
  gpuray->SetIntArgument(5, globalSize[1]);
  gpuray->SetIntArgument(6, a);
  gpuray->SetIntArgument(7, b);

  gpuray->EnqueueWrite2DImage(0, rayDir);
  gpuray->EnqueueWrite2DImage(1, rayO);

  gpuray->Run();
  gpuray->WaitForKernel();

  gpuray->SetPersistentBuff(0);
  gpuray->SetPersistentBuff(1);
  gpuray->SetPersistentBuff(2);
  gpuray->SetPersistentBuff(3);

  *roffsetX = 0;
  int woffsetX = globalSize[0];

  size_t gws[2] = {globalSize[0]/2, globalSize[1]/2};
  size_t tasknum = ocl->CreateTask(KERNEL_RAYLEVELCONSTRUCTP, 2, gws, localSize, cmd);
  OpenCLTask* gpurayl = ocl->getTask(tasknum,cmd);
  gpurayl->InitBuffers(4);
  gpurayl->CopyBuffer(2,0,gpuray);
  gpurayl->CopyBuffer(3,1,gpuray);
  gpurayl->CreateImage2D(2, CL_MEM_READ_ONLY, &imageFormat, 2*globalSize[0], 2*globalSize[1], 0);
  gpurayl->CreateImage2D(3, CL_MEM_READ_ONLY, &hitFormat, 2*globalSize[0], globalSize[1], 0);

  for ( cl_uint i = 1; i < height; i++){
    gpurayl->CopyImage2D(0,2,gpurayl);
    gpurayl->CopyImage2D(1,3,gpurayl);
    gpurayl->SetIntArgument(4, *roffsetX);
    gpurayl->SetIntArgument(5, woffsetX);
    gpurayl->SetIntArgument(6, globalSize[0]); //width
    gpurayl->SetIntArgument(7, globalSize[1]); //height
    ocl->Finish();

    gpurayl->Run();

    globalSize[0] /= 2;
    globalSize[1] /= 2;
    *roffsetX = woffsetX;
    woffsetX += globalSize[0];

    gpurayl->WaitForKernel();

    //if one size is even or 1 end with building the hierarchy
    if ( globalSize[0] & 0x1 || globalSize[1] & 0x1 ){
      cout << "Hierarchy height only " << i << endl;
      break;
    }
  }

  *xWidth = globalSize[0];
  *yWidth = globalSize[1];

  gpurayl->SetPersistentBuff(0);
  gpurayl->SetPersistentBuff(1);
  ocl->delTask(tasknum, cmd);

  return tn; //return index to first task - so that buffers can be copied
}

#if defined STAT_PRAY_TRIANGLE || defined STAT_RAY_TRIANGLE
#define MAX_COLOR_VALUE 255
#define MIN_COLOR_VALUE 0
inline RGBSpectrum RainbowColorMapping(const float _value)
{
  float color[3];

  float value = 4.0f*(1.0f - (_value));

  if (value < 0.0f)
        value = 0.0f;
  else
        if (value > 4.0f)
          value = 4.0f;

  int band = (int)(value);
  value -= band;

  switch (band) {
  case 0:
        color[0] = MAX_COLOR_VALUE;
        color[1] = value*MAX_COLOR_VALUE;
        color[2] = MIN_COLOR_VALUE;
        break;
  case 1:
        color[0] = (1.0f - value)*MAX_COLOR_VALUE;
        color[1] = MAX_COLOR_VALUE;
        color[2] = MIN_COLOR_VALUE;
        break;
  case 2:
        color[0] = MIN_COLOR_VALUE;
        color[1] = MAX_COLOR_VALUE;
        color[2] = value*MAX_COLOR_VALUE;
        break;
  case 3:
        color[0] = MIN_COLOR_VALUE;
        color[1] = (1.0f - value)*MAX_COLOR_VALUE;
        color[2] = MAX_COLOR_VALUE;
        break;
  default:
        color[0] = value*MAX_COLOR_VALUE;
        color[1] = MIN_COLOR_VALUE;
        color[2] = MAX_COLOR_VALUE;
        break;
  }

  return RGBSpectrum::FromRGB(color);
}
#endif

//intersect computed on gpu with more rays (not primary rays)
void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                    bool* hit, const unsigned int count, const int bounce
    ){
  cl_float* rayDirArray = new cl_float[count*4];
  cl_float* rayOArray = new cl_float[count*4];
  cl_float* rayBoundsArray = new cl_float[count*2];
  cl_float* tHitArray = new cl_float[count];
  cl_int* indexArray = new cl_int[count];

  for (unsigned int k = 0; k < count; ++k){
    rayDirArray[4*k] = r[k].d[0];
    rayDirArray[4*k + 1] = r[k].d[1];
    rayDirArray[4*k + 2] = r[k].d[2];
    rayDirArray[4*k + 3] = 1;

    rayOArray[4*k] = r[k].o[0];
    rayOArray[4*k + 1] = r[k].o[1];
    rayOArray[4*k + 2] = r[k].o[2];
    rayOArray[4*k + 3] = 0;

    rayBoundsArray[2*k] = r[k].mint;
    rayBoundsArray[2*k + 1] = INFINITY;

    indexArray[k] = -1;
    tHitArray[k] = INFINITY; //should initialize on scene size

    if ( !hit[k] ) //indicate that the ray is invalid
      rayDirArray[4*k + 3] = -1;
  }

  workerSemaphore->Wait();
  #ifdef GPU_TIMES
  for ( unsigned int k = 0; k < count; ++k){
    if ( hit[k]) ++secondaryRays;
  }
  #endif
  int roffsetX, xWidth, yWidth;
  size_t tn1 = ConstructRayHierarchyP(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
  OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

  size_t gws = trianglePartCount;
  size_t lws = 64;
  size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTION2, 1, &gws, &lws, cmd);
  OpenCLTask* gput = ocl->getTask(tn2,cmd);

  unsigned int c = 9;
  gput->InitBuffers(c);

  gput->CopyBuffer(0,1,gpuray); //ray dir
  gput->CopyBuffer(1,2,gpuray); //ray o
  gput->CopyBuffer(2,3,gpuray); //nodes
  gput->CopyBuffer(3,4,gpuray); //ray validity
  ocl->delTask(tn1,cmd);

  cl_image_format imageFormatBounds;
  imageFormatBounds.image_channel_data_type = CL_FLOAT;
  imageFormatBounds.image_channel_order = CL_RG;

  gput->CreateBuffer(0,sizeof(cl_float)*3*3*trianglePartCount, CL_MEM_READ_ONLY ); //vertices
  gput->CreateImage2D(5, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0); //ray bounds
  gput->CreateBuffer(6,sizeof(cl_float)*count, CL_MEM_WRITE_ONLY); //tHit
  gput->CreateBuffer(7,sizeof(cl_int)*count, CL_MEM_WRITE_ONLY); //index array
  gput->CreateBuffer(8,sizeof(cl_int)*(gws+lws)*5*(xWidth*yWidth+5*height), CL_MEM_READ_WRITE); //stack
  gput->SetIntArgument(9, roffsetX);
  gput->SetIntArgument(10, xWidth);
  gput->SetIntArgument(11, yWidth);
  gput->SetIntArgument(12,a);
  gput->SetIntArgument(13,b);
  gput->SetIntArgument(14, trianglePartCount);
  gput->SetIntArgument(16, 5*gws); //stack size

  gput->EnqueueWrite2DImage(5, rayBoundsArray);
  gput->EnqueueWriteBuffer(6, tHitArray);
  gput->EnqueueWriteBuffer(7, indexArray);
  for ( int i = 0; i < parts - 1; i++){
    gput->SetIntArgument(15,(cl_uint)i*trianglePartCount); //offset
    gput->EnqueueWriteBuffer(0, vertices + 9*i*trianglePartCount);
    gput->Run();
    gput->WaitForKernel();
  }
  //last part of vertices
  gput->SetIntArgument(14,(cl_int)triangleLastPartCount);
  gput->SetIntArgument(15,(cl_uint)(parts-1)*trianglePartCount);
  gput->EnqueueWriteBuffer(0, vertices + 9*(parts-1)*trianglePartCount, sizeof(cl_float)*9*triangleLastPartCount);
  gput->Run();

  gput->EnqueueReadBuffer( 6, tHitArray );
  gput->EnqueueReadBuffer( 7, indexArray);

  gput->SetPersistentBuffers(0,3);
  gput->SetPersistentBuff(7);

  gput->WaitForRead();
  gws = count;
  size_t tn3 = ocl->CreateTask(KERNEL_COMPUTEDPTUTV, 1, &gws, &lws, cmd);
  OpenCLTask* gpuRayO = ocl->getTask(tn3,cmd);
  gpuRayO->InitBuffers(7);
  gpuRayO->CopyBuffers(0,3,0,gput); // 0 vertex, 1 dir, 2 origin
  gpuRayO->CopyBuffer(7,3,gput); // 3 index
  ocl->delTask(tn2,cmd);
  gpuRayO->CreateBuffer(4,sizeof(cl_float)*6*trianglePartCount, CL_MEM_READ_ONLY ); //uvs
  gpuRayO->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY ); // tu,tv
  gpuRayO->CreateBuffer(6,sizeof(cl_float)*6*count, CL_MEM_WRITE_ONLY); //dpdu, dpdv
  gpuRayO->SetIntArgument(10,xResolution*samplesPerPixel);

  gpuRayO->SetIntArgument(7,(cl_uint)count);

  for ( int i = 0; i < parts - 1; i++){
    gpuRayO->EnqueueWriteBuffer( 4 , uvs + 6*i*trianglePartCount);
    gpuRayO->SetIntArgument(8,i*trianglePartCount);
    gpuRayO->SetIntArgument(9,(i+1)*trianglePartCount);
    gpuRayO->EnqueueWriteBuffer(0, vertices + 9*i*trianglePartCount);
    gpuRayO->Run();
    gpuRayO->WaitForKernel();
  }
  gpuRayO->SetIntArgument(8,(cl_int)(parts-1)*trianglePartCount);
  gpuRayO->SetIntArgument(9,(cl_int)triangleCount);
  gpuRayO->EnqueueWriteBuffer(4, uvs + 6*(parts-1)*trianglePartCount, 6*sizeof(cl_float)*triangleLastPartCount);
  gpuRayO->EnqueueWriteBuffer(0, vertices + 9*(parts-1)*trianglePartCount,9*sizeof(cl_float)*triangleLastPartCount);
  gpuRayO->Run();

  cl_float* tutvArray = new cl_float[2*count]; // for tu, tv
  cl_float* dpduArray = new cl_float[2*3*count]; // for dpdu, dpdv
  gpuRayO->EnqueueReadBuffer( 5, tutvArray );
  gpuRayO->EnqueueReadBuffer( 6, dpduArray );

  gpuRayO->WaitForRead();
  ocl->delTask(tn3,cmd);
  workerSemaphore->Post();
  cl_uint index;

  Vector dpdu, dpdv;
  //deserialize rectangles
  for ( unsigned int i = 0; i < count; i++) {
      if ( hit[i] == false) continue;
      index = indexArray[i];
      hit[i] = false;
      if ( index == -1 ) continue;
      Assert( index < triangleCount);
      const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
      const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

      dpdu = Vector(dpduArray[6*i],dpduArray[6*i+1],dpduArray[6*i+2]);
      dpdv = Vector(dpduArray[6*i+3],dpduArray[6*i+4],dpduArray[6*i+5]);

      // Test intersection against alpha texture, if present
      if (shape->GetMeshPtr()->alphaTexture) {
          DifferentialGeometry dgLocal(r[i](tHitArray[i]), dpdu, dpdv,
                                       Normal(0,0,0), Normal(0,0,0),
                                       tutvArray[2*i], tutvArray[2*i+1], shape);
          if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
              continue;
      }
      // Fill in _DifferentialGeometry_ from triangle hit
      in[i].dg =  DifferentialGeometry(r[i](tHitArray[i]), dpdu, dpdv,
                                       Normal(0,0,0), Normal(0,0,0),
                                       tutvArray[2*i],tutvArray[2*i+1], shape);
      in[i].primitive = p;
      in[i].WorldToObject = *shape->WorldToObject;
      in[i].ObjectToWorld = *shape->ObjectToWorld;
      in[i].shapeId = shape->shapeId;
      in[i].primitiveId = p->primitiveId;
      in[i].rayEpsilon = 1e-3f * tHitArray[i]; //thit
      r[i].maxt = tHitArray[i];
      hit[i] = true;
      PBRT_RAY_TRIANGLE_INTERSECTION_HIT(&r[i], r[i].maxt);
  }

  delete [] rayDirArray;
  delete [] rayOArray;
  delete [] rayBoundsArray;
  delete [] tHitArray;
  delete [] tutvArray;
  delete [] dpduArray;
  delete [] indexArray;
}

//intersect computed on gpu with more rays
void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                               bool* hit, unsigned int count
  #ifdef STAT_RAY_TRIANGLE
  , Spectrum *Ls
  #endif
  )  {

  cl_float* rayDirArray = new cl_float[count*4];
  cl_float* rayOArray = new cl_float[count*4];
  cl_float* rayBoundsArray = new cl_float[count*2];
  cl_float* tHitArray = new cl_float[count];
  cl_int* indexArray = new cl_int[count];
  #ifdef STAT_RAY_TRIANGLE
  cl_int* picture = new cl_int[count];
  #endif
  unsigned int c;

  for (unsigned int k = 0; k < count; ++k){
    rayDirArray[4*k] = r[k].d[0];
    rayDirArray[4*k + 1] = r[k].d[1];
    rayDirArray[4*k + 2] = r[k].d[2];
    rayDirArray[4*k + 3] = 0;

    rayOArray[4*k] = r[k].o[0];
    rayOArray[4*k + 1] = r[k].o[1];
    rayOArray[4*k + 2] = r[k].o[2];
    rayOArray[4*k + 3] = 0;

    rayBoundsArray[2*k] = r[k].mint;
    rayBoundsArray[2*k + 1] = INFINITY;

    indexArray[k] = -1;
    tHitArray[k] = INFINITY; //should initialize on scene size

    #ifdef STAT_RAY_TRIANGLE
    picture[k] = 0;
    #endif
  }

    workerSemaphore->Wait();
    #ifdef GPU_TIMES
    primaryRays += count;
    #endif
    int roffsetX, xWidth, yWidth;
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    size_t gws;
    gws = trianglePartCount;
    #ifdef TRIANGLES_PER_THREAD
     gws = (trianglePartCount + trianlgesPerThread - 1)/trianlgesPerThread;
    #endif
    size_t lws = 64;
    size_t tn2 = ocl->CreateTask(KERNEL_INTERSECTIONR, 1, &gws, &lws, cmd);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);

    #ifdef STAT_RAY_TRIANGLE
    c = 9;
    #else
    c = 8;
    #endif
    gput->InitBuffers(c);

    gput->CopyBuffer(0,1,gpuray); //ray dir
    gput->CopyBuffer(1,2,gpuray); //ray o
    gput->CopyBuffer(2,3,gpuray); //nodes
    ocl->delTask(tn1,cmd);

    cl_image_format imageFormatBounds;
    imageFormatBounds.image_channel_data_type = CL_FLOAT;
    imageFormatBounds.image_channel_order = CL_RG;

    gput->CreateBuffer(0,sizeof(cl_float)*3*3*trianglePartCount, CL_MEM_READ_ONLY ); //vertices
    gput->CreateImage2D(4, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0);
    gput->CreateBuffer(5,sizeof(cl_float)*count, CL_MEM_READ_WRITE); // tHit
    gput->CreateBuffer(6,sizeof(cl_int)*count, CL_MEM_WRITE_ONLY); //index array
    //allocate stack in global memory
    gput->CreateBuffer(7,sizeof(cl_int)*gws*5*(xWidth*yWidth+4*height), CL_MEM_READ_WRITE);
   // gput->SetLocalArgument(7,sizeof(cl_int)*(64*5*(4+3*(height-1)))); //stack for every thread
    gput->SetIntArgument(8,roffsetX);
    gput->SetIntArgument(9,xWidth);
    gput->SetIntArgument(10,yWidth);
    gput->SetIntArgument(11,a);
    gput->SetIntArgument(12,b);
    gput->SetIntArgument(13,trianglePartCount); //number of uploaded triangles to GPU
    gput->SetIntArgument(15,5*gws); //stack size
    #ifdef TRIANGLES_PER_THREAD
      gput->SetIntArgument(16,trianlgesPerThread);
    #endif
    /*cl_image_format imageFormat;
    imageFormat.image_channel_data_type = CL_FLOAT;
    imageFormat.image_channel_order = CL_RGBA;
    gput->CreateImage2D(7, CL_MEM_WRITE_ONLY, &imageFormat, xResolution*samplesPerPixel, yResolution, 0, 16);*/

    #if (defined STAT_RAY_TRIANGLE && !defined TRIANGLES_PER_THREAD)
    gput->CreateBuffer(8,sizeof(cl_int)*count, CL_MEM_WRITE_ONLY,16);
    #endif
    #if (defined TRIANGLES_PER_THREAD && defined TRIANGLES_PER_THREAD)
    gput->CreateBuffer(8, sizeof(cl_int)*count, CL_MEM_WRITE_ONLY, 17);
    #endif

    //if (!gput->EnqueueWriteBuffer( 4, rayBoundsArray ))exit(EXIT_FAILURE);
    gput->EnqueueWrite2DImage(4, rayBoundsArray);
    gput->EnqueueWriteBuffer( 5, tHitArray );
    gput->EnqueueWriteBuffer( 6, indexArray );
    #ifdef STAT_RAY_TRIANGLE
    gput->EnqueueWriteBuffer( 8, picture);
    #endif
    for ( int i = 0; i < parts - 1; i++){
      gput->SetIntArgument(14,(cl_uint)i*trianglePartCount); //offset
      gput->EnqueueWriteBuffer( 0, vertices + 9*i*trianglePartCount);
      gput->Run();
      gput->WaitForKernel();
    }
    //last part of vertices
    gput->SetIntArgument(13,(cl_int)triangleLastPartCount);
    gput->SetIntArgument(14,(cl_uint)(parts-1)*trianglePartCount);
    gput->EnqueueWriteBuffer( 0, vertices + 9*(parts-1)*trianglePartCount, sizeof(cl_float)*3*3*triangleLastPartCount);
    gput->Run();
    gput->WaitForKernel();
    //if (!gput->EnqueueReadBuffer( 6, indexArray)) exit(EXIT_FAILURE);
  //  if (!gput->EnqueueReadBuffer( 5, tHitArray)) exit(EXIT_FAILURE);
   // gput->WaitForRead();
   /* cl_float* kontrola = new cl_float[4*xResolution*samplesPerPixel*yResolution];
    gput->EnqueueRead2DImage(7, kontrola);
    gput->WaitForRead();
  for ( int i = 0; i < yResolution; i++){
    for ( int j = 0; j < xResolution*samplesPerPixel; j++){
      cout << kontrola[4*i*xResolution*samplesPerPixel + 4*j] << " "
           << kontrola[4*i*xResolution*samplesPerPixel + 4*j + 1] << " "
           << kontrola[4*i*xResolution*samplesPerPixel + 4*j + 2] << " "
           << kontrola[4*i*xResolution*samplesPerPixel + 4*j + 3] << " ";
    }
    cout << endl;
  }*/

    #ifdef STAT_RAY_TRIANGLE
    gput->EnqueueReadBuffer( 8, picture);
    uint i = 0;
    int temp = 0;
    gput->WaitForRead();
    for (i = 0; i < count; i++){
      temp = max(picture[i],temp);
      Ls[i] = RainbowColorMapping((float)(picture[i])/(float)scale);
    }
    cout << "Maximum intersection count: " << temp << endl;
    delete [] ((uint*)picture);
    workerSemaphore->Post();
    return;
    #endif

    gput->SetPersistentBuffers(0,8); //vertex,dir,o, nodes, ray bounds, tHit, indexArray

    //counter for changes in ray-triangle intersection
    cl_uint* changedArray = new cl_uint[triangleCount];
    memset(changedArray, 0, sizeof(cl_uint)*triangleCount);

    size_t tn4 = ocl->CreateTask(KERNEL_YETANOTHERINTERSECTION, 1, &gws, &lws, cmd);
    OpenCLTask* anotherIntersect = ocl->getTask(tn4, cmd);
    anotherIntersect->InitBuffers(9);
    anotherIntersect->CopyBuffers(0,8,0,gput);
    ocl->delTask(tn2,cmd);
    anotherIntersect->CreateBuffer(8,sizeof(cl_uint)*trianglePartCount, CL_MEM_WRITE_ONLY); //recording changes
    //anotherIntersect->SetLocalArgument(8,sizeof(cl_int)*(64*5*(4+3*(height-1)))); //stack for every thread
    anotherIntersect->SetIntArgument(9,roffsetX);
    anotherIntersect->SetIntArgument(10,xWidth);
    anotherIntersect->SetIntArgument(11,yWidth);
    anotherIntersect->SetIntArgument(12,a);
    anotherIntersect->SetIntArgument(13,b);
    anotherIntersect->SetIntArgument(14,trianglePartCount); //number of uploaded triangles to GPU
    anotherIntersect->SetIntArgument(16,5*gws); //stack size
    #ifdef TRIANGLES_PER_THREAD
      anotherIntersect->SetIntArgument(17,trianlgesPerThread);
    #endif
    for ( int i = 0; i < parts - 1; i++){
      anotherIntersect->SetIntArgument(15,(cl_int)i*trianglePartCount);
      anotherIntersect->EnqueueWriteBuffer( 0, vertices + 9*i*trianglePartCount);

      for ( int k = 0; k < repairRun; k++){
        memset(changedArray + i*trianglePartCount, 0, sizeof(cl_uint)*trianglePartCount);
        anotherIntersect->EnqueueWriteBuffer(8, changedArray + i*trianglePartCount);
        anotherIntersect->Run();
        anotherIntersect->WaitForKernel();

        /*anotherIntersect->EnqueueReadBuffer(8,changedArray + i*trianglePartCount);
        anotherIntersect->WaitForRead();
        cout << "run " << k << endl;
        for ( int j = 0; j < trianglePartCount; j++)
          if ( changedArray[i*trianglePartCount + j] != 0 )
            cout << "changed[" << j << "]=" << changedArray[i*trianglePartCount + j] << endl;  */
      }

    }
    anotherIntersect->SetIntArgument(14,(cl_int)triangleLastPartCount);
    anotherIntersect->SetIntArgument(15,(cl_int)(parts-1)*trianglePartCount);
    anotherIntersect->EnqueueWriteBuffer( 0, vertices + 9*(parts-1)*trianglePartCount, sizeof(cl_float)*3*3*triangleLastPartCount);
    for ( int k = 0; k < repairRun; k++){
      memset(changedArray + (parts-1)*trianglePartCount, 0, sizeof(cl_uint)*triangleLastPartCount);
      anotherIntersect->EnqueueWriteBuffer(8, changedArray + (parts-1)*trianglePartCount);
      anotherIntersect->Run();
      anotherIntersect->WaitForKernel();
    }

    //anotherIntersect->EnqueueReadBuffer( 8, changedArray );
    anotherIntersect->EnqueueReadBuffer( 5, tHitArray);
    anotherIntersect->EnqueueReadBuffer( 6, indexArray);
    anotherIntersect->WaitForRead();
    anotherIntersect->SetPersistentBuff(0);//vertex
    anotherIntersect->SetPersistentBuff(1);//dir
    anotherIntersect->SetPersistentBuff(2);//origin
    anotherIntersect->SetPersistentBuff(6);//index

    gws = count;
    size_t tn3 = ocl->CreateTask(KERNEL_COMPUTEDPTUTV, 1, &gws, &lws, cmd);
    OpenCLTask* gpuRayO = ocl->getTask(tn3,cmd);
    gpuRayO->InitBuffers(7);
    gpuRayO->CopyBuffers(0,3,0,anotherIntersect); // 0 vertex, 1 dir, 2 origin
    gpuRayO->CopyBuffer(6,3,anotherIntersect); // 3 index
    ocl->delTask(tn4,cmd);
    gpuRayO->CreateBuffer(4,sizeof(cl_float)*6*trianglePartCount, CL_MEM_READ_ONLY ); //uvs
    gpuRayO->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY ); // tu,tv
    gpuRayO->CreateBuffer(6,sizeof(cl_float)*6*count, CL_MEM_WRITE_ONLY ); //dpdu, dpdv
    gpuRayO->SetIntArgument(10,xResolution*samplesPerPixel);

    gpuRayO->SetIntArgument(7,(cl_uint)count);

    for ( int i = 0; i < parts - 1; i++){
      gpuRayO->EnqueueWriteBuffer( 4 , uvs + 6*i*trianglePartCount);
      gpuRayO->SetIntArgument(8,i*trianglePartCount);
      gpuRayO->SetIntArgument(9,(i+1)*trianglePartCount);
      gpuRayO->EnqueueWriteBuffer(0, vertices + 9*i*trianglePartCount);
      gpuRayO->Run();
      gpuRayO->WaitForKernel();
    }
    gpuRayO->SetIntArgument(8,(cl_int)(parts-1)*trianglePartCount);
    gpuRayO->SetIntArgument(9,(cl_int)triangleCount);
    gpuRayO->EnqueueWriteBuffer(4, uvs + 6*(parts-1)*trianglePartCount, 6*sizeof(cl_float)*triangleLastPartCount);
    gpuRayO->EnqueueWriteBuffer(0, vertices + 9*(parts-1)*trianglePartCount,9*sizeof(cl_float)*triangleLastPartCount);
    gpuRayO->Run();

    Info("count %d", count);
    cl_float* tutvArray = new cl_float[2*count]; // for tu, tv
    cl_float* dpduArray = new cl_float[2*3*count]; // for dpdu, dpdv
    gpuRayO->EnqueueReadBuffer( 5, tutvArray );
    gpuRayO->EnqueueReadBuffer( 6, dpduArray );

    gpuRayO->WaitForRead();
    ocl->delTask(tn3,cmd);
    workerSemaphore->Post();
    cl_uint index;

    Vector dpdu, dpdv;
    //deserialize rectangles
    for ( unsigned int i = 0; i < count; i++) {
        index = indexArray[i];
        hit[i] = false;
        if ( index == -1 ) continue;
        Assert( index < triangleCount);
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());
		if ( shape == NULL) continue;

        dpdu = Vector(dpduArray[6*i],dpduArray[6*i+1],dpduArray[6*i+2]);
        dpdv = Vector(dpduArray[6*i+3],dpduArray[6*i+4],dpduArray[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*i], tutvArray[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*i],tutvArray[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * tHitArray[i]; //thit
        r[i].maxt = tHitArray[i];
        hit[i] = true;
        PBRT_RAY_TRIANGLE_INTERSECTION_HIT(&r[i], r[i].maxt);
    }

    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] tHitArray;
    delete [] tutvArray;
    delete [] dpduArray;
    delete [] indexArray;
    delete [] changedArray;
}

bool RayHieararchy::Intersect(const Ray &ray, Intersection *isect) const {
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->Intersect(ray,isect)) {
            hit = true;
        }
    }
    return hit;
}


bool RayHieararchy::IntersectP(const Ray &ray) const {
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->IntersectP(ray)) {
            hit = true;
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&ray,0);
        }
        PBRT_FINISHED_RAY_INTERSECTIONP(&ray, int(hit));
    }
    return hit;
}


void RayHieararchy::IntersectP(const Ray* r, char* occluded, const size_t count, const bool* hit
    #ifdef STAT_PRAY_TRIANGLE
    , Spectrum *Ls
    #endif
) {
  cl_float* rayDirArray = new cl_float[count*4]; //ray directions
  cl_float* rayOArray = new cl_float[count*4]; //ray origins
  cl_float* rayBoundsArray = new cl_float[count*2]; //ray bounds
  #ifdef STAT_PRAY_TRIANGLE
  cl_uint* picture = new cl_uint[count];
  unsigned int elem_counter = 0;
  #endif

  for ( unsigned int k = 0; k < count; ++k){
    rayDirArray[4*k] = r[k].d[0];
    rayDirArray[4*k+1] = r[k].d[1];
    rayDirArray[4*k+2] = r[k].d[2];


    rayOArray[4*k] = r[k].o[0];
    rayOArray[4*k+1] = r[k].o[1];
    rayOArray[4*k+2] = r[k].o[2];
    rayOArray[4*k+3] = 0;

    rayBoundsArray[2*k] = r[k].mint;
    rayBoundsArray[2*k+1] = r[k].maxt;
    occluded[k] = '0';

    if ( hit[k]){
      rayDirArray[4*k+3] = 1;
      #ifdef STAT_PRAY_TRIANGLE
      ++elem_counter;
      #endif
    } else {
      rayDirArray[4*k+3] = -1;
    }
  }

  #ifdef STAT_PRAY_TRIANGLE
  cout << "# prays: " << elem_counter << " all rays: " << count << endl;
  #endif

  workerSemaphore->Wait();
  #ifdef GPU_TIMES
  for ( unsigned int k = 0; k < count; k++){
    if ( hit[k] ) ++shadowRays;
  }
  #endif
  int roffsetX, xWidth, yWidth;
  size_t tn1 = ConstructRayHierarchyP(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
  OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

  size_t gws = trianglePartCount; //2880;
  size_t lws = 64;
  size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTIONP, 1, &gws, &lws, cmd);
  OpenCLTask* gput = ocl->getTask(tn2,cmd);

  unsigned int c = 8;
  #ifdef STAT_PRAY_TRIANGLE
   c = 9;
  #endif
  gput->InitBuffers(c);

  gput->CopyBuffer(0,1,gpuray); //ray dir
  gput->CopyBuffer(1,2,gpuray); //ray o
  gput->CopyBuffer(2,3,gpuray); //nodes
  gput->CopyBuffer(3,4,gpuray); //ray validity
  ocl->delTask(tn1,cmd);

  cl_image_format imageFormatBounds;
  imageFormatBounds.image_channel_data_type = CL_FLOAT;
  imageFormatBounds.image_channel_order = CL_RG;

  gput->CreateBuffer(0,sizeof(cl_float)*3*3*trianglePartCount, CL_MEM_READ_ONLY ); //vertices
  //gput->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_READ_ONLY); //ray bounds
  gput->CreateImage2D(5, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0);
  gput->CreateBuffer(6,sizeof(cl_char)*count, CL_MEM_WRITE_ONLY); //tHit
  #ifdef STAT_PRAY_TRIANGLE
   gput->CreateBuffer(8,sizeof(cl_uint)*count, CL_MEM_WRITE_ONLY, 15);
  #endif
  gput->CreateBuffer(7,sizeof(cl_int)*(gws+lws)*5*(xWidth*yWidth+5*height), CL_MEM_READ_WRITE);
  //Assert(gput->SetLocalArgument(7,sizeof(cl_int)*(64*5*(4 + 3*(height-1)))));
  gput->SetIntArgument(8, roffsetX);
  gput->SetIntArgument(9, xWidth);
  gput->SetIntArgument(10, yWidth);
  gput->SetIntArgument(11,a);
  gput->SetIntArgument(12,b);
  gput->SetIntArgument(13, trianglePartCount);
  gput->SetIntArgument(14, 5*gws); //stack level size
  //int globalPoolNextRay = 2*gws;
  //gput->CreateBuffer(8,sizeof(cl_int),CL_MEM_READ_WRITE,15);
  #ifdef TRIANGLES_PER_THREAD
    gput->SetIntArgument(15,trianlgesPerThread);
  #endif

  //if (!gput->EnqueueWriteBuffer( 5, rayBoundsArray ))exit(EXIT_FAILURE);
  gput->EnqueueWrite2DImage(5, rayBoundsArray);
  gput->EnqueueWriteBuffer( 6, occluded );
  #ifdef STAT_PRAY_TRIANGLE
    gput->EnqueueWriteBuffer( 8, picture);
  #endif
  for ( int i = 0; i < parts - 1; i++){
    gput->EnqueueWriteBuffer(0, vertices + 9*i*trianglePartCount);
    //gput->EnqueueWriteBuffer(8,&globalPoolNextRay);
    gput->Run();
    gput->WaitForKernel();
  }
  //gput->EnqueueWriteBuffer(8,&globalPoolNextRay);
  gput->EnqueueWriteBuffer(0, vertices + 9*(parts-1)*trianglePartCount, sizeof(cl_float)*9*triangleLastPartCount);
  gput->SetIntArgument(13,(cl_int)triangleLastPartCount);
  gput->Run();

  gput->EnqueueReadBuffer( 6, occluded );
  #ifdef STAT_PRAY_TRIANGLE
    gput->EnqueueReadBuffer( 8, picture);
  #endif

  gput->WaitForRead();
  ocl->delTask(tn2,cmd);
  workerSemaphore->Post();

    #ifdef STAT_PRAY_TRIANGLE
    uint i = 0;
    uint temp,t;
    t = 0;
    float colors[3];
    for (i = 0; i < count; i++){
      temp = ((cl_uint*)picture)[i];
      t = max(temp,t);
      Ls[i] = RainbowColorMapping((float)(temp)/scale);
     // cout << ' ' << temp  ;
     // Ls[i] = Ls[i].Clamp(0,255);
    }
    cout << "Maximum intersectionP count: " << t << endl;
    delete [] ((uint*)picture);
    #endif

    #ifdef PBRT_STATS_COUNTERS
    for ( unsigned i = 0; i < count; i++){
        PBRT_FINISHED_RAY_INTERSECTIONP(&r[i], int(occluded[i]));
        if (occluded[i] != '0')
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&r[i],0);
    }
    #endif


    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;

}


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    bool sortVert = ps.FindOneBool("sort",true);
    int chunkX = ps.FindOneInt("chunkXSize", 4);
    int chunkY = ps.FindOneInt("chunkYSize", 4);
    int height = ps.FindOneInt("height",3);
    string node = ps.FindOneString("node", "box_dir");
    string splitMethod = ps.FindOneString("splitmethod", "sah");
    int maxBVHPrim = ps.FindOneInt("maxBVHPrim",1);
    unsigned int repairRun = ps.FindOneInt("repairRun",1);
    #ifdef TRIANGLES_PER_THREAD
    int trianlgesPerThread = ps.FindOneInt("trianlgesPerThread", 1);
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    int scale = ps.FindOneInt("scale",50);
    #endif
    return new RayHieararchy(prims,onGPU,chunkX, chunkY,height,node,sortVert, splitMethod, maxBVHPrim,repairRun
    #ifdef TRIANGLES_PER_THREAD
    , trianlgesPerThread
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , scale
    #endif
    );
}


