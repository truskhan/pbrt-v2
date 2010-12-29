#include "accelerators/rayBVH.h"
#include "accelerators/bvh.h"
#include "core/probes.h"
#include "core/camera.h"
#include "core/film.h"
#include "core/renderer.h"
#include "core/paramset.h"
#include "core/intersection.h"
#include "core/GPUparallel.h"
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


// RayBVH Method Definitions
RayBVH::RayBVH(const vector<Reference<Primitive> > &prim, bool onG, int chunkX, int chunkY,
 int height, int BVHheight, string node, int maxBVHPrim,  const string &sm
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , int scale
    #endif
  ) {
    this->sm = sm;
    this->chunkX = chunkX;
    this->chunkY = chunkY;
    this->height = height;
    this->BVHheight = BVHheight;
    this->maxBVHPrim = maxBVHPrim;
    splitMethod = sm;
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    this->scale = scale;
    #endif
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
  headerName[headerNum++] = "cl/GPUNode.cl";
  if ( node == "ia"){
    names[0] = "cl/intersection6DB_BVH.cl";
    names[1] = "cl/intersectionP6DB_BVH.cl";
    names[2] = "cl/rayhconstruct6DB.cl";
    names[3] = "cl/levelConstruct6DB.cl";
    names[4] = "cl/yetAnotherIntersection6DB_BVH.cl";
    names[6] = "cl/levelConstructP6DB.cl";
    names[7] = "cl/rayhconstructP6DB.cl";
    names[8] = "cl/intersection6DB_BVH2.cl";
    headerName[headerNum++] = "cl/intersectsNodeIA.cl";
    cout << "accel nodes : IA" << endl;
    nodeSize = 13;
  }
  if ( node == "box_uv") {
    names[0] = "cl/intersection5DB_BVH.cl";
    names[1] = "cl/intersectionP5DB_BVH.cl";
    names[2] = "cl/rayhconstruct5DB.cl";
    names[3] = "cl/levelConstruct5DB.cl";
    names[4] = "cl/yetAnotherIntersection5DB_BVH.cl";
    names[6] = "cl/levelConstructP5DB.cl";
    names[7] = "cl/rayhconstructP5DB.cl";
    names[8] = "cl/intersection5DB_BVH2.cl";
    headerName[headerNum++] = "cl/intersectsNode5DB.cl";
    cout << "accel nodes : 5D nodes with boxes" << endl;
    nodeSize = 11;
  }
  if ( node == "box_dir") {
    names[0] = "cl/intersection6DB_BVH.cl";
    names[1] = "cl/intersectionP6DB_BVH.cl";
    names[2] = "cl/rayhconstruct6DB.cl";
    names[3] = "cl/levelConstruct6DB.cl";
    names[4] = "cl/yetAnotherIntersection6DB_BVH.cl";
    names[6] = "cl/levelConstructP6DB.cl";
    names[7] = "cl/rayhconstructP6DB.cl";
    names[8] = "cl/intersection6DB_BVH2.cl";
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

    BVHAccel* bvhTemp = new BVHAccel(prim, maxBVHPrim, sm, true, BVHheight);
    this->p.swap(bvhTemp->primitives);
    delete bvhTemp;

    triangleCount = this->p.size();
    cout << "Scene contains " << triangleCount << " triangles"<< endl;

    //TODO: check how many threads can be proccessed at once (depends on MaxRaysPerCall)
    workerSemaphore = new Semaphore(1);

   #ifdef GPU_TIMES
    for ( int i = 0; i < 10; i++){
      intersectTimes[i] = 0;
      bounceRays[i] = 0;
    }
   #endif
}

RayBVH::~RayBVH() {
  #ifdef GPU_TIMES
  ocl->PrintTimes();
  for ( int i = 0; i < 10; i++)
    cout << "intersection " << i << " .bounce " << intersectTimes[i] << endl;
  for ( int i = 0; i < 10; i++)
    cout << "rays at " << i << " .bounce " << bounceRays[i] << endl;
  #endif
  delete ocl;
  for ( size_t i = 0 ; i < parts; i++){
    delete [] vertices[i];
    delete [] uvs[i];
    delete bvhs[i];
  }
  delete [] vertices;
  delete [] uvs;
  delete [] bvhs;
  delete workerSemaphore;
}

BBox RayBVH::WorldBound() const {
    return bbox;
}

void RayBVH::Preprocess(const Camera* camera, const unsigned samplesPerPixel,
  const int nx, const int ny){
  this->xResolution = camera->film->xResolution;
  this->yResolution = camera->film->yResolution;
  this->samplesPerPixel = samplesPerPixel;
  xResolution /= nx;
  yResolution /= ny;
}

void RayBVH::Preprocess(){
    bvhs = new BVHAccel*[parts];
    vector<Reference<Primitive> >* vec;
    size_t k, low, up;
    bvhTopLevelNodesMax = bvhNodesMax = maxPrims = 0;
    for ( size_t j = 0; j < parts - 1; j++){
      low = j*trianglePartCount;
      up = (j+1)*trianglePartCount;
      vec = new vector<Reference<Primitive> >();
      for ( k = low; k < up ; k++)
        vec->push_back(p[k]);
      bvhs[j] = new BVHAccel(*vec, maxBVHPrim, sm, true, BVHheight);
      bvhTopLevelNodesMax = max(bvhTopLevelNodesMax, bvhs[j]->topLevelNodes);
      bvhNodesMax = max(bvhNodesMax, bvhs[j]->nodeNum);
      maxPrims = max(maxPrims, bvhs[j]->primitives.size());
      delete vec;
    }
    k = 0;
    vec = new vector<Reference<Primitive> >();
    for ( k = (parts-1)*trianglePartCount; k < p.size() ; k++)
      vec->push_back(p[k]);
    p.clear();
    bvhs[parts - 1] = new BVHAccel(*vec, maxBVHPrim, sm, true, BVHheight);
    bvhTopLevelNodesMax = max(bvhTopLevelNodesMax, bvhs[parts-1]->topLevelNodes);
    bvhNodesMax = max(bvhNodesMax, bvhs[parts - 1]->nodeNum);
    maxPrims = max(maxPrims, bvhs[parts-1]->primitives.size());
    delete vec;

    bbox = bvhs[0]->WorldBound();

    //store vertices and uvs in linear order
    vertices = new cl_float*[parts];
    uvs = new cl_float*[parts];
    for ( size_t j = 0; j < parts; j++){
      bbox = Union(bbox, bvhs[j]->WorldBound());
      vertices[j] = new cl_float[9*bvhs[j]->primitives.size()];
      uvs[j] = new cl_float[6*bvhs[j]->primitives.size()];
      for (uint32_t i = 0; i < bvhs[j]->primitives.size(); ++i) {
          const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (bvhs[j]->primitives[i].GetPtr()));
          if ( gp == 0 ) continue;
          const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
          if ( shape == 0) continue;
          const TriangleMesh* mesh = shape->GetMeshPtr();
          const Point &p1 = mesh->p[shape->v[0]];
          const Point &p2 = mesh->p[shape->v[1]];
          const Point &p3 = mesh->p[shape->v[2]];
           (vertices[j])[9*i+0] = p1.x;
           (vertices[j])[9*i+1] = p1.y;
           (vertices[j])[9*i+2] = p1.z;
           (vertices[j])[9*i+3] = p2.x;
           (vertices[j])[9*i+4] = p2.y;
           (vertices[j])[9*i+5] = p2.z;
           (vertices[j])[9*i+6] = p3.x;
           (vertices[j])[9*i+7] = p3.y;
           (vertices[j])[9*i+8] = p3.z;
          if (mesh->uvs) {
              (uvs[j])[6*i] = mesh->uvs[2*shape->v[0]];
              (uvs[j])[6*i+1] = mesh->uvs[2*shape->v[0]+1];
              (uvs[j])[6*i+2] = mesh->uvs[2*shape->v[1]];
              (uvs[j])[6*i+3] = mesh->uvs[2*shape->v[1]+1];
              (uvs[j])[6*i+4] = mesh->uvs[2*shape->v[2]];
              (uvs[j])[6*i+5] = mesh->uvs[2*shape->v[2]+1];
          } else { //todo - indicate this and compute at GPU
              (uvs[j])[6*i] = 0.f;
              (uvs[j])[6*i+1] = 0.f;
              (uvs[j])[6*i+2] = 1.f;
              (uvs[j])[6*i+3] = 0.f;
              (uvs[j])[6*i+4] = 1.f;
              (uvs[j])[6*i+5] = 1.f;
          }
      }
    }
}


unsigned int RayBVH::MaxRaysPerCall(){
    worgGroupSize = 64;

    //TODO: check the OpenCL device and decide, how many rays can be processed at one thread
    // check how many threads can be proccessed at once
    cl_ulong gms = ocl->getGlobalMemSize();
    //ray dir,o, bounds, tHit
    unsigned int x;

    //vertices
    #define MAX_VERTICES 60000
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
             - sizeof(cl_uint)*((xResolution + chunkX - 1) / chunkX) *((yResolution + chunkY - 1) / chunkY) //counts
             ) / (
             sizeof(cl_uint) //index
             + 9*sizeof(cl_float) //ray dir, o, bounds, tHit
             );

    //local mem - stack:
   // cl_ulong lms = ocl->getLocalMemSize();
   // cl_ulong localSize = sizeof(cl_int)*(2 + (height+1)*(height+2)/2)*worgGroupSize;
    //if ( lms < localSize){
    //  Severe("Need local memory size at least %i Bytes, present %i B. Try to decrease hierarchy's height.", localSize, lms);
    //}


    cout << "Global memory size on OpenCL device (in B): " << gms << endl;
    cout << "Maximum memory allocation size at once: " << ocl->getMaxMemAllocSize() << endl;
    cout << "Local memory size on OpenCL device: " << ocl->getLocalMemSize() << endl;
    cout << "Constant memory size: " << ocl->getMaxConstantBufferSize() << endl;
    cout << "Max work group size: " << ocl->getMaxWorkGroupSize() << endl;

    cout << "Needed rays " << xResolution*yResolution*samplesPerPixel << " device maximu rays " << x << endl;
    Preprocess();
    return 409600;
}

//classical method for testing one ray
bool RayBVH::Intersect(const Triangle* shape, const Ray &ray, float *tHit,
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
size_t RayBVH::ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, int *roffsetX, int *xWidth, int *yWidth ){
  Assert(height > 0);
  size_t globalSize[2] = {(xResolution*samplesPerPixel + chunkX -1)/chunkX, (yResolution+chunkY-1)/chunkY};
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
  gpuray->SetIntArgument(5, chunkX);
  gpuray->SetIntArgument(6, chunkY);

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
size_t RayBVH::ConstructRayHierarchyP(cl_float* rayDir, cl_float* rayO,
  int *roffsetX, int *xWidth, int *yWidth ){
  Assert(height > 0);
  size_t globalSize[2] = {(xResolution*samplesPerPixel + chunkX -1)/chunkX, (yResolution+chunkY-1)/chunkY};
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
  gpuray->SetIntArgument(6, chunkX);
  gpuray->SetIntArgument(7, chunkY);

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

void RayBVH::Intersect(const RayDifferential *r, Intersection *in, bool* hit,
    const unsigned int count, const int bounce){
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
	#ifdef GPU_TIMES
    else
      ++bounceRays[bounce+1];
	#endif
  }

  workerSemaphore->Wait();
  int roffsetX, xWidth, yWidth;
  size_t tn1 = ConstructRayHierarchyP(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
  OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

  size_t gws = bvhTopLevelNodesMax;
  size_t lws = 64;
  size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTION2, 1, &gws, &lws, cmd);
  OpenCLTask* gput = ocl->getTask(tn2,cmd);

  unsigned int c = 10;
  gput->InitBuffers(c);

  gput->CopyBuffer(0,1,gpuray); //ray dir
  gput->CopyBuffer(1,2,gpuray); //ray o
  gput->CopyBuffer(2,3,gpuray); //nodes
  gput->CopyBuffer(3,4,gpuray); //ray validity
  ocl->delTask(tn1,cmd);

  cl_image_format imageFormatBounds;
  imageFormatBounds.image_channel_data_type = CL_FLOAT;
  imageFormatBounds.image_channel_order = CL_RG;

  int tempHeight = max(height, BVHheight);
  gput->CreateBuffer(0,sizeof(cl_float)*9*maxPrims, CL_MEM_READ_ONLY ); //vertices
  gput->CreateImage2D(5, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0); //ray bounds
  gput->CreateBuffer(6,sizeof(cl_float)*count, CL_MEM_WRITE_ONLY); //tHit
  gput->CreateBuffer(7,sizeof(cl_int)*count, CL_MEM_WRITE_ONLY); //index array
  gput->CreateBuffer(8,sizeof(cl_int)*(gws+lws)*6*(xWidth*yWidth+8*tempHeight), CL_MEM_READ_WRITE); //stack
  gput->CreateBuffer(9,sizeof(GPUNode)*bvhNodesMax, CL_MEM_READ_ONLY); //bvh nodes
  gput->SetIntArgument(10,roffsetX);
  gput->SetIntArgument(11,xWidth);
  gput->SetIntArgument(12,yWidth);
  gput->SetIntArgument(13,chunkX);
  gput->SetIntArgument(14,chunkY);

  gput->EnqueueWrite2DImage(5, rayBoundsArray);
  gput->EnqueueWriteBuffer( 6, tHitArray );
  gput->EnqueueWriteBuffer( 7, indexArray );
  cl_int offsetGID = 0;
  #ifdef GPU_TIMES
  double itime;
  #endif
  for ( int i = 0; i < parts - 1; i++){
    gput->SetIntArgument(15, bvhs[i]->topLevelNodes);
    gput->SetIntArgument(16, offsetGID); //index offset
    gput->EnqueueWriteBuffer(9, bvhs[i]->gpuNodes, sizeof(GPUNode)*bvhs[i]->nodeNum);
    gput->EnqueueWriteBuffer( 0, vertices[i], 9*sizeof(cl_float)*bvhs[i]->primitives.size());
    #ifdef GPU_TIMES
    itime =
    #endif
    gput->Run();
    #ifdef GPU_TIMES
    intersectTimes[bounce+1] += itime;
    #endif
    offsetGID += bvhs[i]->primitives.size();
    gput->WaitForKernel();
  }
  //last part of vertices
  gput->SetIntArgument(15, bvhs[parts-1]->topLevelNodes);
  gput->SetIntArgument(16, offsetGID);
  gput->EnqueueWriteBuffer(9, bvhs[parts-1]->gpuNodes, sizeof(GPUNode)*bvhs[parts-1]->nodeNum);
  gput->EnqueueWriteBuffer(0, vertices[parts-1], sizeof(cl_float)*9*bvhs[parts-1]->primitives.size());
  #ifdef GPU_TIMES
  itime =
  #endif
  gput->Run();
  #ifdef GPU_TIMES
  intersectTimes[bounce+1] += itime;
  #endif

  gput->EnqueueReadBuffer( 6, tHitArray );
  gput->EnqueueReadBuffer( 7, indexArray);

  gput->SetPersistentBuffers(0,3); //vertices, ray dir, ray orig
  gput->SetPersistentBuff(7); //indexArray

  gput->WaitForRead();
  gws = count;
  size_t tn3 = ocl->CreateTask(KERNEL_COMPUTEDPTUTV, 1, &gws, &lws, cmd);
  OpenCLTask* gpuRayO = ocl->getTask(tn3,cmd);
  gpuRayO->InitBuffers(7);
  gpuRayO->CopyBuffers(0,3,0,gput); // 0 vertex, 1 dir, 2 origin
  gpuRayO->CopyBuffer(7,3,gput); // 3 index
  ocl->delTask(tn2,cmd);
  gpuRayO->CreateBuffer(4,sizeof(cl_float)*6*maxPrims, CL_MEM_READ_ONLY ); //uvs
  gpuRayO->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY ); // tu,tv
  gpuRayO->CreateBuffer(6,sizeof(cl_float)*6*count, CL_MEM_WRITE_ONLY); //dpdu, dpdv
  gpuRayO->SetIntArgument(10,xResolution*samplesPerPixel);

  gpuRayO->SetIntArgument(7,(cl_uint)count);
  cl_int low = 0;
  cl_int up = bvhs[0]->primitives.size();
  for ( int i = 0; i < parts - 1; i++){
    gpuRayO->SetIntArgument(8,low);
    gpuRayO->SetIntArgument(9,up);
    gpuRayO->EnqueueWriteBuffer( 4 , uvs[i], sizeof(cl_float)*6*bvhs[i]->primitives.size());
    gpuRayO->EnqueueWriteBuffer(0, vertices[i], sizeof(cl_float)*9*bvhs[i]->primitives.size());
    gpuRayO->Run();
    gpuRayO->WaitForKernel();
    low = up;
    up += bvhs[i+1]->primitives.size();
  }
  gpuRayO->SetIntArgument(8,low);
  gpuRayO->SetIntArgument(9,up);
  gpuRayO->EnqueueWriteBuffer(4, uvs[parts-1], 6*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
  gpuRayO->EnqueueWriteBuffer(0, vertices[parts-1],9*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
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
  int help;
  //deserialize rectangles
  for ( unsigned int i = 0; i < count; i++) {
      if ( !hit[i]) continue;
      index = indexArray[i];
      hit[i] = false;
      if ( index == -1 ) continue;
      Assert( index < triangleCount);
      help = 0;
      while ( index >= bvhs[help]->primitives.size()){
        index -= bvhs[help]->primitives.size();
        ++help;
      }
      const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (bvhs[help]->primitives[index].GetPtr()));

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
void RayBVH::Intersect(const RayDifferential *r, Intersection *in,
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
  cl_uint* picture = new cl_uint[count];
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
    double itime;
    bounceRays[0] += count;
    #endif
    int roffsetX, xWidth, yWidth;
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    size_t gws = bvhTopLevelNodesMax;
    size_t lws = 64;
    size_t tn2 = ocl->CreateTask(KERNEL_INTERSECTIONR, 1, &gws, &lws, cmd);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);

    c = 9;
    #ifdef STAT_RAY_TRIANGLE
    c = 10;
    #endif
    gput->InitBuffers(c);

    gput->CopyBuffer(0,1,gpuray); //ray dir
    gput->CopyBuffer(1,2,gpuray); //ray o
    gput->CopyBuffer(2,3,gpuray); //nodes
    ocl->delTask(tn1,cmd);

    cl_image_format imageFormatBounds;
    imageFormatBounds.image_channel_data_type = CL_FLOAT;
    imageFormatBounds.image_channel_order = CL_RG;

    gput->CreateBuffer(0, 9*sizeof(cl_float)*maxPrims, CL_MEM_READ_ONLY ); //vertices
    gput->CreateImage2D(4, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0);
    gput->CreateBuffer(5,sizeof(cl_float)*count, CL_MEM_READ_WRITE); // tHit
    gput->CreateBuffer(6,sizeof(cl_int)*count, CL_MEM_WRITE_ONLY); //index array
    cout << count << endl;
    //allocate stack in global memory
    int tempHeight = max(height, BVHheight);
    gput->CreateBuffer(7,sizeof(cl_int)*(gws+lws)*6*(xWidth*yWidth+8*tempHeight), CL_MEM_READ_WRITE);
    gput->CreateBuffer(8,sizeof(GPUNode)*bvhNodesMax, CL_MEM_READ_ONLY); //bvh nodes
    gput->SetIntArgument(9,roffsetX);
    gput->SetIntArgument(10,xWidth);
    gput->SetIntArgument(11,yWidth);
    gput->SetIntArgument(12,chunkX);
    gput->SetIntArgument(13,chunkY);

    #ifdef STAT_RAY_TRIANGLE
    gput->CreateBuffer(9,sizeof(cl_uint)*count, CL_MEM_WRITE_ONLY,18);
    #endif

    gput->EnqueueWrite2DImage(4, rayBoundsArray);
    gput->EnqueueWriteBuffer( 5, tHitArray );
    gput->EnqueueWriteBuffer( 6, indexArray );
    #ifdef STAT_RAY_TRIANGLE
    gput->EnqueueWriteBuffer( 9, picture);
    #endif
    cl_int offsetGID = 0;
    for ( int i = 0; i < parts - 1; i++){
      gput->SetIntArgument(14,bvhs[i]->topLevelNodes);
      gput->SetIntArgument(15, offsetGID);
      gput->EnqueueWriteBuffer(8, bvhs[i]->gpuNodes, sizeof(GPUNode)*bvhs[i]->nodeNum);
      gput->EnqueueWriteBuffer(0, vertices[i], 9*sizeof(cl_float)*bvhs[i]->primitives.size());
      #ifdef GPU_TIMES
      itime =
      #endif
      gput->Run();
      #ifdef GPU_TIMES
      intersectTimes[0] += itime;
      #endif
      offsetGID += bvhs[i]->primitives.size();
      gput->WaitForKernel();
    }
    //last part of vertices
    gput->SetIntArgument(14,bvhs[parts-1]->topLevelNodes);
    gput->SetIntArgument(15,offsetGID);
    gput->EnqueueWriteBuffer(8, bvhs[parts-1]->gpuNodes, sizeof(GPUNode)*bvhs[parts-1]->nodeNum);
    gput->EnqueueWriteBuffer(0, vertices[parts-1], 9*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
    #ifdef GPU_TIMES
    itime =
    #endif
    gput->Run();
    #ifdef GPU_TIMES
    intersectTimes[0] += itime;
    #endif
    gput->WaitForKernel();
    /*int* stackTemp = new int [(gws+lws)*6*(xWidth*yWidth+8*tempHeight)];
    gput->EnqueueReadBuffer(7,stackTemp);
    gput->WaitForRead();
    for ( int i = 0; i < (gws+lws)*6*(xWidth*yWidth+8*tempHeight);i++){
      for ( int j = 0; j < 6; j++){
      cout << stackTemp[i+j] << ' ';
      }
      cout << endl;
      i += 5;
    }
    abort();*/

    #ifdef STAT_RAY_TRIANGLE
    gput->EnqueueReadBuffer( 9, picture);
    unsigned int i = 0;
    unsigned int temp = 0;
    gput->WaitForRead();
    for (i = 0; i < count; i++){
      temp = max(picture[i],temp);
      Ls[i] = RainbowColorMapping((float)(picture[i])/(float)scale);
    }
    delete [] picture;
    workerSemaphore->Post();
    return;
    #endif

    gput->SetPersistentBuffers(0,9); //vertex,dir,o, nodes, ray bounds, tHit, indexArray, stack, bvh nodes

    //counter for changes in ray-triangle intersection
    cl_uint* changedArray = new cl_uint[triangleCount];
    memset(changedArray, 0, sizeof(cl_uint)*triangleCount);

    size_t tn4 = ocl->CreateTask(KERNEL_YETANOTHERINTERSECTION, 1, &gws, &lws, cmd);
    OpenCLTask* anotherIntersect = ocl->getTask(tn4, cmd);
    anotherIntersect->InitBuffers(10);
    anotherIntersect->CopyBuffers(0,9,0,gput);
    ocl->delTask(tn2,cmd);
    anotherIntersect->CreateBuffer(9,sizeof(cl_uint)*trianglePartCount, CL_MEM_WRITE_ONLY); //recording changes
    anotherIntersect->SetIntArgument(10,roffsetX);
    anotherIntersect->SetIntArgument(11,xWidth);
    anotherIntersect->SetIntArgument(12,yWidth);
    anotherIntersect->SetIntArgument(13,chunkX);
    anotherIntersect->SetIntArgument(14,chunkY);
    anotherIntersect->SetIntArgument(15,6*gws); //stack level size

    offsetGID = 0;
    for ( int i = 0; i < parts - 1; i++){
      anotherIntersect->SetIntArgument(16,bvhs[i]->topLevelNodes);
      anotherIntersect->SetIntArgument(17, offsetGID);
      anotherIntersect->EnqueueWriteBuffer(8, bvhs[i]->gpuNodes, sizeof(GPUNode)*bvhs[i]->nodeNum);
      anotherIntersect->EnqueueWriteBuffer(0, vertices[i], 9*sizeof(cl_float)*bvhs[i]->primitives.size());
      anotherIntersect->Run();
      offsetGID += bvhs[i]->primitives.size();
      anotherIntersect->WaitForKernel();
    }
    //last part of vertices
    anotherIntersect->SetIntArgument(16,bvhs[parts-1]->topLevelNodes);
    anotherIntersect->SetIntArgument(17,offsetGID);
    anotherIntersect->EnqueueWriteBuffer( 8, bvhs[parts-1]->gpuNodes, sizeof(GPUNode)*bvhs[parts-1]->nodeNum);
    anotherIntersect->EnqueueWriteBuffer( 0, vertices[parts-1], 9*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
    anotherIntersect->Run();

    anotherIntersect->EnqueueReadBuffer( 9, changedArray );
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
    gpuRayO->CreateBuffer(4,sizeof(cl_float)*6*maxPrims, CL_MEM_READ_ONLY ); //uvs
    gpuRayO->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY ); // tu,tv
    gpuRayO->CreateBuffer(6,sizeof(cl_float)*6*count, CL_MEM_WRITE_ONLY ); //dpdu, dpdv
    gpuRayO->SetIntArgument(10,xResolution*samplesPerPixel);

    gpuRayO->SetIntArgument(7,(cl_uint)count);
    cl_int low = 0;
    cl_int up = bvhs[0]->primitives.size();
    for ( int i = 0; i < parts - 1; i++){
      gpuRayO->SetIntArgument(8,low);
      gpuRayO->SetIntArgument(9,up);
      gpuRayO->EnqueueWriteBuffer( 4 , uvs[i], sizeof(cl_float)*6*bvhs[i]->primitives.size());
      gpuRayO->EnqueueWriteBuffer(0, vertices[i], sizeof(cl_float)*9*bvhs[i]->primitives.size());
      gpuRayO->Run();
      gpuRayO->WaitForKernel();
      low = up;
      up += bvhs[i+1]->primitives.size();
    }
    gpuRayO->SetIntArgument(8,low);
    gpuRayO->SetIntArgument(9,up);
    gpuRayO->EnqueueWriteBuffer(4, uvs[parts-1], 6*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
    gpuRayO->EnqueueWriteBuffer(0, vertices[parts-1],9*sizeof(cl_float)*bvhs[parts-1]->primitives.size());
    gpuRayO->Run();

    Info("count %d", count);
    cl_float* tutvArray = new cl_float[2*count]; // for tu, tv
    cl_float* dpduArray = new cl_float[2*3*count]; // for dpdu, dpdv
    gpuRayO->EnqueueReadBuffer( 5, tutvArray );
    gpuRayO->EnqueueReadBuffer( 6, dpduArray );

    gpuRayO->WaitForRead();
    ocl->delTask(tn3,cmd);
    workerSemaphore->Post();
    cl_int index;

    Vector dpdu, dpdv;
    int help;
    //deserialize rectangles
    for ( unsigned int i = 0; i < count; i++) {
        index = indexArray[i];
        hit[i] = false;
        if ( index == -1 ) continue;
        Assert( index < triangleCount);
        help = 0;
        while ( index >= bvhs[help]->primitives.size()){
          index -= bvhs[help]->primitives.size();
          ++help;
        }
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (bvhs[help]->primitives[index].GetPtr()));
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
    delete [] changedArray;
}

bool RayBVH::Intersect(const Ray &ray, Intersection *isect) const {
  bool hit = false;
  for ( int i = 0; i < parts; i++){
    //if (bvh->primitives.size() == 0) return false;
    for (uint32_t it = 0; it < bvhs[i]->primitives.size(); ++it) {
        if (bvhs[i]->primitives[it]->Intersect(ray,isect)) {
            hit = true;
        }
    }
  }
  return hit;
}


bool RayBVH::IntersectP(const Ray &ray) const {
  bool hit = false;
  for ( int i = 0 ; i < parts; i++){
    //if (bvh->primitives.size() == 0) return false;
    for (uint32_t it = 0; it < bvhs[i]->primitives.size(); ++it) {
        if (bvhs[i]->primitives[it]->IntersectP(ray)) {
            hit = true;
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&ray,0);
        }
        PBRT_FINISHED_RAY_INTERSECTIONP(&ray, int(hit));
    }
  }
  return hit;
}


void RayBVH::IntersectP(const Ray* r, char* occluded, const size_t count, const bool* hit
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
  int roffsetX, xWidth, yWidth;
  size_t tn1 = ConstructRayHierarchyP(rayDirArray, rayOArray, &roffsetX, &xWidth, &yWidth);
  OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

  size_t gws = bvhTopLevelNodesMax;
  size_t lws = 64;
  size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTIONP, 1, &gws, &lws, cmd);
  OpenCLTask* gput = ocl->getTask(tn2,cmd);

  unsigned int c = 9;
  #ifdef STAT_PRAY_TRIANGLE
   c = 10;
  #endif
  gput->InitBuffers(c);

  gput->CopyBuffer(0,1,gpuray); //ray dir
  gput->CopyBuffer(1,2,gpuray); //ray o
  gput->CopyBuffer(2,3,gpuray); //ray-hierarchy nodes
  gput->CopyBuffer(3,4,gpuray); //ray validity
  ocl->delTask(tn1,cmd);

  cl_image_format imageFormatBounds;
  imageFormatBounds.image_channel_data_type = CL_FLOAT;
  imageFormatBounds.image_channel_order = CL_RG;

  gput->CreateBuffer(0,sizeof(cl_float)*3*3*maxPrims, CL_MEM_READ_ONLY ); //vertices
  gput->CreateImage2D(5, CL_MEM_READ_ONLY , &imageFormatBounds, xResolution*samplesPerPixel, yResolution, 0); //bounds
  gput->CreateBuffer(6,sizeof(cl_char)*count, CL_MEM_WRITE_ONLY); //tHit
  #ifdef STAT_PRAY_TRIANGLE
   gput->CreateBuffer(9,sizeof(cl_uint)*count, CL_MEM_WRITE_ONLY, 18);
  #endif
  int tempHeight = max(height, BVHheight);
  gput->CreateBuffer(7,sizeof(cl_int)*(gws+lws)*6*(xWidth*yWidth+4+8*tempHeight), CL_MEM_READ_WRITE); //stack
  gput->CreateBuffer(8,sizeof(GPUNode)*bvhNodesMax, CL_MEM_READ_ONLY); //BVH nodes
  gput->SetIntArgument(9, roffsetX);
  gput->SetIntArgument(10, xWidth);
  gput->SetIntArgument(11, yWidth);
  gput->SetIntArgument(12,chunkX);
  gput->SetIntArgument(13,chunkY);

  gput->EnqueueWrite2DImage(5, rayBoundsArray);
  gput->EnqueueWriteBuffer( 6, occluded );
  #ifdef STAT_PRAY_TRIANGLE
    gput->EnqueueWriteBuffer( 9, picture);
  #endif
  for ( int i = 0; i < parts - 1; i++){
    gput->EnqueueWriteBuffer( 8, bvhs[i]->gpuNodes, sizeof(GPUNode)*bvhs[i]->nodeNum);
    gput->EnqueueWriteBuffer(0, vertices[i], sizeof(float)*9*bvhs[i]->primitives.size());
    gput->SetIntArgument(14, bvhs[i]->topLevelNodes);
    gput->Run();
    gput->WaitForKernel();
  }
  gput->SetIntArgument(14, bvhs[parts-1]->topLevelNodes);
  gput->EnqueueWriteBuffer( 8, bvhs[parts-1]->gpuNodes, sizeof(GPUNode)*bvhs[parts-1]->nodeNum);
  gput->EnqueueWriteBuffer(0, vertices[parts-1], sizeof(cl_float)*9*bvhs[parts-1]->primitives.size());
  gput->Run();

  gput->EnqueueReadBuffer( 6, occluded );
  #ifdef STAT_PRAY_TRIANGLE
    gput->EnqueueReadBuffer( 9, picture);
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


RayBVH *CreateRayBVH(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    int chunkX = ps.FindOneInt("chunkXSize", 4);
    int chunkY = ps.FindOneInt("chunkYSize", 4);
    int height = ps.FindOneInt("height",3);
    int BVHheight = ps.FindOneInt("BVHheight",3);
    int maxBVHPrim = ps.FindOneInt("maxBVHPrim",1);
    string node = ps.FindOneString("node", "box_dir");
    string splitMethod = ps.FindOneString("splitmethod", "sah");
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    int scale = ps.FindOneInt("scale",50);
    #endif
    return new RayBVH(prims,onGPU,chunkX, chunkY,height, BVHheight, node, maxBVHPrim, splitMethod
    #if (defined STAT_RAY_TRIANGLE || defined STAT_PRAY_TRIANGLE)
    , scale
    #endif
    );
}


