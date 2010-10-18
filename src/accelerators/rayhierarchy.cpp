#include "accelerators/rayhierarchy.h"
#include "probes.h"
#include "paramset.h"
#include "intersection.h"
#include "GPUparallel.h"
#include <iostream>
#if ( defined STAT_RAY_TRIANGLE || defined STAT_TRIANGLE_CONE)
#include "imageio.h"
#endif

#define KERNEL_COMPUTEDPTUTV 0
#define KERNEL_RAYLEVELCONSTRUCT 1
#define KERNEL_RAYCONSTRUCT 2
#define KERNEL_INTERSECTIONP 4
#define KERNEL_INTERSECTIONR 3
#define KERNEL_YETANOTHERINTERSECTION 5


using namespace std;
Semaphore *workerSemaphore;

// RayHieararchy Method Definitions
RayHieararchy::RayHieararchy(const vector<Reference<Primitive> > &p, bool onG, int chunk, int height,
  string node) {
    this->chunk = chunk;
    this->height = height;
    triangleCount = 0;
    onGPU = onG;

    Info("Created OpenCL context");
    //precompile OpenCL kernels
    size_t lenP = strlen(PbrtOptions.pbrtPath);
    size_t *lenF = new size_t[6];
    char** names = new char*[6];
    char** file = new char*[6];

  nodeSize = 0;
  if ( node == "cone"){
    names[0] = "cl/intersectionR.cl";
    names[1] = "cl/intersectionP.cl";
    names[2] = "cl/rayhconstruct.cl";
    names[3] = "cl/levelConstruct.cl";
    names[4] = "cl/yetAnotherIntersection.cl";
    cout << "accel nodes : cones" << endl;
    nodeSize = 8;
  }
  if ( node == "ia"){
    names[0] = "cl/intersectionIA.cl";
    names[1] = "cl/intersectionPIA.cl";
    names[2] = "cl/rayhconstructIA.cl";
    names[3] = "cl/levelConstructIA.cl";
    names[4] = "cl/yetAnotherIntersectionIA.cl";
    cout << "accel nodes : IA" << endl;
    nodeSize = 13;
  }
  if ( node == "sphere_uv"){
    names[0] = "cl/intersection5D.cl";
    names[1] = "cl/intersectionP5D.cl";
    names[2] = "cl/rayhconstruct5D.cl";
    names[3] = "cl/levelConstruct5D.cl";
    names[4] = "cl/yetAnotherIntersection5D.cl";
    cout << "accel nodes : 5D nodes with spheres" << endl;
    nodeSize = 9;
  }
  if ( node == "box_uv") {
    names[0] = "cl/intersection5DB.cl";
    names[1] = "cl/intersectionP5DB.cl";
    names[2] = "cl/rayhconstruct5DB.cl";
    names[3] = "cl/levelConstruct5DB.cl";
    names[4] = "cl/yetAnotherIntersection5DB.cl";
    cout << "accel nodes : 5D nodes with boxes" << endl;
    nodeSize = 11;
  }
  if ( nodeSize == 0){
    cout << "Unknown accelarator node type " << node << endl;
    abort();
  }
    names[5] = "cl/computeDpTuTv.cl";

    for ( int i = 0; i < 6; i++){
      lenF[i] = strlen(names[i]);
      file[i] = new char[lenP + lenF[i] + 1];
      strncpy(file[i], PbrtOptions.pbrtPath, lenP);
      strncpy(file[i]+lenP, names[i], lenF[i]+1);
    }

    ocl = new OpenCL(onGPU,6);
    cmd = ocl->CreateCmdQueue();
    ocl->CompileProgram(file[0], "IntersectionR", "oclIntersection.ptx", KERNEL_INTERSECTIONR);
    ocl->CompileProgram(file[2], "rayhconstruct", "oclRayhconstruct.ptx",KERNEL_RAYCONSTRUCT);
    ocl->CompileProgram(file[1], "IntersectionP", "oclIntersectionP.ptx", KERNEL_INTERSECTIONP);
    ocl->CompileProgram(file[3], "levelConstruct", "oclLevelConstruct.ptx",KERNEL_RAYLEVELCONSTRUCT);
    ocl->CompileProgram(file[4], "YetAnotherIntersection", "oclYetAnotherIntersection.ptx", KERNEL_YETANOTHERINTERSECTION);
    ocl->CompileProgram(file[5], "computeDpTuTv", "oclcomputeDpTuTv.ptx", KERNEL_COMPUTEDPTUTV);

    delete [] lenF;
    for (int i = 0; i < 6; i++){
      //delete [] names[i];
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
    workerSemaphore = new Semaphore(1);

   //do brute force factorization -> compute ideal small rectangle sides sizes
   vector<unsigned int> primes;
   unsigned int number = chunk;

   for (unsigned int k = 2; k <= number; k++){
     while ( number % k == 0 )   {
       primes.push_back(k);
       number /= k;
     }
   }

  //compute sides of small rectangles
  a = b = 1;
  for (unsigned int k = 0; k < primes.size(); k++){
    if ( k % 2 == 0 ){
      a *= primes[k];
    } else {
      b *= primes[k];
    }
  }
}

RayHieararchy::~RayHieararchy() {
  ocl->DeleteCmdQueue(cmd);
  #ifdef GPU_TIMES
  ocl->PrintTimes();
  #endif
  delete ocl;
  delete [] vertices;
  delete [] uvs;
  delete workerSemaphore;
}

BBox RayHieararchy::WorldBound() const {
    return bbox;
}

unsigned int RayHieararchy::MaxRaysPerCall(){
    //TODO: check the OpenCL device and decide, how many rays can be processed at one thread
    // check how many threads can be proccessed at once
    return 100000;
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

size_t topLevelCount;

//constructs ray hieararchy on GPU -> creates array of cones
size_t RayHieararchy::ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, cl_uint count,
  cl_uint* countArray, unsigned int threadsCount ){

  Assert(height > 0);
  size_t tn = ocl->CreateTask(KERNEL_RAYCONSTRUCT, threadsCount , cmd, 32);
  OpenCLTask* gpuray = ocl->getTask(tn,cmd);

  int total = 0;
  int levelcount = threadsCount;
  for ( cl_uint i = 0; i < height; i++){
      total += levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
      if ( levelcount == 1 ){
        height = i;
        break;
      }
  }
  topLevelCount = levelcount;
  total += levelcount;
  if ( height < 2) Severe("Too few rays for rayhierarchy! Try smaller chunks");
  gpuray->InitBuffers(5);

  Assert(gpuray->CreateBuffer(0,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY )); //ray directions
  Assert(gpuray->CreateBuffer(1,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY )); //ray origins
  Assert(gpuray->CreateBuffer(2,sizeof(cl_uint)*threadsCount, CL_MEM_READ_ONLY)); //number of rays per initial thread
  Assert(gpuray->CreateBuffer(3, sizeof(cl_float)*nodeSize*total, CL_MEM_READ_WRITE)); //for cones
  Assert(gpuray->CreateBuffer(4, sizeof(cl_int)*2*total, CL_MEM_WRITE_ONLY)); //for pointers to leaves
  Assert(gpuray->SetIntArgument(5,(cl_uint)threadsCount));

  Assert(gpuray->EnqueueWriteBuffer( 0, rayDir));
  Assert(gpuray->EnqueueWriteBuffer( 1, rayO));
  Assert(gpuray->EnqueueWriteBuffer( 2, countArray));

  Assert(gpuray->Run());

  Assert(!gpuray->SetPersistentBuff(0));
  Assert(!gpuray->SetPersistentBuff(1));
  Assert(!gpuray->SetPersistentBuff(3));
  Assert(!gpuray->SetPersistentBuff(4));

  size_t tasknum = ocl->CreateTask(KERNEL_RAYLEVELCONSTRUCT, (threadsCount+1)/2, cmd,32);
  OpenCLTask* gpurayl = ocl->getTask(tasknum,cmd);
  gpurayl->InitBuffers(2);
  gpurayl->CopyBuffer(3,0,gpuray);
  gpurayl->CopyBuffer(4,1,gpuray);
  if (!gpurayl->SetIntArgument(2,(cl_uint)count)) exit(EXIT_FAILURE);
  if (!gpurayl->SetIntArgument(3,(cl_uint)threadsCount)) exit(EXIT_FAILURE);

  int temp = global_a;
  for ( cl_uint i = 1; i <= height; i++){
    if (!gpurayl->SetIntArgument(4,i)) exit(EXIT_FAILURE);
    if (!gpurayl->SetIntArgument(5,temp)) exit(EXIT_FAILURE);
    if (!gpurayl->Run())exit(EXIT_FAILURE);
    if ( i % 2 == 0){
      temp = (temp+1)/2;
    }
    gpurayl->WaitForKernel();
  }

  Assert(!gpurayl->SetPersistentBuff(0));
  Assert(!gpurayl->SetPersistentBuff(1));
  ocl->delTask(tasknum, cmd);

  return tn; //return index to first task - so that buffers can be copied
}

//intersect computed on gpu with more rays
void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                               float* rayWeight, bool* hit, unsigned int count,
                               const unsigned int & xResolution, const unsigned int & yResolution,
                               const unsigned int & samplesPerPixel
  #ifdef STAT_RAY_TRIANGLE
  , Spectrum *Ls
  #endif
  )  {
  #ifdef GPU_TIMES
  cout << "# triangles: " << triangleCount << endl;
  #endif
  this->xResolution = xResolution;
  this->samplesPerPixel = samplesPerPixel;
  //number of rectangles in x axis
  global_a = (xResolution + a - 1) / a; //round up -> +a-1
  //number of rectangles in y axis
  global_b = (yResolution + b - 1) / b;
  //x and y sizes of overlapping area
  rest_x = global_a*a - xResolution;
  rest_y = global_b*b - yResolution;
  threadsCount = global_a * global_b;

  cl_float* rayDirArray = new cl_float[count*3];
  cl_float* rayOArray = new cl_float[count*3];
  cl_float* rayBoundsArray = new cl_float[count*2];
  cl_float* tHitArray = new cl_float[count];
  cl_uint* indexArray = new cl_uint[count];
  cl_uint* countArray = new cl_uint[threadsCount]; //number of rays to deal with for every thread
  unsigned int* elem_index = new unsigned int [count];
  #ifdef STAT_RAY_TRIANGLE
  cl_uint* picture = new cl_uint[count];
  #endif
  #ifdef STAT_TRIANGLE_CONE
  cl_uint* picture = new cl_uint[triangleCount];
  #endif
  unsigned int c;

  unsigned int ix, iy, global_ix, global_iy;
  unsigned help;

  unsigned int new_a, new_b;
  unsigned int elem_counter = 0;
  unsigned int number;

  //TODO: try Z-curve order?
  //store those rectangles, loop through all threads <==> all rectangles
  for (unsigned int k = 0; k < threadsCount; ++k){
    new_a = a; new_b = b;

    global_iy = k / global_a;
    global_ix = k - global_iy*global_a;
    number = chunk;
    if ( global_ix == (global_a -1) ){
      number -= rest_x*b;
      new_a -= rest_x;
    }
    if ( global_iy == (global_b -1) ){
      number -= rest_y*new_a;
      new_b -= rest_y;
    }

    countArray[k] = number*samplesPerPixel;
    //loop inside small rectangle
    for (unsigned int j = 0; j < number; j++){
      iy = j / new_a;
      ix = j - iy*new_a;

      help = iy*xResolution*samplesPerPixel + ix*samplesPerPixel + global_ix*a*samplesPerPixel + global_iy*xResolution*b*samplesPerPixel;
      for ( unsigned int s = 0; s < samplesPerPixel; s++){
        if ( help >= count) {
          --countArray[k];
          continue;
        }
        elem_index[help] = elem_counter;
        rayDirArray[3*elem_counter] = r[help].d[0];
        rayDirArray[3*elem_counter + 1] = r[help].d[1];
        rayDirArray[3*elem_counter + 2] = r[help].d[2];

        rayOArray[3*elem_counter] = r[help].o[0];
        rayOArray[3*elem_counter + 1] = r[help].o[1];
        rayOArray[3*elem_counter + 2] = r[help].o[2];

        rayBoundsArray[2*elem_counter] = r[help].mint;
        rayBoundsArray[2*elem_counter + 1] = INFINITY;

        indexArray[elem_counter] = 0;
        tHitArray[elem_counter] = INFINITY-1; //should initialize on scene size

        #ifdef STAT_RAY_TRIANGLE
        ((cl_uint*)picture)[elem_counter] = 0;
        #endif
        #ifdef STAT_TRIANGLE_CONE
        if ( elem_counter < triangleCount)
          ((cl_uint*)picture)[elem_counter] = 0;
        #endif

        ++elem_counter;
        ++help;
      }
    }

  }

    workerSemaphore->Wait();
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, count, countArray, threadsCount);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    size_t tn2 = ocl->CreateTask(KERNEL_INTERSECTIONR, triangleCount, cmd,32);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);

    #if (!defined STAT_RAY_TRIANGLE && !defined STAT_RAY_CONE)
    c = 8;
    gput->InitBuffers(8);
    #endif
    #if (defined STAT_RAY_TRIANGLE || defined STAT_TRIANGLE_CONE)
    c = 9;
    gput->InitBuffers(9);
    #endif

    gput->CopyBuffer(0,1,gpuray);
    gput->CopyBuffer(1,2,gpuray);
    gput->CopyBuffer(3,3,gpuray);
    gput->CopyBuffer(4,4,gpuray);


    Assert(gput->CreateBuffer(0,sizeof(cl_float)*3*3*triangleCount, CL_MEM_READ_ONLY )); //vertices
    Assert(gput->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_READ_ONLY )); //ray bounds
    Assert(gput->CreateBuffer(6,sizeof(cl_float)*count, CL_MEM_READ_WRITE)); // tHit
    Assert(gput->CreateBuffer(7,sizeof(cl_uint)*count, CL_MEM_WRITE_ONLY)); //index array
    #ifdef STAT_TRIANGLE_CONE
    Assert(gput->CreateBuffer(8,sizeof(cl_uint)*triangleCount, CL_MEM_WRITE_ONLY));
    #endif
    #ifdef STAT_RAY_TRIANGLE
    Assert(gput->CreateBuffer(8,sizeof(cl_uint)*count, CL_MEM_WRITE_ONLY));
    #endif

    //while outside for: 32*(toplevelCount*2 + (height+1)*(2+height)/2)
    //while inside for: 32*(2 + (height+1)*(2+height)/2)
    Assert(gput->SetLocalArgument(c++,sizeof(cl_int)*(32*(2 + (height+1)*(2+height)/2)))); //stack for every thread
    Assert(gput->SetIntArgument(c++,(cl_int)count));
    Assert(gput->SetIntArgument(c++,(cl_int)triangleCount));
    Assert(gput->SetIntArgument(c++,(cl_int)height));
    Assert(gput->SetIntArgument(c++,(cl_int)threadsCount));

    if (!gput->EnqueueWriteBuffer( 0, vertices ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer( 5, rayBoundsArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer( 6, tHitArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer( 7, indexArray ))exit(EXIT_FAILURE);
    #if (defined STAT_RAY_TRIANGLE || defined STAT_TRIANGLE_CONE)
    Assert(gput->EnqueueWriteBuffer( 8, picture));
    #endif

    if (!gput->Run())exit(EXIT_FAILURE);
    gput->WaitForKernel();

    #ifdef STAT_RAY_TRIANGLE
    Assert(gput->EnqueueReadBuffer( 8, picture));
    uint i = 0;
    uint temp;
    gput->WaitForRead();
    float colors[3];
    for (i = 0; i < count; i++){
      colors[0] = picture[elem_index[i]]/25;
      colors[1] = (picture[elem_index[i]] - (uint)colors[0]*25)/5;
      colors[2] = (picture[elem_index[i]] - (uint)colors[0]*25 - (uint)colors[1]*5);
      Ls[i] = RGBSpectrum::FromRGB(colors);
    }
    delete [] ((uint*)picture);
    workerSemaphore->Post();
    return;
    #endif

    #ifdef STAT_TRIANGLE_CONE
    Assert(gput->EnqueueReadBuffer( 8, picture));
    cout << endl << "triangle cone intersections ";
    uint i = 0;
    for ( i = 0; i < triangleCount; i++)
      cout << (((uint*)picture)[i]) << ' ';
    cout <<  endl;
    abort();
    #endif

    Assert(!gput->SetPersistentBuffers(0,8)); //vertex,dir,o, cones, ray bounds, tHit, indexArray

    //counter for changes in ray-triangle intersection
    cl_uint* changedArray = new cl_uint[triangleCount];
    memset(changedArray, 0, sizeof(cl_uint)*triangleCount);

    size_t tn4 = ocl->CreateTask(KERNEL_YETANOTHERINTERSECTION, triangleCount , cmd, 32);
    OpenCLTask* anotherIntersect = ocl->getTask(tn4, cmd);
    anotherIntersect->InitBuffers(9);
    anotherIntersect->CopyBuffers(0,8,0,gput);
    Assert(anotherIntersect->CreateBuffer(8,sizeof(cl_uint)*triangleCount, CL_MEM_WRITE_ONLY)); //recording changes

    Assert(anotherIntersect->SetLocalArgument(9,sizeof(cl_int)*(32*(2 + (height+1)*(2+height)/2)))); //stack for every thread
    Assert(anotherIntersect->SetIntArgument(10,(cl_int)count));
    Assert(anotherIntersect->SetIntArgument(11,(cl_int)triangleCount));
    Assert(anotherIntersect->SetIntArgument(12,(cl_int)height));
    Assert(anotherIntersect->SetIntArgument(13,(cl_int)threadsCount));
    //TODO: make only one counter per work-group, use local memory
    if (!anotherIntersect->EnqueueWriteBuffer( 8 , changedArray)) exit(EXIT_FAILURE);
    if (!anotherIntersect->Run())exit(EXIT_FAILURE);
    if (!anotherIntersect->EnqueueReadBuffer( 8, changedArray ))exit(EXIT_FAILURE);
    if (!anotherIntersect->EnqueueReadBuffer( 6, tHitArray)) exit(EXIT_FAILURE);
    if (!anotherIntersect->EnqueueReadBuffer( 7, indexArray)) exit(EXIT_FAILURE);
    anotherIntersect->WaitForRead();
    /*for ( size_t i = 0; i < triangleCount; i++){
      if ( changedArray[i] != 0){
        cout << "OPRAVA triangle " << i << " ray " << changedArray[i]
        << " tHit " << tHitArray[changedArray[i] ] << endl;
        cout << "rayDir " << rayDirArray[3*changedArray[i]] << ' ' << rayDirArray[3*changedArray[i]+1 ]
              << ' ' << rayDirArray[3*changedArray[i]+2] << endl;
        cout << "rayO " << rayOArray[3*changedArray[i]] << ' ' << rayOArray[3*changedArray[i]+1 ]
              << ' ' << rayOArray[3*changedArray[i]+2] << endl;
      }
    }*/

    Assert(!anotherIntersect->SetPersistentBuff(0));//vertex
    Assert(!anotherIntersect->SetPersistentBuff(1));//dir
    Assert(!anotherIntersect->SetPersistentBuff(2));//origin
    Assert(!anotherIntersect->SetPersistentBuff(7));//index

    size_t tn3 = ocl->CreateTask(KERNEL_COMPUTEDPTUTV, count, cmd, 32);
    OpenCLTask* gpuRayO = ocl->getTask(tn3,cmd);
    gpuRayO->InitBuffers(7);
    gpuRayO->CopyBuffers(0,3,0,gput); // 0 vertex, 1 dir, 2 origin
    gpuRayO->CopyBuffer(7,3,gput); // 3 index
    Assert(gpuRayO->CreateBuffer(4,sizeof(cl_float)*6*triangleCount, CL_MEM_READ_ONLY )); //uvs
    Assert(gpuRayO->CreateBuffer(5,sizeof(cl_float)*2*count, CL_MEM_WRITE_ONLY )); // tu,tv
    Assert(gpuRayO->CreateBuffer(6,sizeof(cl_float)*6*count, CL_MEM_WRITE_ONLY )); //dpdu, dpdv

    if (!gpuRayO->SetIntArgument(7,(cl_uint)count)) exit(EXIT_FAILURE);

    if (!gpuRayO->EnqueueWriteBuffer( 4 , uvs)) exit(EXIT_FAILURE);
    if (!gpuRayO->Run())exit(EXIT_FAILURE);

    Info("count %d", count);
    cl_float* tutvArray = new cl_float[2*count]; // for tu, tv
    cl_float* dpduArray = new cl_float[2*3*count]; // for dpdu, dpdv
    if (!gpuRayO->EnqueueReadBuffer( 5, tutvArray ))exit(EXIT_FAILURE);
    if (!gpuRayO->EnqueueReadBuffer( 6, dpduArray ))exit(EXIT_FAILURE);

    gpuRayO->WaitForRead();
    ocl->delTask(tn1,cmd);
    ocl->delTask(tn2,cmd);
    ocl->delTask(tn3,cmd);
    ocl->delTask(tn4,cmd);
    workerSemaphore->Post();
    cl_uint index;

    Vector dpdu, dpdv;
    //deserialize rectangles
    unsigned int j;
    for ( unsigned int i = 0; i < count; i++) {
        j =  elem_index[i];
        Assert( j < count);
        index = indexArray[j];
        hit[i] = false;
        if ( !index ) continue;
        Assert( index < triangleCount);
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(dpduArray[6*j],dpduArray[6*j+1],dpduArray[6*j+2]);
        dpdv = Vector(dpduArray[6*j+3],dpduArray[6*j+4],dpduArray[6*j+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](tHitArray[j]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*j], tutvArray[2*j+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](tHitArray[j]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*j],tutvArray[2*j+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * tHitArray[j]; //thit
        r[i].maxt = tHitArray[j];
        hit[i] = true;
        PBRT_RAY_TRIANGLE_INTERSECTION_HIT(&r[i], r[i].maxt);
    }

    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] tHitArray;
    delete [] tutvArray;
    delete [] dpduArray;
    delete [] elem_index;
    delete [] countArray;
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

#ifdef GPU_TIMES
bool first_run = true;
#endif

void RayHieararchy::IntersectP(const Ray* r, unsigned char* occluded, const size_t count, const bool* hit
    #ifdef STAT_PRAY_TRIANGLE
    , Spectrum *Ls
    #endif
) {
  cl_float* rayDirArray = new cl_float[count*3]; //ray directions
  cl_float* rayOArray = new cl_float[count*3]; //ray origins
  cl_float* rayBoundsArray = new cl_float[count*2]; //ray bounds
  cl_uint* countArray = new cl_uint[threadsCount];
  size_t* size = new size_t[20];

  unsigned int elem_counter = 0;
  #ifdef STAT_PRAY_TRIANGLE
  cl_uint* picture = new cl_uint[count];
  #endif


  for (int k = 0; k < count; ++k) {
    if ( !hit[k] ){
          continue;
   }
   rayDirArray[3*elem_counter] = r[k].d[0];
   rayDirArray[3*elem_counter+1] = r[k].d[1];
   rayDirArray[3*elem_counter+2] = r[k].d[2];

   rayOArray[3*elem_counter] = r[k].o[0];
   rayOArray[3*elem_counter+1] = r[k].o[1];
   rayOArray[3*elem_counter+2] = r[k].o[2];

   rayBoundsArray[2*elem_counter] = r[k].mint;
   rayBoundsArray[2*elem_counter+1] = INFINITY;
   occluded[elem_counter] = '0';

  ++elem_counter;
  }

  threadsCount = (elem_counter + chunk - 1)/chunk;
  for (int k = 0; k < threadsCount; ++k)
    countArray[k] = chunk;
  //memset(countArray, chunk, sizeof(cl_uint)*number);
  countArray[threadsCount - 1] -= (chunk*threadsCount - elem_counter);

    #ifdef GPU_TIMES
    if ( first_run) {
      cout << "# prays: " << elem_counter << " all rays: " << count << endl;
      first_run = false;
    }
    #endif

    if ( elem_counter == 0 ){
      //nothing to intersect
      #ifdef GPU_TIMES
      first_run = true;
      #endif
      delete [] rayDirArray;
      delete [] rayOArray;
      delete [] rayBoundsArray;
      delete [] countArray;
      delete [] size;
      return;
    }

    workerSemaphore->Wait();
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, elem_counter, countArray, threadsCount);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    Info("height of the ray hieararchy in IntersectP is %d",height);
    size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTIONP, elem_counter, cmd,32);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);

    unsigned int c = 7;
    #ifdef STAT_PRAY_TRIANGLE
     c = 8;
    #endif
    gput->InitBuffers(c);

    Assert(gput->CreateBuffer(0,sizeof(cl_float)*3*3*triangleCount, CL_MEM_READ_ONLY )); //vertices
    gput->CopyBuffer(0,1,gpuray); //ray dir
    gput->CopyBuffer(1,2,gpuray); //ray o
    gput->CopyBuffer(3,3,gpuray); //cones
    gput->CopyBuffer(4,4,gpuray); //pointers to leaves
    Assert(gput->CreateBuffer(5,sizeof(cl_float)*2*elem_counter, CL_MEM_READ_ONLY)); //ray bounds
    Assert(gput->CreateBuffer(6,sizeof(cl_uchar)*elem_counter, CL_MEM_READ_WRITE)); //tHit
    #ifdef STAT_PRAY_TRIANGLE
    Assert(gput->CreateBuffer(7,sizeof(cl_uint)*elem_counter, CL_MEM_WRITE_ONLY));
    #endif

    if (!gput->SetLocalArgument(c++,sizeof(cl_int)*sizeof(cl_int)*(32*(2 + (height+1)*(2+height)/2))));
    if (!gput->SetIntArgument(c++,(cl_uint)elem_counter)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument(c++,(cl_uint)triangleCount)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument(c++,(cl_uint)height)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument(c++,(cl_uint)threadsCount)) exit(EXIT_FAILURE);

    if (!gput->EnqueueWriteBuffer( 0, vertices ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer( 5, rayBoundsArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer( 6, occluded ))exit(EXIT_FAILURE);
    #ifdef STAT_PRAY_TRIANGLE
    Assert(gput->EnqueueWriteBuffer( 7, picture));
    #endif
    if (!gput->Run())exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer( 6, occluded ))exit(EXIT_FAILURE);
    #ifdef STAT_PRAY_TRIANGLE
    Assert(gput->EnqueueReadBuffer( 7, picture));
    #endif

    gput->WaitForRead();
    workerSemaphore->Post();
    unsigned int j = 0;
    unsigned char* temp_occluded = new unsigned char[count];
    for ( unsigned i = 0; i < count; i++){
      if ( !hit[i]) {
        continue;
      }
      temp_occluded[i] = occluded[j];
      ++j;
    }
    for ( unsigned i = 0; i < count; i++){
      occluded[i] = temp_occluded[i];
    }
    delete [] temp_occluded;

    #ifdef STAT_PRAY_TRIANGLE
    uint i = 0;
    j = 0;
    uint temp;
    for (i = 0; i < count; i++){
      if (!hit[i]) continue;
      temp = ((cl_uint*)picture)[j]; ///256;
     // if ( temp > 0 ) cout << "spectrum " << temp << endl;
      Ls[i] += Spectrum(temp);
      Ls[i] = Ls[i].Clamp(0,255);
      ++j;
    }
    delete [] ((uint*)picture);
    #endif

    #ifdef PBRT_STATS_COUNTERS
    for ( unsigned i = 0; i < count; i++){
        PBRT_FINISHED_RAY_INTERSECTIONP(&r[i], int(occluded[i]));
        if (occluded[i] != '0')
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&r[i],0);
    }
    #endif

    ocl->delTask(tn1,cmd);
    ocl->delTask(tn2,cmd);
    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] size;
    delete [] countArray;
}


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    int chunk = ps.FindOneInt("chunkSize",20);
    int height = ps.FindOneInt("height",3);
    string node = ps.FindOneString("node", "sphere_uv");
    return new RayHieararchy(prims,onGPU,chunk,height,node);
}


