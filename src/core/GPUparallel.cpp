#include "GPUparallel.h"
#include <float.h>
using namespace std;

cl_context OpenCL::cxContext; //OpenCL context
#ifdef GPU_TIMES
double* kernel_times;
unsigned int* kernel_calls;
double wmem_times;
double rmem_times;
#endif

OpenCL::~OpenCL(){
  for ( size_t i = 0; i < numKernels; i++)
    if (cpPrograms[i]) clReleaseProgram(cpPrograms[i]);
  delete [] cpPrograms;
  delete [] functions;
  for ( size_t i = 0; i < numqueues; i++)
    delete queue[i];
  delete [] queue;
  clReleaseContext(cxContext);
  Info("Released OpenCL context");
  Mutex::Destroy(mutex);
}

OpenCL::OpenCL(bool onGPU, size_t numKernels){
// create the OpenCL context on a GPU device
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
      Severe("clGetPlatformIDs failed.");
    }
    if (0 < numPlatforms)
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
          Severe("clGetPlatformIDs failed.");
        }
        for (unsigned i = 0; i < numPlatforms; ++i)
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);

            if (status != CL_SUCCESS) {
              Severe("clGetPlatformIDs failed.");
            }

            platform = platforms[i];
            if ( (onGPU && !strcmp(pbuf, "NVIDIA Corporation")) ||
            (!onGPU && !strcmp(pbuf, "Advanced Micro Devices, Inc.")))
            {
                  /*cl_device_id device_id;
                  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
                  if ( err != CL_SUCCESS)
                    Severe("Error geting device ID %i", err);
                  cl_ulong size_max;
                  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_max), &size_max, NULL);
                  cout << "Maxim size of local memory " << size_max << endl;
                  abort();*/
                break;
            }
        }
        delete[] platforms;
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    /* Use NULL for backward compatibility */
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;

  if (onGPU)
    cxContext = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
  else
    cxContext = clCreateContextFromType(cprops, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL); //works only with ATI-stream SDK
  if ( cxContext == (cl_context)0){
    Severe("Error creating context ");
  }
  mutex = Mutex::Create();
  maxqueues = 100;
  numqueues = 0;
  queue = new OpenCLQueue*[maxqueues];
  for (size_t i = 0; i < maxqueues; i++)
    queue[i] = 0;

  this->numKernels = numKernels;

  cpPrograms = new cl_program[numKernels];
  functions = new const char*[numKernels];
  #ifdef GPU_TIMES
  kernel_times = new double[numKernels];
  kernel_calls = new unsigned int[numKernels];
  for ( size_t i = 0; i < numKernels; i++)
    kernel_times[i] = kernel_calls[i] = 0;
  wmem_times = rmem_times = 0;
  #endif
}

Mutex* buildLog;

OpenCLQueue::~OpenCLQueue(){
  for ( size_t i = 0; i < numtasks; i++){
    if ( tasks[i] != NULL) delete tasks[i];
    tasks[i] = NULL;
  }
  if ( tasks != NULL) delete [] tasks;
  tasks = NULL;
  clReleaseCommandQueue(cmd_queue);
}

OpenCLQueue::OpenCLQueue(cl_context & context){
  buildLog = Mutex::Create();
  size_t cb;
  cl_device_id* devices;
  // get the list of GPU devices associated with context
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  devices = new cl_device_id[cb];
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);
  // create a command-queue
  device = devices[0];
  #ifdef GPU_TIMES
  cmd_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE , NULL); //take first device
  #else
  cmd_queue = clCreateCommandQueue(context, devices[0], NULL , NULL);
  #endif
  if (cmd_queue == (cl_command_queue)0)
  {
      clReleaseContext(context);
      free(devices);
      abort();
  }
  /*#ifdef GPU_TIMES
      cl_int ciErrNum = clSetCommandQueueProperty(cmd_queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
      if (ciErrNum != CL_SUCCESS)
          Severe(" Error %i %s in clSetCommandQueueProperty call !!!\n\n", ciErrNum, stringError(ciErrNum));
  #endif*/
  delete [] devices;
  maxtasks = 3;
  numtasks = 0;
  tasks = new OpenCLTask*[maxtasks];
  for ( unsigned int i = 0; i < maxtasks; i++)
    tasks[i] = 0;
  globalmutex = Mutex::Create();
}

void pfn_notify(cl_program p, void* user_data){
  MutexLock(*buildLog);
 char* log = new char[1000];
 size_t len;
 cl_int ciErrNum = clGetProgramBuildInfo(p, *((cl_device_id*)user_data), CL_PROGRAM_BUILD_LOG, sizeof(char)*1000, log, &len);
 if ( ciErrNum != CL_SUCCESS)
  Severe("Failed to get program build log %s %i", stringError(ciErrNum), ciErrNum);
 //cout << endl << (char*)user_data<< " len " << len << " " << log << endl;
 cout << endl << " len " << len << " " << log << endl;
 delete [] log;

 return;
}

char* oclLoadSource(const char* cFilename, char** headers, const int headerNum, size_t* szFinalLength)
{
    // locals
    FILE* pFileStream = NULL;
    FILE** pHeaderStream = new FILE*[headerNum];
    for ( int i = 0; i < headerNum; i++)
      pHeaderStream[i] = NULL;
    size_t* headerLength = new size_t[headerNum];
    size_t totalHeaderLen = 0;
    size_t szSourceLength;

    // open the OpenCL source code file
    #ifdef _WIN32   // Windows version
        if(fopen_s(&pFileStream, cFilename, "rb") != 0)
        {
            return NULL;
        }
        for ( int i = 0; i < headerNum; i++){
          if(fopen_s(&pHeaderStream[i], headers[i], "rb") != 0)
          {
              return NULL;
          }
        }
    #else           // Linux version
        pFileStream = fopen(cFilename, "rb");
        if(pFileStream == 0)
        {
            return NULL;
        }
        for ( int i = 0; i < headerNum; i++){
          pHeaderStream[i] = fopen(headers[i], "rb");
          if(pHeaderStream[i] == 0)
          {
              return NULL;
          }
        }
    #endif

    for ( int i = 0; i < headerNum; i++){
      fseek(pHeaderStream[i], 0, SEEK_END);
      headerLength[i] = ftell(pHeaderStream[i]);
      totalHeaderLen+= headerLength[i];
      fseek(pHeaderStream[i], 0, SEEK_SET);
    }

    // get the length of the source code
    fseek(pFileStream, 0, SEEK_END);
    szSourceLength = ftell(pFileStream);
    fseek(pFileStream, 0, SEEK_SET);

    // allocate a buffer for the source code string and read it in
    char* cSourceString = (char *)malloc(szSourceLength + totalHeaderLen + 1);
    totalHeaderLen = 0;
    for ( int i = 0; i < headerNum; i++){
      if (fread((cSourceString) + totalHeaderLen, headerLength[i], 1, pHeaderStream[i]) != 1){
        fclose(pHeaderStream[i]);
        fclose(pFileStream);
        free(cSourceString);
        return 0;
      }
      totalHeaderLen += headerLength[i];
      fclose(pHeaderStream[i]);
    }
    if (fread((cSourceString) + totalHeaderLen, szSourceLength, 1, pFileStream) != 1)
    {
        fclose(pFileStream);
        free(cSourceString);
        return 0;
    }

    // close the file and return the total length of the combined (preamble + source) string
    fclose(pFileStream);

    if(szFinalLength != 0)
    {
        *szFinalLength = szSourceLength + totalHeaderLen;
    }
    cSourceString[szSourceLength + totalHeaderLen] = '\0';
    delete [] pHeaderStream;
    delete [] headerLength;

    return cSourceString;
}

void OpenCL::CompileProgram(const char* cPathAndName, const char* function,
      const char* program, size_t i, char** headers, const int headerNum){
    char* cSourceCL ;         // Buffer to hold source for compilation
    size_t szKernelLength;			// Byte size of kernel code
    cl_int ciErrNum;
  MutexLock(*buildLog);

  if ( !headerNum )
    cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
  else
    cSourceCL = oclLoadSource(cPathAndName, headers, headerNum, &szKernelLength);
  if ( cSourceCL == NULL){
    Severe( "File \"%s\" not found ",cPathAndName);
  }

  // Create the program
  cpPrograms[i] = clCreateProgramWithSource(cxContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

#ifdef HAVE_ATI
  ciErrNum = clBuildProgram(cpPrograms[i], 0, NULL,
  #ifdef TRIANGLES_PER_THREAD
   "-DTRIANGLES_PER_THREAD -g",
  #endif
  #ifdef STAT_RAY_TRIANGLE
   "-DSTAT_RAY_TRIANGLE -g",
  #endif
  #ifdef STAT_PRAY_TRIANGLE
    "-DSTAT_PRAY_TRIANGLE -g",
  #endif
  #ifdef STAT_TRIANGLE_CONE
    "-DSTAT_TRIANGLE_CONE -g",
  #endif
  #if ( !defined STAT_RAY_TRIANGLE && !defined STAT_TRIANGLE_CONE && !defined STAT_PRAY_TRIANGLE)
    "-g",
  #endif
  NULL, NULL);
#else
  ciErrNum = clBuildProgram(cpPrograms[i], 0, NULL,
  #ifdef STAT_RAY_TRIANGLE
   "-DSTAT_RAY_TRIANGLE",
  #endif
  #ifdef STAT_PRAY_TRIANGLE
    "-DSTAT_PRAY_TRIANGLE",
  #endif
  #ifdef STAT_TRIANGLE_CONE
    "-DSTAT_TRIANGLE_CONE",
  #endif
  #if ( !defined STAT_RAY_TRIANGLE && !defined STAT_TRIANGLE_CONE && !defined STAT_PRAY_TRIANGLE)
    "-cl-nv-verbose -cl-nv-maxrregcount=90 -Werror",
  #endif
  &pfn_notify, (void*) &(queue[0]->device));
#endif

  if (ciErrNum != CL_SUCCESS){
    // write out standard error, Build Log and PTX, then cleanup and exit
    shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
    oclLogBuildInfo(cpPrograms[i], oclGetFirstDev(cxContext));
    oclLogPtx(cpPrograms[i], oclGetFirstDev(cxContext), program);
    Severe( "Failed building program \"%s\" !", function);
  }

  size_t size;
  clGetProgramInfo(cpPrograms[i], CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size,0);
  unsigned char* binary = new unsigned char [size];
  clGetProgramInfo(cpPrograms[i], CL_PROGRAM_BINARIES, sizeof(unsigned char)*size, &binary, 0);
  FILE* f = fopen(function, "w");
  fputs((const char*)binary, f);
  fclose(f);
  delete [] binary;
 //cout << binary << endl;

  functions[i] =  function;
}

OpenCLTask::OpenCLTask(size_t kernel, cl_context & context, cl_command_queue & queue, Mutex* gm, cl_program & cpProgram,
 const char* function, cl_uint dim, size_t* szLWS, size_t* szGWS){
  this->context = context;
  this->queue = queue;
  this->kernel = kernel;
  this->dim = dim;
  for ( int i = 0; i < dim; i++){
    szLocalWorkSize[i] = szLWS[i];
    szGlobalWorkSize[i] = szGWS[i];
  }
  persistent = createBuff = NULL;

  cl_int ciErrNum;
  ckKernel =  clCreateKernel(cpProgram, function, &ciErrNum);
  if (ciErrNum != CL_SUCCESS){
    Severe("Invalid kerel, errNum: %d %s ", ciErrNum, stringError(ciErrNum));
  }

  #define EVENTS 20
  writeEvents = new cl_event[EVENTS];
  readEvents = new cl_event[EVENTS/2];
  writeENum = readENum = 0;

  globalmutex = gm;
}

void OpenCLTask::InitBuffers(size_t count){
  cmBuffers = new cl_mem[count];
  width = new size_t[count];
  height = new size_t[count];
  buffCount = count;
  persistent = new bool[count];
  createBuff = new bool[count];
  sizeBuff = new size_t[count];
  for (size_t i = 0; i < count; i++) {
    persistent[i] = false;
    createBuff[i] = true;
  }
}

void OpenCLTask::CopyBuffer(size_t src, size_t dst, OpenCLTask* oclt){
  cmBuffers[dst] = oclt->cmBuffers[src];
  sizeBuff[dst] = oclt->sizeBuff[src];
  width[dst] = oclt->width[src];
  height[dst]= oclt->height[src];
  createBuff[dst] = false;

  cl_int ciErrNum = clSetKernelArg(ckKernel, dst, sizeof(cl_mem), (void*)(&(cmBuffers[dst])));
  if (ciErrNum != CL_SUCCESS)
    Severe("failed setting %i parameter, error: %i %s", dst, ciErrNum, stringError(ciErrNum));
}

void OpenCLTask::CopyBuffers(size_t srcstart, size_t srcend, size_t dststart, OpenCLTask* oclt){
  size_t j = dststart;
  cl_int ciErrNum;

  for ( size_t i = srcstart; i < srcend; i++){
    cmBuffers[j] = oclt->cmBuffers[i];
    sizeBuff[j] = oclt->sizeBuff[i];
    width[j] = oclt->width[i];
    height[j] = oclt->height[i];
    createBuff[j] = false;

    ciErrNum = clSetKernelArg(ckKernel, j, sizeof(cl_mem),  (void*)(&(cmBuffers[j])));
    if (ciErrNum != CL_SUCCESS)
      Severe("failed setting %i parameter, error: %i %s", j, ciErrNum, stringError(ciErrNum));
    j++;
  }
}

void OpenCLTask::CreateBuffers( size_t* size, cl_mem_flags* flags){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  for ( size_t it = 0; it < buffCount; it++){
    sizeBuff[it] = size[it];
    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    if ( !createBuff[it]) continue;
    #ifdef DEBUG_OUTPUT
    cout<<"clCreateBuffer " << it << endl;
    #endif
    cmBuffers[it] = clCreateBuffer(context, flags[it], size[it], NULL, &ciErrNum);
    if ( ciErrNum != CL_SUCCESS){
      for ( size_t j = 0; j < buffCount; j++)
        if ( cmBuffers[j]) clReleaseMemObject(cmBuffers[j]);
      delete [] cmBuffers;
      cmBuffers = NULL;
      Severe("clCreateBuffer failed at buffer number %d with error %d %s", it, ciErrNum, stringError(ciErrNum));
    }
    ciErrNum = clSetKernelArg(ckKernel, it, sizeof(cl_mem), (void*)&cmBuffers[it]);
    if (ciErrNum != CL_SUCCESS)
      Severe("failed setting %i parameter, error: %i %s", it, ciErrNum, stringError(ciErrNum));
  }

  return;
}

void OpenCLTask::CreateBuffer( size_t i, size_t size, cl_mem_flags flags, int argPos){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
  sizeBuff[i] = size;
  if ( !createBuff[i]) return;
  cmBuffers[i] = clCreateBuffer(context, flags, size, NULL, &ciErrNum);
  if ( ciErrNum != CL_SUCCESS){
    for ( size_t j = 0; j < buffCount; j++)
      if ( cmBuffers[j]) clReleaseMemObject(cmBuffers[j]);
    delete [] cmBuffers;
    cmBuffers = NULL;
    Severe("clCreateBuffer failed at buffer number %d with error %d %s", i, ciErrNum, stringError(ciErrNum));
  }
  if ( argPos == -1 ) argPos = i;
  ciErrNum = clSetKernelArg(ckKernel, argPos, sizeof(cl_mem), &cmBuffers[i]);
  if (ciErrNum != CL_SUCCESS)
    Severe("failed setting %i parameter, error: %i %s", i, ciErrNum, stringError(ciErrNum));

  return;
}

void OpenCLTask::CreateImage2D(size_t i, cl_mem_flags flags, const cl_image_format * imageFormat, size_t width,
  size_t height, void* host_ptr, int argPos){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  if (!createBuff[i]) return;
  this->width[i] = width;
  this->height[i] = height;
  cmBuffers[i] = clCreateImage2D(context, flags, imageFormat, width, height, 0, host_ptr, &ciErrNum);
  if (ciErrNum != CL_SUCCESS)
    Severe("failed creating %i image2D, error: %i %s", i, ciErrNum, stringError(ciErrNum));
  if ( argPos == -1 ) argPos = i;
  ciErrNum = clSetKernelArg(ckKernel, argPos, sizeof(cl_mem), (void*)&cmBuffers[i]);
  if (ciErrNum != CL_SUCCESS)
    Severe("failed setting %i parameter, error: %i %s", i, ciErrNum, stringError(ciErrNum));
}

void OpenCLTask::CopyImage2D(size_t src, size_t dst, OpenCLTask* oclt){
  cl_int ciErrNum;
  size_t origin[3];
  origin[0] = origin[1] = origin[2] = 0;
  size_t region[3];
  region[0] = oclt->width[src]; region[1] = oclt->height[src]; region[2] = 1;
  width[dst] = oclt->width[src];
  height[dst] = oclt->height[src];
  MutexLock lock(*globalmutex);
  ciErrNum = clEnqueueCopyImage(queue, oclt->cmBuffers[src], cmBuffers[dst], origin, origin, region, 0, 0, &writeEvents[writeENum++]);
  if (ciErrNum != CL_SUCCESS)
    Severe("failed copying image, error: %i %s\n", ciErrNum, stringError(ciErrNum));
}

void OpenCLTask::CreateConstantBuffer( size_t i, size_t size, void* data){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  #ifdef DEBUG_OUTPUT
  cout<<"clCreateConstantBuffer " << i << endl;
  #endif
  sizeBuff[i] = size;
  cmBuffers[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, &ciErrNum);
  if ( ciErrNum != CL_SUCCESS){
      Severe("clCreateBuffer failed at constant buffer %s", stringError(ciErrNum));
  }
  ciErrNum = 0;
  #ifdef DEBUG_OUTPUT
  cout << "setting argument " << i << endl;
  #endif
  ciErrNum = clSetKernelArg(ckKernel, i , sizeof(cl_mem), (void*)&cmBuffers[i]);
  if (ciErrNum != CL_SUCCESS)
    Severe("Failed setting %d .parameter %d %s", i, ciErrNum, stringError(ciErrNum));

  return;
}


void OpenCLTask::SetIntArgument(const size_t & it, const cl_int & arg){
  #ifdef DEBUG_OUTPUT
  cout << "set int argument " << it << endl;
  #endif

  cl_int ciErrNum;
  ciErrNum = clSetKernelArg(ckKernel, it, sizeof(cl_int), (void*)&arg);
  if (ciErrNum != CL_SUCCESS){
    Severe("Failed setting parameters %i %s\n", ciErrNum, stringError(ciErrNum));
  }
}


void OpenCLTask::SetLocalArgument(const size_t & it, const size_t & size){
  #ifdef DEBUG_OUTPUT
  cout << "set local argument " << endl;
  #endif
  cl_int ciErrNum;

  ciErrNum = clSetKernelArg(ckKernel, it, size, 0);
  if (ciErrNum != CL_SUCCESS)
    Severe("Failed setting local parameters %i %s \n", ciErrNum ,stringError(ciErrNum));

  return;
}

void OpenCLTask::EnqueueWriteBuffer(cl_mem_flags* flags,void** data){
  size_t it = 0;
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  while ( it < buffCount ){
    if ( !createBuff[it] || flags[it] == CL_MEM_WRITE_ONLY) {
      ++it;
      continue;
    }
    ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, sizeBuff[it] , data[it], 0, NULL, &writeEvents[writeENum++]);
    //probably useless test, if it is asynchronous?
    if ( ciErrNum != CL_SUCCESS)
      Severe("Failed asynchronous data transfer at %i buffer, error: %i %s",it, ciErrNum, stringError(ciErrNum));

    #ifdef GPU_TIMES
    ciErrNum = clFinish(queue);
    if ( ciErrNum != CL_SUCCESS)
      Severe("failed data transfer at %i buffer, error: %i %s",it,ciErrNum, stringError(ciErrNum));
    wmem_times += executionTime(writeEvents[writeENum - 1]);
    #endif
    ++it;
  }
  return;
}

void OpenCLTask::EnqueueWriteBuffer( size_t it, void* data, size_t size){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);

  ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, (size==0)?sizeBuff[it]:size , data, 0, NULL, &writeEvents[writeENum++]);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed asynchronous data transfer at %i buffer %i %s", it, ciErrNum, stringError(ciErrNum));

  #ifdef GPU_TIMES
  ciErrNum = clFinish(queue);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed data transfer at %i buffer, error: %i %s",it,ciErrNum, stringError(ciErrNum));
  wmem_times += executionTime(writeEvents[writeENum - 1]);
  #endif
  return;
}

void OpenCLTask::EnqueueWrite2DImage( size_t it, void* data){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  size_t origin[3];
  origin[0] = origin[1] = origin[2] = 0;
  size_t region[3];
  region[0] = width[it]; region[1] = height[it]; region[2] = 1;
  ciErrNum = clEnqueueWriteImage(queue, cmBuffers[it], CL_FALSE, origin, region,
              0, 0, data, 0, NULL, &writeEvents[writeENum++]);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed asynchronous data transfer at %i image buffer %i %s", it, ciErrNum, stringError(ciErrNum));

  #ifdef GPU_TIMES
  ciErrNum = clFinish(queue);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed data transfer at %i buffer, error: %i %s",it,ciErrNum, stringError(ciErrNum));
  wmem_times += executionTime(writeEvents[writeENum - 1]);
  #endif
}

void OpenCLTask::EnqueueReadBuffer(size_t it ,void* odata){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_TRUE, 0, sizeBuff[it] , odata, 1, &kernelEvent, &readEvents[readENum++]);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed read %i buffer %i %s", it, ciErrNum, stringError(ciErrNum));
  #ifdef GPU_TIMES
  ciErrNum = clFinish(queue);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed running kernel %d %s",ciErrNum, stringError(ciErrNum));
  rmem_times += executionTime(readEvents[readENum - 1]);
  #endif
  return;
}

void OpenCLTask::EnqueueRead2DImage( size_t it, void* data){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  size_t origin[3];
  origin[0] = origin[1] = origin[2] = 0;
  size_t region[3];
  region[0] = width[it]; region[1] = height[it]; region[2] = 1;
  ciErrNum = clEnqueueReadImage(queue, cmBuffers[it], CL_FALSE, origin, region,
              0, 0, data, 1, &kernelEvent, &readEvents[readENum++]);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed asynchronous data transfer at %i image buffer %i %s", it, ciErrNum, stringError(ciErrNum));

  #ifdef GPU_TIMES
  ciErrNum = clFinish(queue);
  if ( ciErrNum != CL_SUCCESS)
    Severe("failed data transfer at %i buffer, error: %i %s",it,ciErrNum, stringError(ciErrNum));
  rmem_times += executionTime(readEvents[readENum - 1]);
  #endif
}

double executionTime(cl_event &event)
{
    cl_ulong start, end;
    cl_int err;
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    if ( err != CL_SUCCESS)
      Severe("fialed getting profiling end time, error: %i %s\n", err, stringError(err));
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    if ( err != CL_SUCCESS)
      Severe("fialed getting profiling end time, error: %i %s\n", err, stringError(err));

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

void OpenCLTask::Run(){
    #ifdef DEBUG_OUTPUT
    cout << "clEnqueueNDRangeKernel...\n";
    #endif
    Info("Waiting on %d write events",writeENum);
    MutexLock lock(*globalmutex);
    cl_int ciErrNum;

    //cout << "Running kernel " << kernel << endl;
    if ( szLocalWorkSize == 0) //let OpenCL implementation to choose local work size
    ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, dim, NULL, szGlobalWorkSize, NULL, writeENum, (writeENum == 0)?0:writeEvents,  &kernelEvent);
    else
    ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, dim, NULL, szGlobalWorkSize, szLocalWorkSize, writeENum, (writeENum == 0)?0:writeEvents,  &kernelEvent);
    //OpenCL implementace vybere velikost bloku
    //ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, NULL, 0, NULL,  NULL);
    if ( ciErrNum != CL_SUCCESS)
      Severe("failed enqueue kernel %d %s", ciErrNum, stringError(ciErrNum));

    for ( size_t i = 0; i < writeENum; i++)
      clReleaseEvent(writeEvents[i]);
    writeENum = 0;

    #ifdef GPU_TIMES
    ciErrNum = clFinish(queue);
    if ( ciErrNum != CL_SUCCESS)
      Severe("failed running kernel %d %s",ciErrNum, stringError(ciErrNum));
    double temp = executionTime(kernelEvent);
    if ( temp < 10000)
      kernel_times[kernel] += temp;
    else
      Warning("Failed to get proper kernel time %f\n", temp);
    ++kernel_calls[kernel];
    #endif

    return;
}

void OpenCLTask::EnqueueReadBuffer(cl_mem_flags* flags,void** data){
  cl_int ciErrNum;
  size_t it = 0;
  MutexLock lock(*globalmutex);
  while ( it < buffCount){
    if ( flags[it] == CL_MEM_READ_ONLY){
      it++;
      continue;
    }
    ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_FALSE, 0, sizeBuff[it], data[it], 1, &kernelEvent,  &readEvents[readENum++]);
    if ( ciErrNum != CL_SUCCESS)
      Severe("failed read %i buffer %i %s", it, ciErrNum, stringError(ciErrNum));
    it++;

    #ifdef GPU_TIMES
    ciErrNum = clFinish(queue);
    if ( ciErrNum != CL_SUCCESS)
      Severe("failed running kernel %d %s",ciErrNum, stringError(ciErrNum));
    rmem_times += executionTime(readEvents[readENum - 1]);
    #endif
  }
  return ;
}

OpenCLTask::~OpenCLTask(){
  clReleaseKernel(ckKernel);
  if (cmBuffers){
    for ( size_t i = 0; i < buffCount; i++){
       clReleaseMemObject(cmBuffers[i]);
    }
    delete [] width;
    delete [] height;
    if (cmBuffers) delete [] cmBuffers;
    if (createBuff) delete [] createBuff;
    if (persistent) delete [] persistent;
    cmBuffers = NULL;
  }
  for ( size_t i = 0; i < writeENum; i++)
    clReleaseEvent(writeEvents[i]);
  delete [] writeEvents;
  for ( size_t i = 0; i < readENum; i++)
    clReleaseEvent(readEvents[i]);
  delete [] readEvents;
  delete [] sizeBuff;
  clReleaseEvent(kernelEvent);
}

const char* stringError(cl_int errNum){
  switch (errNum) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  }
  return "";
}

