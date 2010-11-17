
#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
#define __CL_ENABLE_EXCEPTIONS
/**Auxiliary classes to make OpenCL API more comfortable
@todo enable sharing cl_mem buffer among command queues (for vertices...)
**/
// core/GPUparallel.h*
#include <CL/cl.h>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <string.h>
#include "core/parallel.h"
#include "core/error.h"
#include "cl/oclUtils.h"
#include "cl/shrUtils.h"

#ifdef GPU_TIMES
extern double* kernel_times;
extern unsigned int* kernel_calls;
extern double wmem_times;
extern double rmem_times;
#endif

const char* stringError(cl_int errNum);
double executionTime(cl_event &event);

/**Class for holding OpenCL kernel and auxiliary variables**/
class OpenCLTask {
    size_t kernel;
    cl_uint dim;
    /// # of work-itmes in 1D work group
    size_t szLocalWorkSize[3];
    /// Total # of work items in the 1D range
    size_t szGlobalWorkSize[3];
    /// OpenCL kernel
    cl_kernel ckKernel;
    /// array of memory buffers
    cl_mem* cmBuffers;
    size_t* width;
    size_t* height;
    /// total number of buffer count
    size_t buffCount;
    /// indicates which memory buffer should stay in the memory after this task is finished
    bool* persistent;
    /// indicates which memory buffer should be created and which should be copied
    bool* createBuff;
    /// size of buffers
    size_t *sizeBuff;
    /// reference to the OpenCL context
    cl_context context;
    /// reference to the OpenCL command queue
    cl_command_queue queue;
    cl_uint writeENum;
    cl_uint readENum;
    cl_event* writeEvents;
    cl_event kernelEvent;
    cl_event* readEvents;
    Mutex* globalmutex;
  public:
    OpenCLTask(size_t kernel,cl_context & context, cl_command_queue & queue, Mutex* gm, cl_program & cpProgram,
    const char* function, cl_uint dim, size_t* szLWS, size_t* szGWS);
    ~OpenCLTask();
    void InitBuffers(size_t count);
    bool CreateConstantBuffer( size_t i, size_t size, void* data);
    /**
    Same as other CreateBuffer functions
    \sa CreateBuffers"(size_t, cl_mem_flags*)"
    \sa CreateBuffers"("")"
    **/
    bool CreateBuffer( size_t i, size_t size, cl_mem_flags flags, int argPos = -1);
    bool CreateBuffers(size_t* size, cl_mem_flags* flags);
    void CreateImage2D(size_t i, cl_mem_flags flags, const cl_image_format * imageFormat,
                    size_t width, size_t height, void* host_ptr, int argPos = -1);
    void CopyBuffer(size_t src, size_t dst, OpenCLTask* oclt);
    void CopyBuffers(size_t srcstart, size_t srcend, size_t dststart, OpenCLTask* oclt);
    int SetPersistentBuff( size_t i ) { return clRetainMemObject(cmBuffers[i]);}
    int SetPersistentBuffers( size_t start, size_t end){
      int result = CL_SUCCESS;
      for ( size_t i = start; i < end; i++){
        result |= clRetainMemObject(cmBuffers[i]);
      }
      return result;
    }
    void SetIntArgument(const size_t & it, const cl_int & arg);
    bool SetLocalArgument(const size_t & it, const size_t & size);
    bool EnqueueWriteBuffer(cl_mem_flags* flags,void** data);
    bool EnqueueWriteBuffer(size_t it, void* data, size_t size = 0);
    void EnqueueRead2DImage( size_t it, void* data);
    void EnqueueWrite2DImage( size_t it, void* data);
    void CopyImage2D(size_t src, size_t dst, OpenCLTask* oclt);
    bool EnqueueReadBuffer(cl_mem_flags* flags,void** odata);
    bool EnqueueReadBuffer( size_t it, void* odata);
    bool Run();
    void WaitForKernel(){
      clWaitForEvents(1, &kernelEvent);
    }
    void WaitForRead(){
      clWaitForEvents(readENum, readEvents);
    }
};

/** Class for holding OpenCL queue **/
class OpenCLQueue {
   ///OpenCL command queue
   cl_command_queue cmd_queue;
   ///OpenCLTasks in command queue
   OpenCLTask** tasks;
   ///auxiliary variables for resizing and monitoring tasks array size
   size_t numtasks;
   size_t maxtasks;
   Mutex* globalmutex;
   public:
   cl_device_id device;
   /**Creation of OpenCL queue
   param[in] context needs for all OpenCL API calls
   **/
    OpenCLQueue( cl_context & context);
    ~OpenCLQueue();
    /**Creation of OpenCL Task
    \see OpenCL:CreateTask() for detail description of parameters
    **/
    size_t CreateTask(size_t kernel, cl_context & context, cl_program & program, const char* func, cl_uint dim, size_t* szLWS, size_t* szGWS){
      MutexLock lock(*globalmutex);
      if ( numtasks == maxtasks) {
        OpenCLTask** temp = new OpenCLTask*[2*maxtasks];
        for ( unsigned int i = 0; i < maxtasks; i++)
            temp[i] = tasks[i];
        for ( unsigned int i = maxtasks; i < 2*maxtasks; i++)
            temp[i] = 0;
       delete [] tasks;
       tasks = temp;
       maxtasks *= 2;
      }
      tasks[numtasks++] = (new OpenCLTask(kernel, context, cmd_queue, globalmutex, program, func , dim, szLWS, szGWS));
      return numtasks-1;
    }
    OpenCLTask* getTask(size_t i = 0){
       return tasks[i];
    }
    void delTask(size_t i = 0){
      MutexLock lock(*globalmutex);
      if ( tasks[i] != NULL){
        delete tasks[i];
        tasks[i] = NULL;
      }
      while ( numtasks > 0 && tasks[numtasks-1] == NULL)
        --numtasks;
    }
    void Finish(){
       clFinish(cmd_queue);
    }
};

/**Class for OpenCL initialization and context creation, should be only one instance at the time in the program**/
class OpenCL {
  ///OpenCL context
  static cl_context cxContext;
  ///Holds pointers to created queues
  OpenCLQueue** queue;
  ///Number of different kernels user won't to compile
  size_t numKernels;
  ///Number of different command queueus
  size_t numqueues;
  ///Auxiliary variable - just for controlling pointer to queues' array size
  size_t maxqueues;
  ///OpenCL functions names to compile
  const char** functions;
  ///OpenCL programs compiled
  cl_program* cpPrograms;
  ///Mutex for thread safe call to CreateQueues etc.
  Mutex* mutex;
  public:
    size_t getMaxWorkGroupSize(size_t q = 0) {
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    cl_ulong getMaxMemAllocSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    cl_ulong getMaxConstantBufferSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    cl_ulong getLocalMemSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    cl_ulong getGlobalMemSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**Initialize OpenCL
    @param[in] onGPU run on GPU with NVIDIA or on CPU with ATI-stream SDK
    @param[in] numKernels
    **/
    OpenCL(bool onGPU, size_t numKernels);
    /**
    Release consumpted sources (clPrograms, queues, context)
    **/
    ~OpenCL();
    /** Compiles program from give file and function name, if it is unsuccesfull, it aborts the entire program
    @param[in] file name of the file with OpenCL source code
    @param[in] function name of the desired OpenCL function to compile and run
    @param[in] program name of the compiled OpenCL program with .ptx extension
    @param[in] i index to the functions and cpPrograms arrays where to store information about compiled program and its name
    **/
    void CompileProgram(const char* cPathAndName, const char* function,
      const char* program, size_t i);
    /**
    Creates new command queue
    @return returns index to queues array, so that user can query for concrete queue
    **/
    size_t CreateCmdQueue(){

        if ( numqueues == maxqueues) {
            OpenCLQueue** temp = new OpenCLQueue*[2/3*maxqueues];
            for ( unsigned int i = 0; i < maxqueues; i++)
                temp[i] = queue[i];
            for ( unsigned int i = maxqueues; i < 2/3*maxqueues; i++)
                temp[i] = 0;
            delete [] queue;
            queue = temp;
            maxqueues *= 2/3;
        }
        queue[numqueues++] = new OpenCLQueue(cxContext);
        Info("Created Cmd Queue %d.", numqueues-1);
        return numqueues-1;
    }
    /**
    Deletes command queue
    **/
    void DeleteCmdQueue(size_t i){

        Info("Started Deletion of cmd queue %d.", i);
        if ( queue[i] != NULL){
            delete queue[i];
            queue[i] = NULL;
        }
        while ( numqueues > 0 && queue[numqueues-1] == NULL)
        --numqueues;

        Info("Deleted Cmd Queue %d.", i);
    }
    /**
    Creates new OpenCL Task which is simply a one kernel
    @param[in] kernel index to cpPrograms and functions which kernel to make
    @param[in] count total number of tasks
    @param[in] dim dimension of NDrange
    @param[in] i index to command queues
    @param[in] szLWS number of work-itmes in a block
    @param[in] szGWS global number of work-items, should be multiple of szLWS
    **/
    size_t CreateTask(size_t kernel, size_t dim, size_t* szGWS = 0, size_t* szLWS = 0, size_t i = 0){
        for ( int i = 0; i < dim; i++){
          szGWS[i] = shrRoundUp((int)szLWS[i], szGWS[i]);
        }
        size_t task = queue[i]->CreateTask(kernel, cxContext, cpPrograms[kernel], functions[kernel], dim, szLWS, szGWS);
        Info("Created Task %d in queue %d.",task, i);
        return task;
    }
    /** Getter for OpenCLTask
    @param[in] task task index in array
    @param[in] i queue where the task is
    @return returns OpenCLTask so that user can directly call ist methods
    **/
    OpenCLTask* getTask(size_t task = 0, size_t i = 0){
      return queue[i]->getTask(task);
    }
    /** Delete OpenCLTask
    @param[in] task task index in array
    @param[in] i queue where the task is
    **/
    void delTask(size_t task = 0, size_t i = 0){
      queue[i]->delTask(task);
      Info("Deleted %d Task in queue %d.",task,i);
    }
    /**
     Finish all commands queued in a queue
     @param[in] i queue to flush
    **/
    void Finish(size_t i = 0){
      queue[i]->Finish();
    }
    #ifdef GPU_TIMES
    void PrintTimes(){
      printf("Measured times in seconds\n");
      for ( size_t i = 0; i < numKernels; i++)
        printf("total %i kernel (%i run) time:\t %f\n", i, kernel_calls[i], kernel_times[i]);
      printf("total wmem transfer time:\t %f\n", wmem_times);
      printf("total rmem transfer time:\t %f\n", rmem_times);
    }
    #endif
};




#endif // PBRT_CORE_PARALLEL_H
