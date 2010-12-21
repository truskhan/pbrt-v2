/**
 * @file GPUparallel.h
 * @author: Hana Truskova hana.truskova@seznam.cz
 * @description: Auxiliary classes to make OpenCL API more comfortable
**/
#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <string.h>
#include <CL/cl.h>
#include "core/parallel.h"
#include "core/error.h"

#ifdef GPU_TIMES
/** total kernel run times array **/
extern double* kernel_times;
/** number of kernel calls' array **/
extern unsigned int* kernel_calls;
/** total time spent by transmitting data to OpenCL device **/
extern double wmem_times;
/** total time spent by reading data from the OpenCL device **/
extern double rmem_times;
#endif

/**
 * Auxiliary function for translating integer error codes to strings
 * @param[in] errNum error code
 * @return string error description
**/
const char* stringError(cl_int errNum);
/**
 * Auxiliary function for getting kernel run time
 * @param[in] event the event to get kernel start and end time via clGetEventProfilingInfo
 * @return kernel run time in seconds
**/
double executionTime(cl_event &event);

/**
 * Auxiliary function to round up global thread size so that it is divisable by block size
 * @param[in] group_size local block size
 * @param[in] global_size global thread size
 * @return new global size evenly divisable by local block size
**/
size_t RoundUp(int group_size, int global_size);

/**Class for holding OpenCL kernel and auxiliary variables**/
class OpenCLTask {
    /// index into kernel's array
    size_t kernel;
    /// dimension of kernel
    cl_uint dim;
    /// # of work-itmes in 1D work group
    size_t szLocalWorkSize[3];
    /// Total # of work items in the 1D range
    size_t szGlobalWorkSize[3];
    /// OpenCL kernel
    cl_kernel ckKernel;
    /// array of memory buffers
    cl_mem* cmBuffers;
    /// width array of OpenCL images
    size_t* width;
    /// hieght array of OpenCL images
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
    /// number of write events to wait on
    cl_uint writeENum;
    /// nubmer of read events to wait on
    cl_uint readENum;
    /// pointer to write events to wait on
    cl_event* writeEvents;
    /// kernel event to wait on
    cl_event kernelEvent;
    /// pointer to read events to wait on
    cl_event* readEvents;
    /// mutex to avoid race condition when modifying buffers
    Mutex* globalmutex;
  public:
    /** OpenCL Task constructor for initializing and allocating necessary arrays
      @param[in] kernel index into kernel's array
      @param[in] context OpenCL context needed for OpenCL API calls
      @param[in] queue OpenCL queue needed for OpenCL API calls
      @param[in] gm mutex to avoid race condition on shared variables
      @param[in] cpProgram OpenCL program to kreate kernel
      @param[in] function which function from cpProgram should be used for kernel
      @param[in] dim OpenCL kernel dimension
      @param[in] szLWS local work-group size
      @param[in] szGWS global work size
    **/
    OpenCLTask(size_t kernel,cl_context & context, cl_command_queue & queue, Mutex* gm, cl_program & cpProgram,
    const char* function, cl_uint dim, size_t* szLWS, size_t* szGWS);
    /** OpenCL Task destructor for deallocating arrays, releasing OpenCL events and buffers
    **/
    ~OpenCLTask();
    /** Auxiliary function for allocating and initializing arrays
      @param[in] count number of needed buffers
    **/
    void InitBuffers(size_t count);
    /** Auxiliary function for creating constant OpenCL buffers and setting it as kernel argument.
      @param[in] i buffer position in kernel arguments and buffer array
      @param[in] size buffer size in bytes
      @param[in] data pointer to buffer data
    **/
    void CreateConstantBuffer( size_t i, size_t size, void* data);
    /**
      Creates one OpenCL buffer and sets it as kernel argument.
      @param[in] i buffer position in buffer array
      @param[in] size buffer size in bytes
      @param[in] flags OpenCL flags (CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE)
      @param[in] argPos buffer position in kernel arguments if it is different than its
        position in buffer array
    **/
    void CreateBuffer( size_t i, size_t size, cl_mem_flags flags, int argPos = -1);
    /**
      Creates one OpenCL image and sets it as kernel argument.
      @param[in] i buffer position in buffer array
      @param[in] flags OpenCL flags (CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY, CL_MEM_READ_WRITE)
      @param[in] imageFormat OpenCL image format
      @param[in] width desired image width
      @param[in] height desired image height
      @param[in] host_ptr image data
      @param[in] argPos buffer position in kernel arguments if it is different than its
        position in buffer array
    **/
    void CreateImage2D(size_t i, cl_mem_flags flags, const cl_image_format * imageFormat,
                    size_t width, size_t height, void* host_ptr, int argPos = -1);
    /**
      Copys existing OpenCL buffer and sets it as kernel argument.
      @param[in] src source buffer's index to buffer array
      @param[in] dst destination buffer's index to buffer array
      @param[in] oclt OpenCL Task which holds source buffer
    **/
    void CopyBuffer(size_t src, size_t dst, OpenCLTask* oclt);
    /**
      Copys multiple existing OpenCL buffers and sets them as kernel arguments.
      @param[in] srcstart source buffers' start index to buffer array
      @param[in] srcend source buffers' end index to buffer array
      @param[in] dststart destination buffers' index to buffer array
      @param[in] oclt OpenCL Task which holds source buffers
    **/
    void CopyBuffers(size_t srcstart, size_t srcend, size_t dststart, OpenCLTask* oclt);
    /**
      Function for increasing buffer reference count, so that those buffers won't be
      destroyed after the OpenCL Task destructor is called.
      @param[in] i buffer's index to buffer array, which should retain in memory
    **/
    void SetPersistentBuff( size_t i ) {
      cl_int err = clRetainMemObject(cmBuffers[i]);
      if ( err !=  CL_SUCCESS)
        Severe("Failed at clRetainMemObject at buffer %i, err %i %s", i, err, stringError(err));
    }
    /**
      Function for increasing buffers reference's counts, so that those buffers won't be
      destroyed after the OpenCL Task destructor is called.
      @param[in] start buffer's index to buffer array, which should retain in memory
    **/
    void SetPersistentBuffers( size_t start, size_t end){
      cl_int err = CL_SUCCESS;
      for ( size_t i = start; i < end; i++){
        err = clRetainMemObject(cmBuffers[i]);
        if ( err !=  CL_SUCCESS)
          Severe("Failed at clRetainMemObject at buffer %i, err %i %s", i, err, stringError(err));
      }
    }
    void SetIntArgument(const size_t & it, const cl_int & arg);
    void SetLocalArgument(const size_t & it, const size_t & size);
    void EnqueueWriteBuffer(cl_mem_flags* flags,void** data);
    void EnqueueWriteBuffer(size_t it, void* data, size_t size = 0);
    void EnqueueRead2DImage( size_t it, void* data);
    void EnqueueWrite2DImage( size_t it, void* data);
    void CopyImage2D(size_t src, size_t dst, OpenCLTask* oclt);
    void EnqueueReadBuffer(cl_mem_flags* flags,void** odata);
    void EnqueueReadBuffer( size_t it, void* odata);
    #ifdef GPU_TIMES
    double
    #else
    void
    #endif
    Run();
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
    size_t getMaxImage2DWidth(size_t q = 0){
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &result,0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    size_t getMaxImage2DHeight(size_t q = 0){
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &result,0);
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
      const char* program, size_t i, char** headers = NULL, const int headerNum = 0);
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
        for ( int j = 0; j < dim; j++){
          szGWS[j] = RoundUp((int)szLWS[j], szGWS[j]);
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
