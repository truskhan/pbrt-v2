#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
/**
 * @file GPUparallel.h
 * @author: Hana Truskova hana.truskova@seznam.cz
 * @description: Auxiliary classes to make OpenCL API more comfortable
**/

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
    /**
      Function that sets integer kernel argument and checks for error.
      @param[in] it kernel parameter position
      @param[in] arg kernel parameter
    **/
    void SetIntArgument(const size_t & it, const cl_int & arg);
    /**
      Function that sets local array parameter.
      @param[in] it kernel parameter position
      @param[in] size kernel parameter size in bytes
    **/
    void SetLocalArgument(const size_t & it, const size_t & size);
    /**
      Enqueues data transfer from host to OpenCL device.
      @param[in] it buffer's postion in buffer array, the buffer to trasnfer
      @param[in] data host pointer to the data to transfer
      @param[in] size optional parameter for specifying other kernel position than buffer's position
                 in the buffer array
    **/
    void EnqueueWriteBuffer(size_t it, void* data, size_t size = 0);
    /**
      Enqueus data transfer from the OpenCL device image to the host memory
      @param[in] it image's position in buffer array
      @param[in] data pointer to host array where to store the data
    **/
    void EnqueueRead2DImage( size_t it, void* data);
    /**
      Enqueues data transfer from the host memory to the OpenCL device.
      @param[in] it image's position in buffer array and in kernel arguments
      @param[in] data pointer to the host array, where are the data to be copyed
    **/
    void EnqueueWrite2DImage( size_t it, void* data);
    /**
      Enqueues image copy on OpenCL device without the need to transfer image back to the host.
      @param[in] src source image's position in buffer array
      @param[in] dst destination image's position in buffer array
      @param[in] oclt OpenCL Task which holds source image
    **/
    void CopyImage2D(size_t src, size_t dst, OpenCLTask* oclt);
    /**
      Enqueues data transfer from the OpenCL device buffer to the host memory.
      @param[in] it buffer's position in buffer array
      @param[in] odata data pointer to host array where to store read data
    **/
    void EnqueueReadBuffer( size_t it, void* odata);
    /**
      Enqueues kernel execution on OpenCL device.
      @return returns kernel execution time in seconds if the code is compiled with GPU_TIMES
    **/
    #ifdef GPU_TIMES
    double
    #else
    void
    #endif
    Run();
    /**
      Waits till the enqueued kernel is finished.
    **/
    void WaitForKernel(){
      clWaitForEvents(1, &kernelEvent);
    }
    /**
      Waits till all the enqeued data transfers from the OpenCL device to the host memory are finished.
    **/
    void WaitForRead(){
      clWaitForEvents(readENum, readEvents);
    }
};

/** Class for holding OpenCL queue and auxiliary variables**/
class OpenCLQueue {
  ///OpenCL command queue
  cl_command_queue cmd_queue;
  ///OpenCLTasks in command queue
  OpenCLTask** tasks;
  ///auxiliary variables for resizing and monitoring tasks array size
  size_t numtasks;
  size_t maxtasks;
  ///mutex to avoid race condition when modifying class properties (deleting and adding tasks)
  Mutex* globalmutex;
  public:
  ///OpenCL device to querry clGetDeviceInfo and find out detail information about binded OpenCL device
  cl_device_id device;
  /**
    OpenCL queue constructor for creating OpenCL queue and initializing the mutex
    @param[in] context context is needed for all OpenCL API calls
  **/
  OpenCLQueue( cl_context & context);
  /**
    OpenCL queue destructor for deleting remaining OpenCL tasks, releasing OpenCL queue and destroying mutex
  **/
    ~OpenCLQueue();
  /**
    Creation of OpenCL Task
      @param[in] kernel index into kernel's array
      @param[in] context OpenCL context needed for OpenCL API calls
      @param[in] cpProgram OpenCL program to kreate kernel
      @param[in] queue OpenCL queue needed for OpenCL API calls
      @param[in] function which function from cpProgram should be used for kernel
      @param[in] dim OpenCL kernel dimension
      @param[in] szLWS local work-group size
      @param[in] szGWS global work size
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
  /**
    Returns pointer to OpenCL task to querry it's functions directly.
    @param[in] i OpenCL task's position in the queue
  **/
  OpenCLTask* getTask(size_t i = 0){
     return tasks[i];
  }
  /**
    Deletes OpenCL task.
    @[param[in] i OpenCL task's position in the queue
  **/
  void delTask(size_t i = 0){
    MutexLock lock(*globalmutex);
    if ( tasks[i] != NULL){
      delete tasks[i];
      tasks[i] = NULL;
    }
    while ( numtasks > 0 && tasks[numtasks-1] == NULL)
      --numtasks;
  }
  /**
    Ensures that all queued commands submitted to the queue are finished.
  **/
  void Finish(){
     clFinish(cmd_queue);
  }
};

/**Class for OpenCL initialization and context creation, should be only one instance at time in the program**/
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
    /**
      Querrys binded OpenCL device for it's capable maximum work group size.
      @param[in] q selected queue to query
      @return maximum work group size of binded OpenCL device
    **/
    size_t getMaxWorkGroupSize(size_t q = 0) {
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's capable maximum allocation memory size at once
      @param[in] q selected queue to query
      @return maximum allocation memory size at once in B
    **/
    cl_ulong getMaxMemAllocSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's capable maximum 2D image width.
      @param[in] q selected queue to query
      @return maximum 2D image width
    **/
    size_t getMaxImage2DWidth(size_t q = 0){
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &result,0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's capable maximum 2D image height.
      @param[in] q selected queue to query
      @return maximum 2D image height
    **/
    size_t getMaxImage2DHeight(size_t q = 0){
      size_t result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &result,0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's constant buffer size.
      @param[in] q selected queue to query
      @return maximum constant buffer size
    **/
    cl_ulong getMaxConstantBufferSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's local buffer size.
      @param[in] q selected queue to query
      @return maximum constant buffer size
    **/
    cl_ulong getLocalMemSize(size_t q = 0) {
      cl_ulong result;
      cl_int err;
      err = clGetDeviceInfo(queue[q]->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &result, 0);
      if ( err != CL_SUCCESS)
        Severe("Failed to run clGetDeviceInfo %s %i", stringError(err), err);
      return result;
    }
    /**
      Querrys binded OpenCL device for it's global memory size.
      @param[in] q selected queue to query
      @return global memory size
    **/
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
    @param[in] numKernels number of different kernels that wil be executed
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
    @param[in] i index to queue array, identifies which queue should be deleted
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
    /**
      Prints accumulated kernel runs for profilying purposes.
    **/
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
