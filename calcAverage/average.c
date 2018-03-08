#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "average.cl"
#define KERNEL_FUNC "sum_float"

#define NUM_VEC 1024

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

int main()
{

    /* Host/device data structures */
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_int i, err;

    /*create timer variables */
    clock_t start_cpu, start_gpu, end_cpu, end_gpu;
    double cpu_time, gpu_time;

    /* Program/kernel data structures */
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    cl_kernel kernel;

    /* Data and buffers */
    float vec[NUM_VEC];
    float result[1];
    float correct[NUM_VEC];
    for (i = 0; i < NUM_VEC; i++)
    {
        correct[i] = 0.0f;
    }
    cl_mem vec_buff, res_buff;
    size_t work_units_per_kernel;

    /* Initialize the vector */
    for (i = 0; i < NUM_VEC; i++)
    {
        vec[i] = i * 1.0f;
    }
    /* check the CPU compute timings */
    int j = 0;
    float cpu_sum = 0.0;
    float cpu_avg = 0.0;
    start_cpu = clock();
    for (i = 0; i < NUM_VEC; i++)
    {
        cpu_sum += vec[i];
    }
    cpu_avg = cpu_sum / NUM_VEC;
    // printf("average by cpu %f\n",cpu_avg);

    end_cpu = clock();
    cpu_time = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0)
    {
        perror("Couldn't find any platforms");
        exit(1);
    }

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0)
    {
        perror("Couldn't find any devices");
        exit(1);
    }

    /* Create the context */
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    /* Read program file and place content into buffer */
    program_handle = fopen(PROGRAM_FILE, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(context, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    /* Create kernel for the mat_vec_mult function */
    kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    if (err < 0)
    {
        perror("Couldn't create the kernel");
        exit(1);
    }

    /* Create CL buffers to hold input and output data */
    vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * NUM_VEC, vec, NULL);
    if (err < 0)
    {
        perror("Couldn't create a buffer object");
        exit(1);
    }

    res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);

    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &vec_buff);
    if (err < 0)
    {
        perror("Couldn't set the kernel argument");
        exit(1);
    }
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &res_buff);

    /* Create a CL command queue for the device*/
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create the command queue");
        exit(1);
    }

    /* Enqueue the command queue to the device */
    work_units_per_kernel = 16;
    /*start the timer */
    start_gpu = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
                                 NULL, 0, NULL, NULL);
    if (err < 0)
    {
        perror("Couldn't enqueue the kernel execution command");
        exit(1);
    }
    end_gpu = clock();
    gpu_time = ((double)(end_gpu - start_gpu)) / CLOCKS_PER_SEC;

    /* Read the result */
    err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float),
                              result, 0, NULL, NULL);
    if (err < 0)
    {
        perror("Couldn't enqueue the read buffer command");
        exit(1);
    }

    /* Test the result */
    if (result[0] == cpu_avg)
    {
        
        printf("avg:%f\tcpu_time:%f\tgpu_time:%f\n", result[0], cpu_time, gpu_time);
    }
    else
    {
        printf("average calculation unsuccessful.\n");
    }

    // printf("%f\t%f\n", cpu_time, gpu_time);

    /* Deallocate resources */

    clReleaseMemObject(vec_buff);
    clReleaseMemObject(res_buff);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}
