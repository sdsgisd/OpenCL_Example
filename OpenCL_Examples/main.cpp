//
//  main.cpp
//
//  Created by Sadashige Ishida on 12/16/17.
//
//  Reference (in Japanese): http://peta.okechan.net/blog/archives/2538

#include <iostream>
#include <vector>
#include <functional>
#include <OpenCL/opencl.h>

#define PLATFORM_MAX 4
#define DEVICE_MAX 4

#include <chrono>

template <typename Func, typename ...Args>
void measure_runtime(Func &target_func,const unsigned iteration,Args...args ){
    
    
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();
    
    for(int i=0;i<iteration;++i){
        target_func(args...);
    }
    
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //Convert time to ms.
    
    std::cout<<elapsed<<"ms"<<std::endl;
    
};


void EC(cl_int result, const char *title)
{
    if (result != CL_SUCCESS) {
        std::cout << "Error: " << title << "(" << result << ")\n";
    }
}


cl_int err = CL_SUCCESS;
void EC2(const char *title)
{
    if (err != CL_SUCCESS) {
        std::cout << "Error: " << title << "(" << err << ")\n";
    }
    err = CL_SUCCESS;
}

void prepare_arrays(float * array1, float *array2,float*output,unsigned num_elements){
    
    for (int i = 0; i < num_elements; i++) {
        array1[i] = float(i);
    }
    
    for (int i = 0; i < num_elements; i++) {
        array2[i] = float(i);
    }
    
    for (int i = 0; i < num_elements; i++) {
        output[i] =0;
    }
}

void add_matrix_cpu(float * input1, float *input2,float*output,unsigned num_elements){
    
    int times=1;
    for(int j=0;j<times;++j){
        for(int j=0;j<num_elements;++j){
            output[j] += input1[j]+input2[j];
        }
    }
    
}

void add_array_cpu(float * array1, float *array2,float*result_array_normal,unsigned n){
    
    measure_runtime(add_matrix_cpu,1, array1,array2,result_array_normal,n);
    
    const bool display_computational_results=false;
    if(display_computational_results){
        for (int i = 0; i < n; i++) {
            std::cout << result_array_normal[i] << ", ";
        }
        std::cout << "\n";
        
    }
    
}

int add_array_opencl(float * array1, float *array2,float*result_array_cl,unsigned n){
    // Obtain platform.
    cl_platform_id platforms[PLATFORM_MAX];
    cl_uint platformCount;
    EC(clGetPlatformIDs(PLATFORM_MAX, platforms, &platformCount), "clGetPlatformIDs");
    if (platformCount == 0) {
        std::cerr << "No platform.\n";
        return EXIT_FAILURE;
    }
    
    // Display platform information
    for (int i = 0; i < platformCount; i++) {
        char vendor[100] = {0};
        char version[100] = {0};
        EC(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr), "clGetPlatformInfo");
        EC(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr), "clGetPlatformInfo");
        std::cout << "Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << "\n";
    }
    
    // Obtain devices
    cl_device_id devices[DEVICE_MAX];
    cl_uint deviceCount;
    EC(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, DEVICE_MAX, devices, &deviceCount), "clGetDeviceIDs");
    if (deviceCount == 0) {
        std::cerr << "No device.\n";
        return EXIT_FAILURE;
    }
    
    // Display devices information
    std::cout << deviceCount << " device(s) found.\n";
    for (int i = 0; i < deviceCount; i++) {
        char name[100] = {0};
        size_t len;
        EC(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, &len), "clGetDeviceInfo");
        std::cout << "Device id: " << i << ", Name: " << name << "\n";
    }
    
    // Make context
    cl_context ctx = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
    EC2("clCreateContext");
    
    // Read precompiled opencl source
    const char* bitcode_path = "OpenCL/kernel.cl.gpu_64.bc";
    
    size_t len = strlen(bitcode_path);
    cl_program program = clCreateProgramWithBinary(ctx, 1, devices, &len, (const unsigned char**)&bitcode_path, nullptr, &err);
    EC2("clCreateProgramWithBinary");
    
    // build
    EC(clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr), "clBuildProgram");
    
    // make kernel
    cl_kernel kernel = clCreateKernel(program, "add_matrix_cl", &err);
    EC2("clCreateKernel");
    
    // allocate device memory
    cl_mem device_mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, array1, &err);
    
    cl_mem device_mem2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, array2, &err);
    
    cl_mem device_mem_result = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, result_array_cl, &err);
    EC2("clCreateBuffer");
    
    // set parameters to kernels
    EC(clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_mem1), "clSetKernelArg");
    
    EC(clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_mem2), "clSetKernelArg");
    
    EC(clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_mem_result), "clSetKernelArg");
    
    
    EC(clSetKernelArg(kernel, 3, sizeof(int), &n), "clSetKernelArg");
    
    // make command queue
    cl_command_queue q = clCreateCommandQueue(ctx, devices[0], 0, &err);
    EC2("clCreateCommandQueue");
    
    auto exec_kernel=[&](){
        // execute kernels
        size_t global = n;
        EC(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        
        // Read the results
        EC(clEnqueueReadBuffer(q, device_mem_result, CL_TRUE, 0, sizeof(float) * n, result_array_cl, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        
        
    };
    
    
    measure_runtime(exec_kernel,1);
    

    
    const bool display_computational_results=false;
    if(display_computational_results){
        for (int i = 0; i < n; i++) {
            std::cout << result_array_cl[i] << ", ";
        }
        std::cout << "\n";
        
    }
    
    // Release command queue
    EC(clReleaseCommandQueue(q), "clReleaseCommandQueue");
    
    // Release memory of the devices
    EC(clReleaseMemObject(device_mem1), "clReleaseMemObject");
    
    // Release kernel
    EC(clReleaseKernel(kernel), "clReleaseKernel");
    
    // Release program
    EC(clReleaseProgram(program), "clReleaseProgram");
    
    // Release context
    EC(clReleaseContext(ctx), "clReleaseContext");
    
    std::cout << "Done.\n";
    return EXIT_SUCCESS;
}


int main(int argc, const char * argv[])
{
    // Prepare data
    int n = 100000000;
    float *array1,*array2,*result_array_cl,*result_array_normal;
    array1=new float[n];
    array2=new float[n];
    result_array_cl=new float[n];
    result_array_normal=new float[n];
    
    prepare_arrays(array1,array2,result_array_normal,n);
    prepare_arrays(array1,array2,result_array_cl,n);
    
    //Compute two ways, straightforward v.s. opencl.
    std::cout<<"Straightforwad computation:";
    add_array_cpu(array1,array2,result_array_normal,n);
    std::cout<<std::endl;
    std::cout<<"Via OpenCL:"<<std::endl;
    add_array_opencl(array1,array2,result_array_cl,n);
    
    
    
}