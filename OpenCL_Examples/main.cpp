//
//  main.cpp
//
//  Created by Sadashige Ishida on 12/16/17.
//

#include <iostream>
#include <vector>
#include <functional>
#include <chrono>

#include <OpenCL/opencl.h>

const unsigned MAX_PLATFORM = 4;
const unsigned MAX_DEVICE = 4;

template <typename Func, typename ...Args>
void measure_runtime(Func &target_func,const unsigned iteration,Args...args ){
    
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();
    
    for(int i=0;i<iteration;++i){
        target_func(args...);
    }
    
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //Convert time to ms.
    
    std::cout<<"Computational Timing: "<<elapsed<<"ms"<<std::endl;
    
};

void ErrorChecker(cl_int result, const char *title)
{
    if (result != CL_SUCCESS) {
        std::cout << "Error: " << title << "(" << result << ")\n";
    }
}

cl_int err = CL_SUCCESS;
void EerrorChecker(const char *title)
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

void add_array_cpu(float * input1, float *input2,float*output,unsigned num_elements){
    
    int iterations=30;
    for(int ite=0;ite<iterations;++ite){
        for(int j=0;j<num_elements;++j){
            output[j] += input1[j]+input2[j];
        }
    }
    
}

void perform_via_cpu(float * array1, float *array2,float*result_array_normal,unsigned num_elements){
    
    measure_runtime(add_array_cpu,1, array1,array2,result_array_normal,num_elements);
    
    const bool display_computational_results=false;
    if(display_computational_results){
        for (int i = 0; i < num_elements; i++) {
            std::cout << result_array_normal[i] << ", ";
        }
        std::cout << "\n";
        
    }
    
}

int perform_via_opencl(float * array1, float *array2,float*result_array_cl,unsigned num_elements){
    
    auto exec_kernel=[&](){
        // Obtain platform.
        cl_platform_id platforms[MAX_PLATFORM];
        cl_uint platformCount;
        ErrorChecker(clGetPlatformIDs(MAX_PLATFORM, platforms, &platformCount), "clGetPlatformIDs");
        if (platformCount == 0) {
            std::cerr << "No platform.\n";
            exit(EXIT_FAILURE);
        }
        
        // Display platform information
        for (int i = 0; i < platformCount; i++) {
            char vendor[100] = {0};
            char version[100] = {0};
            ErrorChecker(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr), "clGetPlatformInfo");
            ErrorChecker(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, nullptr), "clGetPlatformInfo");
            std::cout << "Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << "\n";
        }
        
        // Obtain devices
        cl_device_id devices[MAX_DEVICE];
        cl_uint deviceCount;
        ErrorChecker(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, MAX_DEVICE, devices, &deviceCount), "clGetDeviceIDs");
        if (deviceCount == 0) {
            std::cerr << "No device.\n";
            exit(EXIT_FAILURE);
        }
        
        // Display devices information
        std::cout << deviceCount << " device(s) found.\n";
        for (int i = 0; i < deviceCount; i++) {
            char name[100] = {0};
            size_t len;
            ErrorChecker(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, &len), "clGetDeviceInfo");
            std::cout << "Device id: " << i << ", Name: " << name << "\n";
        }
        
        // Make context
        cl_context ctx = clCreateContext(nullptr, 1, devices, nullptr, nullptr, &err);
        EerrorChecker("clCreateContext");
        
        // Read precompiled opencl source
        const char* bitcode_path = "OpenCL/kernel.cl.gpu_64.bc";
        
        size_t len = strlen(bitcode_path);
        cl_program program = clCreateProgramWithBinary(ctx, 1, devices, &len, (const unsigned char**)&bitcode_path, nullptr, &err);
        EerrorChecker("clCreateProgramWithBinary");
        
        // build
        ErrorChecker(clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr), "clBuildProgram");
        
        // make kernel
        cl_kernel kernel = clCreateKernel(program, "add_array_opencl", &err);
        EerrorChecker("clCreateKernel");
        
        // allocate device memory
        cl_mem device_mem1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_elements, array1, &err);
        
        cl_mem device_mem2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_elements, array2, &err);
        
        cl_mem device_mem_result = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * num_elements, result_array_cl, &err);
        EerrorChecker("clCreateBuffer");
        
        // set parameters to kernels
        ErrorChecker(clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_mem1), "clSetKernelArg");
        
        ErrorChecker(clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_mem2), "clSetKernelArg");
        
        ErrorChecker(clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_mem_result), "clSetKernelArg");
        
        ErrorChecker(clSetKernelArg(kernel, 3, sizeof(int), &num_elements), "clSetKernelArg");
        
        // make command queue
        cl_command_queue q = clCreateCommandQueue(ctx, devices[0], 0, &err);
        EerrorChecker("clCreateCommandQueue");
        
        // execute kernels
        size_t global = num_elements;
        ErrorChecker(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr), "clEnqueueNDRangeKernel");
        
        // Read the results
        ErrorChecker(clEnqueueReadBuffer(q, device_mem_result, CL_TRUE, 0, sizeof(float) * num_elements, result_array_cl, 0, nullptr, nullptr), "clEnqueueReadBuffer");
        
        // Release command queue
        ErrorChecker(clReleaseCommandQueue(q), "clReleaseCommandQueue");
        
        // Release memory of the devices
        ErrorChecker(clReleaseMemObject(device_mem1), "clReleaseMemObject");
        
        // Release kernel
        ErrorChecker(clReleaseKernel(kernel), "clReleaseKernel");
        
        // Release program
        ErrorChecker(clReleaseProgram(program), "clReleaseProgram");
        
        // Release context
        ErrorChecker(clReleaseContext(ctx), "clReleaseContext");
        
    };
    
    measure_runtime(exec_kernel,1);
    
    const bool display_computational_results=false;
    if(display_computational_results){
        for (int i = 0; i < num_elements; i++) {
            std::cout << result_array_cl[i] << ", ";
        }
        std::cout << "\n";
        
    }
    
    return EXIT_SUCCESS;
}

int main(int argc, const char * argv[])
{
    // Prepare data
    int num_elements = 100000000;
    float *array1,*array2,*result_array_cl,*result_array_normal;
    array1=new float[num_elements];
    array2=new float[num_elements];
    result_array_cl=new float[num_elements];
    result_array_normal=new float[num_elements];
    
    prepare_arrays(array1,array2,result_array_normal,num_elements);
    prepare_arrays(array1,array2,result_array_cl,num_elements);
    
    //Compute two ways, straightforward v.s. opencl.
    std::cout<<"[Straightforwad Computation via CPU]"<<std::endl;
    perform_via_cpu(array1,array2,result_array_normal,num_elements);
    std::cout<<std::endl;
    std::cout<<"[Parallel Computation via OpenCL]"<<std::endl;
    perform_via_opencl(array1,array2,result_array_cl,num_elements);
    
}