__kernel
void add_array_opencl(__global float* input1,__global float* input2,__global float* output, const int n)
{
    int index = get_global_id(0);
    
    int iterations=30;
    for(int j=0;j<iterations;++j){
        if (index < n) {
            output[index] += input1[index]+input2[index];
        }
    }
}