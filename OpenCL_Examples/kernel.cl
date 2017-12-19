__kernel
void add_matrix_cl(__global float* input1,__global float* input2,__global float* output, const int n)
{
    int index = get_global_id(0);
    
    int times=1;
    for(int j=0;j<times;++j){
        if (index < n) {
            output[index] += input1[index]+input2[index];
        }
    }
}