//__kernel
//void addone(__global float* data, const int n)
//{
//    int index = get_global_id(0);
//
//
//    printf("%d\n", index);
//
//    if (index < n) {
//        data[index] += 1.0f;
//    }
//}


__kernel
void addmatrix(__global float* input1,__global float* input2,__global float* output, const int n)
{
    int index = get_global_id(0);
    
    int times=30;
    for(int j=0;j<times;++j){
        if (index < n) {
            output[index] += input1[index]+input2[index];
        }
    }
}