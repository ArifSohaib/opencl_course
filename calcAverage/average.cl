__kernel void sum_float(__global float* vector, __global float *result)
{
    int i = get_global_id(0);
    float value = 0.0;
    for(int idx = 0; idx<1024;idx++){
        value+=vector[idx];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) == 0){
        result[i] = value/1024;
    }
}