__kernel void matvec_mult(__global float4* matrix,
                          __global float4* vector,
                          __global float* result) {
   
   int i = get_global_id(0);
   result[i] = dot(matrix[i], vector[0]);

}

__kernel void matvec_mult_loop(const __global float* M,
                                    const __global float* V,
                                    uint width, uint height,
                                    __global float* W)
{
    // Row index
    uint y = get_global_id(0);
    if (y < height) {
    
        // Row pointer
        const __global float* row = M + y * width;

        // Compute dot product  
        float dotProduct = 0;
        for (int x = 0; x < width; ++x)
            dotProduct += row[x] * V[x];

        // Write result to global memory
        W[y] = dotProduct;
    }
}