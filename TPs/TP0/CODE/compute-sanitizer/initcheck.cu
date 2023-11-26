const int bs = 1;

__global__ void kernel(char *in, char *out) { out[threadIdx.x] = in[threadIdx.x]; }

int main(void) {
    char *d1, *d2;
    cudaMalloc(&d1, bs);
    cudaMalloc(&d2, bs);
    kernel<<<1, bs>>>(d1, d2);
    cudaDeviceSynchronize();
}
