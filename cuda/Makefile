CC = nvcc
CFLAGS = -arch=sm_70

gpu_neuro: neuro_gpu_cuda.cu
	$(CC) $(CFLAGS) -o gpu_neuro neuro_gpu_cuda.cu