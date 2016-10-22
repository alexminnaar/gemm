#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <driver_types.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <ostream>

//fill gpu with random numbers
void GPU_fill_rand(float *A, int n_rows_A, int n_cols_A) {

	//pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	//set seed
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	//fill the array with random numbers on the gpu
	curandGenerateUniform(prng, A, n_rows_A * n_cols_A);
}

//On the gpu multiply C(m,n)=A(m,k)*B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m,
		const int k, const int n) {

	int lda = m, ldb = k, ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	//cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	//do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B,
			ldb, beta, C, ldc);

	//destroy the handle
	cublasDestroy(handle);

}

//print matrix
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for (int i = 0; i < nr_rows_A; ++i) {
		for (int j = 0; j < nr_cols_A; ++j) {
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main() {

	//Allocate 3 arrays on CPU
	int n_rows_A, n_cols_A, n_rows_B, n_cols_B, n_rows_C, n_cols_C;

	//Try square matrices
	n_rows_A = n_cols_A = n_rows_B = n_cols_B = n_rows_C = n_cols_C = 3;

	//actually allocate on CPU
	float *h_A = (float *) malloc(n_rows_A * n_cols_A * sizeof(float));
	float *h_B = (float *) malloc(n_rows_B * n_cols_B * sizeof(float));
	float *h_C = (float *) malloc(n_rows_C * n_cols_C * sizeof(float));

	//allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, n_rows_A * n_cols_A * sizeof(float));
	cudaMalloc(&d_B, n_rows_B * n_cols_B * sizeof(float));
	cudaMalloc(&d_C, n_rows_C * n_cols_C * sizeof(float));

	//actually fill the allocated arrays
	GPU_fill_rand(d_A, n_rows_A, n_cols_A);
	GPU_fill_rand(d_B, n_rows_B, n_cols_B);

	//copy the filled arrays back to the cpu to print them
	cudaMemcpy(h_A, d_A, n_rows_A * n_cols_A * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, n_rows_B * n_cols_B * sizeof(float),
			cudaMemcpyDeviceToHost);

	//print A and B matrices
	std::cout << "A = " << std::endl;
	print_matrix(h_A, n_rows_A, n_cols_A);
	std::cout << "B = " << std::endl;
	print_matrix(h_A, n_rows_A, n_cols_A);

	//multiply on gpu
	gpu_blas_mmul(d_A, d_B, d_C, n_rows_A, n_cols_A, n_cols_B);

	//copy multiplication result back to cpu
	cudaMemcpy(h_C,d_C,n_rows_C * n_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

	std::cout << "C = " << std::endl;
	print_matrix(h_C, n_rows_C, n_cols_C);

	//free gpu memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//free cpu memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
