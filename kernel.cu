#include <cuda.h>
#include <stdio.h>

//#include <cuda_runtime.h>  


#define TF 5.
#define INTERVAL_NUM 100 // N

#define INTERPOLATION_NUM_NORM 6 // m
#define INTERPOLATION_NUM_LAMBDA ((INTERPOLATION_NUM_NORM) + 1) // n

#define LEN_LAMBDA ((INTERVAL_NUM) * ((INTERPOLATION_NUM_LAMBDA) - 1) + 1)
#define LEN_NORM ((INTERVAL_NUM) * (INTERPOLATION_NUM_NORM))

#define NX 2
#define NY 1

#define INTEGRATION_POINT_NUM 8
#define INTEGRATION_ARRAY_LEN ((INTEGRATION_POINT_NUM) * (INTERVAL_NUM))

#define GRID_X (threadIdx.x + blockIdx.x * blockDim.x)

/*
void init_interpolation_and_integration(double* interpolation_coef_norm, double* interpolation_coef_lambda
	, double* integration_point, double* integration_weight) {

}

void interpolation_and_integration_data_transfer(double* interpolation_coef_norm, double* interpolation_coef_lambda
	, double* integration_point, double* integration_weight, double* d_interpolation_coef_norm
	, double* d_interpolation_coef_lambda, double* d_integration_point, double* d_integration_weight) {
	cudaMalloc((void**)&d_interpolation_coef_norm, sizeof(double) * INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM);
	cudaMemcpy(d_interpolation_coef_norm, interpolation_coef_norm
		, sizeof(double) * INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_interpolation_coef_lambda, sizeof(double) * INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA);
	cudaMemcpy(d_interpolation_coef_lambda, interpolation_coef_lambda
		, sizeof(double) * INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_integration_point, sizeof(double) * INTEGRATION_POINT_NUM);
	cudaMemcpy(d_integration_point, integration_point
		, sizeof(double) * INTEGRATION_POINT_NUM, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_integration_weight, sizeof(double) * INTEGRATION_POINT_NUM);
	cudaMemcpy(d_integration_weight, integration_weight
		, sizeof(double) * INTEGRATION_POINT_NUM, cudaMemcpyHostToDevice);
}

void allocate_and_initialize_arrays(double* x_array, double* y_array, double* lambda_array, double* lambda_g_array
	, double* d_x_array, double* d_y_array, double* d_lambda_array, double* d_lambda_g_array
	, double* d_x_integration_array, double* d_y_integration_array, double* d_lambda_integration_array
	, double* d_lambda_g_integration_array) {
	cudaMalloc((void**)&d_x_array, sizeof(double) * NX * LEN_NORM);
	cudaMalloc((void**)&d_y_array, sizeof(double) * NY * LEN_NORM);
	cudaMalloc((void**)&d_lambda_array, sizeof(double) * NX * LEN_LAMBDA);
	cudaMalloc((void**)&d_lambda_g_array, sizeof(double) * NY * LEN_NORM);
	cudaMemcpy(d_x_array, x_array, sizeof(double) * NX * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_array, y_array, sizeof(double) * NY * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_array, lambda_array, sizeof(double) * NX * LEN_LAMBDA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_g_array, lambda_g_array, sizeof(double) * NY * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_x_integration_array, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_y_integration_array, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_lambda_integration_array, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_lambda_g_integration_array, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_x_integration_array, 0, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_y_integration_array, 0, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_lambda_integration_array, 0, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_lambda_g_integration_array, 0, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
}
*/

__global__
void interpolate_norm(double* d_x_array, double* d_x_integration_array, double* d_interpolation_coef_norm) {
	__shared__ double s_interpolation_coef_norm[INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM];
	if (threadIdx.x < INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM) {
		s_interpolation_coef_norm[threadIdx.x] = d_interpolation_coef_norm[threadIdx.x];
	}
	__syncthreads();

	int offset = GRID_X;
	if (offset < NX * INTEGRATION_ARRAY_LEN) {
		int index0 = offset / INTEGRATION_ARRAY_LEN;
		int index1 = offset % INTEGRATION_ARRAY_LEN;
		int index10 = index1 / INTEGRATION_POINT_NUM;
		int index11 = index1 % INTEGRATION_POINT_NUM;

		double sum = 0;
		for (int i = 0; i < INTERPOLATION_NUM_NORM; ++i) {
			double dx = (double)index11 * INTERPOLATION_NUM_NORM / INTEGRATION_POINT_NUM - i;
			double x = 1.;
			for (int j = 0; j < INTERPOLATION_NUM_NORM; ++j) {
				sum += d_x_array[index0 * LEN_NORM + index10 * INTERPOLATION_NUM_NORM + i] * s_interpolation_coef_norm[i * INTERPOLATION_NUM_NORM + j] * x;
				x *= dx;
			}
		}
		d_x_integration_array[offset] = sum;
	}
}


__global__
void interpolate_lambda(double* d_lambda_array, double* d_lambda_integration_array, double* d_interpolation_coef_lambda) {
	__shared__ double s_interpolation_coef_lambda[INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA];
	if (threadIdx.x < INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA) {
		s_interpolation_coef_lambda[threadIdx.x] = d_interpolation_coef_lambda[threadIdx.x];
	}
	__syncthreads();

	int offset = GRID_X;
	if (offset < NX * INTEGRATION_ARRAY_LEN) {
		int index0 = offset / INTEGRATION_ARRAY_LEN;
		int index1 = offset % INTEGRATION_ARRAY_LEN;
		int index10 = index1 / INTEGRATION_POINT_NUM;
		int index11 = index1 % INTEGRATION_POINT_NUM;

		double sum = 0;
		for (int i = 0; i < INTERPOLATION_NUM_LAMBDA; ++i) {
			double dx = (double)index11 * INTERPOLATION_NUM_LAMBDA / INTEGRATION_POINT_NUM - i;
			double x = 1.;
			for (int j = 0; j < INTERPOLATION_NUM_LAMBDA; ++j) {
				sum += d_lambda_array[index0 * LEN_LAMBDA + index10 * (INTERPOLATION_NUM_LAMBDA - 1) + i] * d_interpolation_coef_lambda[i * INTERPOLATION_NUM_LAMBDA + j] * x;
				x *= dx;
			}
		}
		d_lambda_integration_array[offset] = sum;
	}
}




int main()
{
	double interpolation_coef_norm[INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM] = { 1.0, -2.283333333333333, 1.875, -0.7083333333333333, 0.125, -0.008333333333333333, 1.0, -1.0833333333333333, -0.625, 1.0416666666666665, -0.375, 0.041666666666666664, 1.0, -0.3333333333333333, -1.25, 0.41666666666666663, 0.25, -0.08333333333333333, 1.0, 0.33333333333333326, -1.25, -0.4166666666666666, 0.24999999999999997, 0.08333333333333333, 1.0, 1.083333333333333, -0.6249999999999998, -1.0416666666666665, -0.37499999999999994, -0.041666666666666664, 1.0, 2.283333333333333, 1.875, 0.7083333333333334, 0.125, 0.008333333333333333 };
	double interpolation_coef_lambda[INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA] = { 1.0, -2.4499999999999997, 2.2555555555555555, -1.0208333333333333, 0.24305555555555552, -0.029166666666666667, 0.0013888888888888887, 1.0, -1.2833333333333332, -0.4083333333333333, 1.1666666666666665, -0.5833333333333333, 0.11666666666666667, -0.008333333333333333, 1.0, -0.5833333333333333, -1.1666666666666667, 0.7291666666666666, 0.14583333333333334, -0.14583333333333331, 0.020833333333333332, 1.0, -5.551115123125783e-17, -1.3611111111111112, 5.551115123125783e-17, 0.38888888888888884, 1.3877787807814457e-17, -0.027777777777777776, 1.0, 0.583333333333333, -1.1666666666666663, -0.7291666666666666, 0.14583333333333331, 0.14583333333333331, 0.020833333333333332, 1.0, 1.2833333333333332, -0.4083333333333332, -1.1666666666666665, -0.5833333333333334, -0.11666666666666667, -0.008333333333333333, 1.0, 2.45, 2.2555555555555555, 1.0208333333333333, 0.24305555555555552, 0.029166666666666664, 0.0013888888888888887 };

	double integration_point[INTEGRATION_POINT_NUM] = { -0.9602898564975362, -0.7966664774136267, -0.525532409916329, -0.18343464249564978, 0.18343464249564978, 0.525532409916329, 0.7966664774136267, 0.9602898564975362 };
	double integration_weight[INTEGRATION_POINT_NUM] = { 0.10122853629037669, 0.22238103445337434, 0.31370664587788705, 0.36268378337836177, 0.36268378337836177, 0.31370664587788705, 0.22238103445337434, 0.10122853629037669 };

	double x_array[NX * LEN_NORM] = { 0. };
	for (int i = 0; i < NX * LEN_NORM; ++i) {
		x_array[i] = 2. * i / INTERVAL_NUM;
	}
	double y_array[NY * LEN_NORM] = { 0. };
	double lambda_array[NX * LEN_LAMBDA] = { 0. };
	double lambda_g_array[NY * LEN_NORM] = { 0. };

	for (int i = 0; i < NX * LEN_LAMBDA; ++i) {
		lambda_array[i] = 2. * i;
	}

	double* d_interpolation_coef_norm, * d_interpolation_coef_lambda, * d_integration_point, * d_integration_weight;
	double* d_x_array, * d_y_array, * d_lambda_array, * d_lambda_g_array;
	double* d_x_integration_array, * d_y_integration_array, * d_lambda_integration_array, * d_lambda_g_integration_array;


	//	init_interpolation_and_integration(interpolation_coef_norm, interpolation_coef_lambda, integration_point, integration_weight);
	//	interpolation_and_integration_data_transfer(interpolation_coef_norm, interpolation_coef_lambda
	//		, integration_point, integration_weight, d_interpolation_coef_norm
	//		, d_interpolation_coef_lambda, d_integration_point, d_integration_weight);
	//	allocate_and_initialize_arrays(x_array, y_array, lambda_array, lambda_g_array, d_x_array, d_y_array, d_lambda_array, d_lambda_g_array,
	//		d_x_integration_array, d_y_integration_array, d_lambda_integration_array, d_lambda_g_integration_array);


	cudaMalloc((void**)&d_interpolation_coef_norm, sizeof(double) * INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM);
	cudaMemcpy(d_interpolation_coef_norm, interpolation_coef_norm
		, sizeof(double) * INTERPOLATION_NUM_NORM * INTERPOLATION_NUM_NORM, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_interpolation_coef_lambda, sizeof(double) * INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA);
	cudaMemcpy(d_interpolation_coef_lambda, interpolation_coef_lambda
		, sizeof(double) * INTERPOLATION_NUM_LAMBDA * INTERPOLATION_NUM_LAMBDA, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_integration_point, sizeof(double) * INTEGRATION_POINT_NUM);
	cudaMemcpy(d_integration_point, integration_point
		, sizeof(double) * INTEGRATION_POINT_NUM, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&d_integration_weight, sizeof(double) * INTEGRATION_POINT_NUM);
	cudaMemcpy(d_integration_weight, integration_weight
		, sizeof(double) * INTEGRATION_POINT_NUM, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_x_array, sizeof(double) * NX * LEN_NORM);
	cudaMalloc((void**)&d_y_array, sizeof(double) * NY * LEN_NORM);
	cudaMalloc((void**)&d_lambda_array, sizeof(double) * NX * LEN_LAMBDA);
	cudaMalloc((void**)&d_lambda_g_array, sizeof(double) * NY * LEN_NORM);
	cudaMemcpy(d_x_array, x_array, sizeof(double) * NX * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_array, y_array, sizeof(double) * NY * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_array, lambda_array, sizeof(double) * NX * LEN_LAMBDA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_g_array, lambda_g_array, sizeof(double) * NY * LEN_NORM, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_x_integration_array, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_y_integration_array, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_lambda_integration_array, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMalloc((void**)&d_lambda_g_integration_array, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_x_integration_array, 0, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_y_integration_array, 0, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_lambda_integration_array, 0, sizeof(double) * NX * INTEGRATION_ARRAY_LEN);
	cudaMemset(d_lambda_g_integration_array, 0, sizeof(double) * NY * INTEGRATION_ARRAY_LEN);


	int blocksize = 512;
	interpolate_norm << <int(ceil((double)NX * INTEGRATION_ARRAY_LEN / blocksize)), blocksize >> > (d_x_array, d_x_integration_array, d_interpolation_coef_norm);
	interpolate_norm << <int(ceil((double)NY * INTEGRATION_ARRAY_LEN / blocksize)), blocksize >> > (d_y_array, d_y_integration_array, d_interpolation_coef_norm);
	interpolate_lambda << <int(ceil((double)NX * INTEGRATION_ARRAY_LEN / blocksize)), blocksize >> > (d_lambda_array, d_lambda_integration_array, d_interpolation_coef_lambda);
	interpolate_norm << <int(ceil((double)NY * INTEGRATION_ARRAY_LEN / blocksize)), blocksize >> > (d_lambda_g_array, d_lambda_g_integration_array, d_interpolation_coef_norm);


	double o[NX * INTEGRATION_ARRAY_LEN];
	cudaMemcpy(o, d_lambda_integration_array, sizeof(double) * NX * INTEGRATION_ARRAY_LEN, cudaMemcpyDeviceToHost);
	for (int i = 0; i < NX * INTEGRATION_ARRAY_LEN; ++i) {
		printf("%lf\n", o[i]);
	}
	printf("\n\n\n\n\n");
	double o2[INTEGRATION_POINT_NUM];
	cudaMemcpy(o2, d_integration_point, sizeof(double) * INTEGRATION_POINT_NUM, cudaMemcpyDeviceToHost);
	for (int i = 0; i < INTEGRATION_POINT_NUM; ++i) {
		printf("%lf\n", o2[i]);
	}

	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//execute(c, a, b, 5);

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//	c[0], c[1], c[2], c[3], c[4]);

	//// cudaDeviceReset must be called before exiting in order for profiling and
	//// tracing tools such as Nsight and Visual Profiler to show complete traces.

	cudaDeviceReset();
	return 0;
}
