#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "mnist_reader.hpp"
#include "mnist_reader_less.hpp"

using namespace std;
using namespace std::chrono;

__device__ size_t const N_INPUT_UNIT = 784;
__device__ size_t const N_OUTPUT_UNIT = 10;
__device__ float const LEARN_RATE = 1.;

__device__ float sigmoid(float x) {
    return (float)1. / (1. + exp((-1.) * x));
}

__device__ float sigmoid_derivative(float x) {
    return sigmoid(x) * (1. - sigmoid(x));
}

void initRandom(float* A, int size) {
    for (size_t i = 0; i < size; i++) {
        A[i] = ((float)2. * rand() / (RAND_MAX)) - 1.;
    }
}

void printArray(float* A, int size) {
    for (size_t i = 0; i < size; i++)
        printf("%f ", A[i]);
    printf("\n");
}

void zeroArray(float* A, int size) {
    for (size_t i = 0; i < size; i++)
        A[i] = 0.;
}
void zeroArray(int* A, int size) {
    for (size_t i = 0; i < size; i++)
        A[i] = 0;
}

void copyArray(float* A, float* B, int size) {
    for (size_t i = 0; i < size; i++)
        A[i] = B[i];
}

/* Cumulative gradient descent for input_weight of each batch */
__global__ void gradientDescentInputWeight(float* input_weight, float const* delta_dense, float const* input_layer,
    int n_sample) {
    // <<<N_INPUT_UNIT, n_unit>>>
    // Input layer weight update
    input_weight[blockIdx.x * blockDim.x + threadIdx.x] -= LEARN_RATE / (float)n_sample * input_layer[blockIdx.x] *
        delta_dense[threadIdx.x];
    // Input layer has no Bias!!!
}

/* Cumulative gradient descent for dense_bias of each batch */
__global__ void gradientDescentDenseBias(float* dense_bias, float const* delta_dense, int n_sample) {
    // <<<n_layer, n_unit>>>
    dense_bias[blockIdx.x * blockDim.x + threadIdx.x] -= LEARN_RATE / (float)n_sample *
        delta_dense[blockIdx.x * blockDim.x + threadIdx.x];
}

/* Cumulative gradient descent for dense_weight of each batch */
__global__ void gradientDescentDenseWeight(float* dense_weight, float const* delta_dense, float const* dense_layer, int n_sample) {
    // <<<dim3(n_layer, n_unit), n_unit>>>, gridDim.y = blockDim.x = n_unit
    // Dense layer weight update
    dense_weight[blockIdx.x * blockDim.x * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x] -= LEARN_RATE / (float)n_sample *
        dense_layer[blockIdx.x * blockDim.x + blockIdx.y] * delta_dense[(blockIdx.x + 1) * blockDim.x + threadIdx.x];
}

/* Cumulative gradient descent for output_weight of each batch */
__global__ void gradientDescentOutputWeight(float* output_weight, float const* delta_output, float const* dense_layer, int n_layer,
    int n_sample) {
    // <<<n_unit, N_OUTPUT_UNIT>>>
    output_weight[blockIdx.x * blockDim.x + threadIdx.x] -= LEARN_RATE / (float)n_sample * dense_layer[(n_layer - 1) * gridDim.x + blockIdx.x] *
        delta_output[threadIdx.x];
}

__global__ void sigmoidBackPropagateDense(float* delta_dense, float const* dense_weight, float const* dense_pre_activation,
    float const* delta_output, int curr_layer) {
    // <<<1,n_unit>>>
    // Back propagate one layer at a time, curr_layer decreases from (n_layer - 2) -> 0
    float derivative = 0.;
    float sum = 0.;

    sum = 0.;
    for (size_t j = 0; j < blockDim.x; j++) {
        sum += dense_weight[curr_layer * blockDim.x * blockDim.x + threadIdx.x * blockDim.x + j]
            * delta_dense[(curr_layer + 1) * blockDim.x + j];
    }
    derivative = sigmoid_derivative(dense_pre_activation[curr_layer * blockDim.x + threadIdx.x]);
    delta_dense[curr_layer * blockDim.x + threadIdx.x] += sum * derivative;
}

__global__ void sigmoidBackPropagateOutput(float* delta_dense, float const* dense_pre_activation,
    float const* delta_output, float const* output_weight, int n_layer) {
    // <<<1,n_unit>>>, outermost dense layer in contact with the output layer
    float derivative = 0.;
    float sum = 0.;

    sum = 0.;
    for (size_t j = 0; j < N_OUTPUT_UNIT; j++) {
        sum += output_weight[threadIdx.x * N_OUTPUT_UNIT + j] * delta_output[j];
    }
    derivative = sigmoid_derivative(dense_pre_activation[(n_layer - 1) * blockDim.x + threadIdx.x]);
    delta_dense[(n_layer - 1) * blockDim.x + threadIdx.x] = sum * derivative;
}

/* Softmax + Categorical Cross Entropy Cost Function */
__global__ void getCatCrossEntropyDeltaOutput(float* output_layer, float* delta_output, float* desired_output) {
    // <<<1,N_OUTPUT_UNIT>>>
    // Make sure to zero the delta_output array before running each sample!!!
    // TODO: Change to atomicAdd for parallel sample back propagation
    delta_output[threadIdx.x] += (output_layer[threadIdx.x] - desired_output[threadIdx.x]);
}

__global__ void sigmoidActivationForwardInput(float* dense_layer, float* dense_pre_activation, float* input_layer,
    float* input_weight, float* dense_bias) {
    // <<<1, n_unit>>>
    float sum = 0.;
    for (size_t i = 0; i < N_INPUT_UNIT; i++) {
        sum += input_weight[i * blockDim.x + threadIdx.x] * input_layer[i];
    }
    dense_pre_activation[threadIdx.x] = sum + dense_bias[threadIdx.x];
    dense_layer[threadIdx.x] = sigmoid(dense_pre_activation[threadIdx.x]);
}

__global__ void sigmoidActivationForwardDense(float* dense_layer, float* dense_pre_activation, int curr_layer,
    float const* dense_weight, float const* dense_bias) {
    // <<<1, n_unit>>>
    // unlike the host version, we propagate only one layer forward, increases curr_layer from 1 -> n_layer-1
    float sum = 0.;
    for (size_t j = 0; j < blockDim.x; j++) {
        sum += dense_weight[(curr_layer - 1) * blockDim.x * blockDim.x + j * blockDim.x + threadIdx.x] *
            dense_layer[(curr_layer - 1) * blockDim.x + j];
    }
    dense_pre_activation[curr_layer * blockDim.x + threadIdx.x] = sum + dense_bias[curr_layer * blockDim.x + threadIdx.x];
    dense_layer[curr_layer * blockDim.x + threadIdx.x] = sigmoid(dense_pre_activation[curr_layer * blockDim.x + threadIdx.x]);
}

__global__ void softmaxActivationForwardOutput(float* dense_layer, float* output_layer, float* output_weight,
    float* output_pre_activation, int n_unit, int n_layer) {
    // <<<1, N_OUTPUT_UNIT>>>
    // each thread takes care of a unit on the output layer
    float sum = 0.;
    for (size_t i = 0; i < n_unit; i++) {
        sum += output_weight[i * blockDim.x + threadIdx.x] * dense_layer[(n_layer - 1) * n_unit + i];
    }
    output_pre_activation[threadIdx.x] = sum;

    // apply exponent
    __shared__ float temp_val[N_OUTPUT_UNIT];
    __shared__ float sum_exp[1];
    temp_val[threadIdx.x] = exp(sum);

    __syncthreads();

    // use thread 0 to calcalculate the sum of exp's in the softmax equation
    if (threadIdx.x == 0) {
        sum = 0.;
        for (int i = 0; i < blockDim.x; i++) {
            sum += temp_val[i];
        }
        sum_exp[0] = sum;
    }

    __syncthreads();
    output_layer[threadIdx.x] = temp_val[threadIdx.x] / sum_exp[0];
}

int main(int argc, char** argv)
{
    int n_layer{ 5 }, n_unit{ 100 }, n_sample{ 20 }, n_epoch{ 20 };

    if (argc > 4) {
        n_layer = atoi(argv[1]); // nl
        n_unit = atoi(argv[2]); // nh
        n_epoch = atoi(argv[3]); // ne
        n_sample = atoi(argv[4]); // nb
    }

    /***************************************************************************************/
    /****************************-HOST MEMORY INITIALIZATION-*******************************/
    /***************************************************************************************/

    /***************************************************************************************/
    float* input_weight = (float*)malloc((size_t)N_INPUT_UNIT * n_unit * sizeof(float));
    /* dense_weight[i][j][k] is the weight going from layer i unit j to layer i+1 unit k */
    float* dense_weight = (float*)malloc((size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float)); // A
    float* dense_bias = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* dense_layer = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* input_layer = (float*)malloc((size_t)N_INPUT_UNIT * sizeof(float));
    float* output_layer = (float*)malloc((size_t)N_OUTPUT_UNIT * sizeof(float));
    float* output_weight = (float*)malloc((size_t)n_unit * N_OUTPUT_UNIT * sizeof(float));
    /***************************************************************************************/
    float* dense_pre_activation = (float*)malloc((size_t)n_layer * n_unit * sizeof(float)); // Z
    float* output_pre_activation = (float*)malloc((size_t)N_OUTPUT_UNIT * sizeof(float));
    /***************************************************************************************/
    float* delta_dense = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* delta_output = (float*)malloc((size_t)N_OUTPUT_UNIT * sizeof(float));
    initRandom(input_weight, N_INPUT_UNIT * n_unit);
    initRandom(dense_weight, (n_layer - 1) * n_unit * n_unit);
    initRandom(dense_bias, n_layer * n_unit);
    initRandom(output_weight, n_unit * N_OUTPUT_UNIT);

    float* desired_output = (float*)malloc((size_t)N_OUTPUT_UNIT * sizeof(float)); // Y

    float* dense_weight_temp = (float*)malloc((size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float));
    float* output_weight_temp = (float*)malloc((size_t)n_unit * N_OUTPUT_UNIT * sizeof(float));
    float* input_weight_temp = (float*)malloc((size_t)N_INPUT_UNIT * n_unit * sizeof(float));
    float* dense_bias_temp = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));

    float totalTime = 0.;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data");
    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
    /***************************************************************************************/
    /***************************-DEVICE MEMORY INITIALIZATION-******************************/
    /***************************************************************************************/
    float* input_weight_dev, * dense_weight_dev, * dense_bias_dev, * dense_layer_dev, * input_layer_dev,
        * output_layer_dev, * output_weight_dev, * dense_pre_activation_dev, * output_pre_activation_dev,
        * delta_dense_dev, * delta_output_dev, * desired_output_dev, * dense_weight_temp_dev,
        * output_weight_temp_dev, * input_weight_temp_dev, * dense_bias_temp_dev;

    cudaMalloc((void**)&input_weight_dev, (size_t)N_INPUT_UNIT * n_unit * sizeof(float));
    cudaMalloc((void**)&dense_weight_dev, (size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float));
    cudaMalloc((void**)&dense_bias_dev, (size_t)n_layer * n_unit * sizeof(float));
    cudaMalloc((void**)&dense_layer_dev, (size_t)n_layer * n_unit * sizeof(float));
    cudaMalloc((void**)&input_layer_dev, (size_t)N_INPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&output_layer_dev, (size_t)N_OUTPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&output_weight_dev, (size_t)n_unit * N_OUTPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&dense_pre_activation_dev, (size_t)n_layer * n_unit * sizeof(float));
    cudaMalloc((void**)&output_pre_activation_dev, (size_t)N_OUTPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&delta_dense_dev, (size_t)n_layer * n_unit * sizeof(float));
    cudaMalloc((void**)&delta_output_dev, (size_t)N_OUTPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&desired_output_dev, (size_t)N_OUTPUT_UNIT * sizeof(float));

    cudaMalloc((void**)&dense_weight_temp_dev, (size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float));
    cudaMalloc((void**)&output_weight_temp_dev, (size_t)n_unit * N_OUTPUT_UNIT * sizeof(float));
    cudaMalloc((void**)&input_weight_temp_dev, (size_t)N_INPUT_UNIT * n_unit * sizeof(float));
    cudaMalloc((void**)&dense_bias_temp_dev, (size_t)n_layer * n_unit * sizeof(float));

    cudaMemcpy(input_weight_dev, input_weight, (size_t)N_INPUT_UNIT * n_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dense_weight_dev, dense_weight, (size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dense_bias_dev, dense_bias, (size_t)n_layer * n_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_weight_dev, output_weight, (size_t)n_unit * N_OUTPUT_UNIT * sizeof(float), cudaMemcpyHostToDevice);
    /***************************************************************************************/
    /*****************************-EPOCH PREPARATION-*********************************/
    /***************************************************************************************/
    int epoch_sample_size = dataset.test_images.size();   // the first epoch_sample_size samples to be studied in each epoch
    epoch_sample_size = 200;    // !!!!!!!!!!!!Use this line to change the content you want Neuro-Sama to study!!!!!!!!!!!!!!!!!!!!!!!!!
    size_t randomSample = 0;
    vector<size_t> origin_samples, epoch_samples;
    /***************************************************************************************/
    /*****************************-LOOP TROUGH THE BATCHES-*********************************/
    /***************************************************************************************/
    for (size_t curr_epoch = 0; curr_epoch < n_epoch; curr_epoch++) {
        // Prepare the study material for this epoch
        for (size_t i = 0; i < epoch_sample_size; i++) {
            origin_samples.push_back(i);
        }
        while (!origin_samples.empty()) {
            randomSample = rand() % origin_samples.size();
            epoch_samples.push_back(origin_samples.at(randomSample));
            origin_samples.erase(origin_samples.begin() + randomSample);
        }

        for (size_t curr_batch = 0; curr_batch < epoch_sample_size / n_sample; curr_batch++) {
            // Create copies of the weights and biases which are constant during a batch
            cudaMemcpy(dense_weight_temp_dev, dense_weight_dev, (size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(output_weight_temp_dev, output_weight_dev, (size_t)n_unit * N_OUTPUT_UNIT * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(input_weight_temp_dev, input_weight_dev, (size_t)N_INPUT_UNIT * n_unit * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(dense_bias_temp_dev, dense_bias_dev, (size_t)n_layer * n_unit * sizeof(float), cudaMemcpyDeviceToDevice);

            auto start = high_resolution_clock::now();
            /***************************************************************************************/
            /************************-ACCUMULATE ERRORS THROUGH SAMPLES-****************************/
            /***************************************************************************************/
            for (size_t sample_count = 0; sample_count < n_sample; sample_count++) {
                size_t curr_sample = epoch_samples.back();
                epoch_samples.pop_back();
                // Zeroing the output error array
                cudaMemset(delta_output_dev, 0, (size_t)N_OUTPUT_UNIT * sizeof(float));
                // Zeroing the delta_dense array
                cudaMemset(delta_dense_dev, 0, (size_t)n_layer * n_unit * sizeof(float));
                // Enter the input layer
                for (size_t i = 0; i < N_INPUT_UNIT; i++) {
                    input_layer[i] = (float)dataset.test_images.at(curr_sample).at(i);
                }
                cudaMemcpy(input_layer_dev, input_layer, (size_t)N_INPUT_UNIT * sizeof(float), cudaMemcpyHostToDevice);
                /***************************************************************************************/
                /*******************************-FORWARD PROPAGATION-***********************************/
                /***************************************************************************************/
                // From input layer to first dense layer
                sigmoidActivationForwardInput << <1, n_unit >> > (dense_layer_dev, dense_pre_activation_dev, input_layer_dev,
                    input_weight_temp_dev, dense_bias_temp_dev);
                // through the dense layers
                for (size_t curr_layer = 1; curr_layer < n_layer; curr_layer++) {
                    sigmoidActivationForwardDense << <1, n_unit >> > (dense_layer_dev, dense_pre_activation_dev, curr_layer,
                        dense_weight_temp_dev, dense_bias_temp_dev);
                }
                // Outermost dense layer to output. Softmax output.
                softmaxActivationForwardOutput << <1, N_OUTPUT_UNIT >> > (dense_layer_dev, output_layer_dev, output_weight_temp_dev,
                    output_pre_activation_dev, n_unit, n_layer);

                // update the desired output Y
                zeroArray(desired_output, N_OUTPUT_UNIT);
                desired_output[(size_t)dataset.test_labels.at(curr_sample)] = 1.;   // change test label into train label to have more data
                cudaMemcpy(desired_output_dev, desired_output, (size_t)N_OUTPUT_UNIT * sizeof(float), cudaMemcpyHostToDevice);
                /***************************************************************************************/
                /*********************************-BACK PROPAGATION-************************************/
                /***************************************************************************************/
                // Get the output layer errors for the current sample
                getCatCrossEntropyDeltaOutput << <1, N_OUTPUT_UNIT >> > (output_layer_dev, delta_output_dev, desired_output_dev);
                // Start back propagating the full delta matrix using the output error for the current sample
                sigmoidBackPropagateOutput << <1, n_unit >> > (delta_dense_dev, dense_pre_activation_dev, delta_output_dev,
                    output_weight_temp_dev, n_layer);

                for (int curr_layer = n_layer - 2; curr_layer >= 0; curr_layer--) {
                    sigmoidBackPropagateDense << <1, n_unit >> > (delta_dense_dev, dense_weight_temp_dev, dense_pre_activation_dev,
                        delta_output_dev, curr_layer);
                }

                /***************************************************************************************/
                /************************-LEARNING THROUGH GRADIENT DESCENT-****************************/
                /***************************************************************************************/

                // Start gradient descent to change the weights and biases accumulated through each sample, order does not matter.
                // Note that we use output_weight_temp, dense_weight_temp, etc in each sample forward back propagation as
                // those matrices are not modified within each batch calculation. One batch is one learning experience for all the samples.

                gradientDescentOutputWeight << <n_unit, N_OUTPUT_UNIT >> > (output_weight_dev, delta_output_dev, dense_layer_dev, n_layer, n_sample);
                dim3 grid(n_layer - 1, n_unit);
                gradientDescentDenseWeight << <grid, n_unit >> > (dense_weight_dev, delta_dense_dev, dense_layer_dev, n_sample);
                gradientDescentDenseBias << <n_layer, n_unit >> > (dense_bias_dev, delta_dense_dev, n_sample);
                gradientDescentInputWeight << <N_INPUT_UNIT, n_unit >> > (input_weight_dev, delta_dense_dev, input_layer_dev, n_sample);
            }

            auto stop = high_resolution_clock::now();
            // Get duration. Substart timepoints to
            // get duration. To cast it to proper unit
            // use duration cast method
            auto duration = duration_cast<milliseconds>(stop - start);
            if (curr_batch > 1) {   // First iter is warm up
                totalTime += (float)duration.count() / (float)1000.;
            }
            printf("In the %dth epoch, Neuro-Sama finished studing the %dth batch in %f seconds. The grind rate is %f.\n", curr_epoch, curr_batch, 
                (float)duration.count() / (float)1000., (float)1000/(float)duration.count());
        }
    }

    /***************************************************************************************/
    /*********************************-TESTING RESULTS-*************************************/
    /***************************************************************************************/
    float avgTime = totalTime / (float)(n_epoch * epoch_sample_size / n_sample - 1);
    printf("Total time = %f\n", totalTime);
    printf("The average time needed for each batch is %fs.\nShowing results after learning through samples.\n", avgTime);

    bool continue_test = true;
    string input;
    int curr_sample;
    while (continue_test)
    {
        cout << "Enter the sample # you want to test: ";
        cin >> input;
        curr_sample = stoi(input);
        //cout << "\nThinking....\n";
        for (size_t i = 0; i < N_INPUT_UNIT; i++) {
            input_layer[i] = (float)dataset.test_images.at(curr_sample).at(i);
        }
        printf("\n");

        // Forward propagation through three regions
        // Enter the input layer
        cudaMemcpy(input_layer_dev, input_layer, (size_t)N_INPUT_UNIT * sizeof(float), cudaMemcpyHostToDevice);

        // Forward propagation through three regions
        sigmoidActivationForwardInput << <1, n_unit >> > (dense_layer_dev, dense_pre_activation_dev, input_layer_dev,
            input_weight_dev, dense_bias_dev);
        for (size_t curr_layer = 1; curr_layer < n_layer; curr_layer++) {
            sigmoidActivationForwardDense << <1, n_unit >> > (dense_layer_dev, dense_pre_activation_dev, curr_layer,
                dense_weight_dev, dense_bias_dev);
        }
        // Use sigmoidActivationForwardOutput for sigmoid output layer
        softmaxActivationForwardOutput << <1, N_OUTPUT_UNIT >> > (dense_layer_dev, output_layer_dev, output_weight_dev,
            output_pre_activation_dev, n_unit, n_layer);

        cudaMemcpy(output_layer, output_layer_dev, (size_t)N_OUTPUT_UNIT * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Result after learning for digit %d: ", dataset.test_labels.at(curr_sample));
        printArray(output_layer, N_OUTPUT_UNIT);
        cout << "Continue? (y/n): ";
        cin >> input;
        if (input != "y") continue_test = false;
    }

    /***************************************************************************************/
    cudaFree(input_weight_dev);
    cudaFree(dense_weight_dev);
    cudaFree(dense_bias_dev);
    cudaFree(dense_layer_dev);
    cudaFree(input_layer_dev);
    cudaFree(output_layer_dev);
    cudaFree(dense_pre_activation_dev);
    cudaFree(output_pre_activation_dev);
    cudaFree(delta_dense_dev);
    cudaFree(delta_output_dev);
    cudaFree(desired_output_dev);
    cudaFree(dense_weight_temp_dev);
    cudaFree(output_weight_temp_dev);
    cudaFree(input_weight_temp_dev);
    cudaFree(dense_bias_temp_dev);

    free(input_weight);
    free(dense_weight);
    free(dense_bias);
    free(dense_layer);
    free(input_layer);
    free(output_layer);
    free(dense_pre_activation);
    free(output_pre_activation);
    free(delta_dense);
    free(delta_output);
    free(desired_output);
    free(dense_weight_temp);
    free(output_weight_temp);
    free(input_weight_temp);
    free(dense_bias_temp);

    return 0;
}