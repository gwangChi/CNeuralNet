#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>
#include "mnist_reader.hpp"
#include "mnist_reader_less.hpp"

using namespace std;
using namespace std::chrono;

size_t const N_INPUT_UNIT = 784;
float const LEARN_RATE = 1.;

auto sigmoid = [](float x) {
    return (float)1. / (1. + exp((-1.) * x));
};

auto sigmoid_derivative = [](float x){
    return sigmoid(x) * (1. - sigmoid(x));
};

void initRandom(float* A, int size){
    for(size_t i=0; i<size; i++){
        A[i] = ((float)2. * rand() / (RAND_MAX)) - 1.;
    }
}

void printArray(float* A, int size){
    for(size_t i = 0; i < size; i++)
        printf("%f ", A[i]);
    printf("\n");
}

void zeroArray(float* A, int size){
    for(size_t i=0; i<size; i++)
        A[i] = 0;
}

void copyArray(float* A, float* B, int size){
    for(size_t i=0; i<size; i++)
        A[i] = B[i];
}

float crossEntropyLoss(float* output_layer, int y){
    return -1. * log(output_layer[y]);
}

/* Cumulative gradient descent for input_weight of each batch */
void gradientDescentInputWeight(float* input_weight, float const* delta_dense, float const* input_layer, 
                                int n_unit, int n_sample){
    for(size_t i = 0; i < N_INPUT_UNIT; i++){
        for(size_t j = 0; j < n_unit; j++){
            // Input layer weight update
            input_weight[i * n_unit + j] -= LEARN_RATE / (float)n_sample * input_layer[i] * delta_dense[j];
        }
        // Input layer has no Bias!!!
    }
}

/* Cumulative gradient descent for dense_bias of each batch */
void gradientDescentDenseBias(float* dense_bias, float const* delta_dense, int n_layer, 
                            int n_unit, int n_sample){
    for(size_t curr_layer = 0; curr_layer < n_layer; curr_layer++){
        for(size_t i = 0; i < n_unit; i++){
            dense_bias[curr_layer * n_unit + i] -= LEARN_RATE / (float)n_sample * delta_dense[curr_layer * n_unit + i];
        }
    }
}

/* Cumulative gradient descent for dense_weight of each batch */
void gradientDescentDenseWeight(float* dense_weight, float const* delta_dense, float const* dense_layer, int n_layer, 
                            int n_unit, int n_sample){
    for(size_t curr_layer = 0; curr_layer < n_layer - 1; curr_layer++){
        for(size_t i = 0; i < n_unit; i++){
            for(size_t j = 0; j < n_unit; j++){
                // Dense layer weight update
                dense_weight[curr_layer * n_unit * n_unit + i * n_unit + j] -= LEARN_RATE / (float)n_sample * 
                                dense_layer[curr_layer * n_unit + i] * delta_dense[(curr_layer + 1) * n_unit + j];
            }
        }
    }
}

/* Cumulative gradient descent for output_bias of each batch !!! NOT USED WITH SOFTMAX ACTIVATION!!! */
void gradientDescentOutputBias(float* output_bias, float const* delta_output, int n_layer, 
                            int n_output_unit, int n_sample){
    for(size_t i = 0; i < n_output_unit; i++){
        output_bias[i] -= LEARN_RATE / (float)n_sample * delta_output[i];
    }
}

/* Cumulative gradient descent for output_weight of each batch */
void gradientDescentOutputWeight(float* output_weight, float const* delta_output, float const* dense_layer, int n_layer,
                            int n_unit, int n_output_unit, int n_sample){
    for(size_t i = 0; i<n_unit; i++){
        for(size_t j=0; j<n_output_unit; j++){
            // Output layer weight update
            output_weight[i*n_output_unit+j] -= LEARN_RATE / (float)n_sample * dense_layer[(n_layer-1) * n_unit + i] *
                                delta_output[j];
        }
        // Note: Output layer has no bias when using softmax
    }
}

/* Similar to the output layer error, the back propagation only updates for one sample */
void sigmoidBackPropagate(float* delta_dense, float const* dense_weight, float const* dense_pre_activation, 
                            float const* delta_output, float const* output_weight, int n_layer, 
                            int n_unit, int n_output_unit){
    float derivative = 0.;
    float sum = 0.;

    // First obtain the outermost dense layer delta from the output layer delta
    for(size_t i = 0; i<n_unit; i++){
        sum = 0.;
        for(size_t j = 0; j<n_output_unit; j++){
            sum += output_weight[i * n_output_unit + j] * delta_output[j];
        }
        derivative = sigmoid_derivative(dense_pre_activation[(n_layer-1)*n_unit + i]);
        delta_dense[(n_layer-1)*n_unit + i] = sum * derivative;
    }
    // Back propagate the whole dense layer delta
    for(int curr_layer = n_layer-2; curr_layer >= 0; curr_layer--){
        for (size_t i = 0; i < n_unit; i++){
            sum = 0.;
            for (size_t j = 0; j < n_unit; j++){
                sum += dense_weight[curr_layer * n_unit * n_unit + i * n_unit + j] 
                                                        * delta_dense[(curr_layer+1) * n_unit + j];
            }
            derivative = sigmoid_derivative(dense_pre_activation[curr_layer * n_unit + i]);
            delta_dense[curr_layer * n_unit + i] += sum * derivative;
        }
    }
}

/* Softmax + Categorical Cross Entropy Cost Function */
void getCatCrossEntropyDeltaOutput(float* output_layer, float* delta_output, float* desired_output, int n_output_unit){
    // Make sure to zero the delta_output array before running each sample
    for(size_t i=0; i<n_output_unit; i++){
        delta_output[i] += (output_layer[i]-desired_output[i]);
    }
}

void getQuadraticDeltaOutput(){
    // TODO
}

void sigmoidActivationForwardInput(float* dense_layer, float* dense_pre_activation, float* input_layer, int n_unit, 
                                    float* input_weight, float* dense_bias) {
    float sum = 0.;

    for (size_t i = 0; i < n_unit; i++) {
        sum = 0.;
        for (size_t j = 0; j < N_INPUT_UNIT; j++) {
            sum += input_weight[j * n_unit + i] * input_layer[j];
        }
        dense_pre_activation[i] = sum + dense_bias[i];
        dense_layer[i] = sigmoid(dense_pre_activation[i]);
    }
}

void sigmoidActivationForwardDense(float* dense_layer, float* dense_pre_activation, int n_layer, int n_unit, 
                                    float* dense_weight, float* dense_bias) {
    float sum = 0.;

    for (size_t curr_layer = 1; curr_layer < n_layer; curr_layer++) {
        for (size_t i = 0; i < n_unit; i++) {
            sum = 0.;
            for (size_t j = 0; j < n_unit; j++) {
                sum += dense_weight[(curr_layer - 1) * n_unit * n_unit + j * n_unit + i] * 
                        dense_layer[(curr_layer - 1) * n_unit + j];
            }
            dense_pre_activation[curr_layer * n_unit + i] = sum + dense_bias[curr_layer * n_unit + i];
            dense_layer[curr_layer * n_unit + i] = sigmoid(dense_pre_activation[curr_layer * n_unit + i]);
        }
    }
}

void softmaxActivationForwardOutput(float* dense_layer, float* output_layer, float* output_weight, 
                                    float* output_pre_activation, int n_output_unit, int n_unit, int n_layer){
    float sum = 0.;

    for(size_t i = 0; i<n_output_unit; i++){
        sum = 0.;
        for(size_t j = 0; j<n_unit; j++){
            sum += output_weight[j * n_output_unit + i]*dense_layer[(n_layer - 1) * n_unit + j];
        }
        output_pre_activation[i] = sum;
    }

    sum = 0.;
    for(size_t i = 0; i<n_output_unit; i++){
        sum += exp(output_pre_activation[i]);
    }

    for(size_t i = 0; i<n_output_unit; i++){
        output_layer[i] =exp(output_pre_activation[i])/sum;
    }
}

void sigmoidActivationForwardOutput(float* dense_layer, float* output_layer, float const* output_bias, float* output_weight, 
                                    float* output_pre_activation, int n_output_unit, int n_unit, int n_layer){
    float sum = 0.;

    for(size_t i = 0; i<n_output_unit; i++){
        sum = 0.;
        for(size_t j = 0; j<n_unit; j++){
            sum += output_weight[j * n_output_unit + i]*dense_layer[(n_layer - 1) * n_unit + j];
        }
        output_pre_activation[i] = sum + output_bias[i];
        output_layer[i] = sigmoid(output_pre_activation[i]);
    }
}

int main(int argc, char **argv)
{
    int n_layer{ 5 }, n_unit{ 100 }, n_output_unit{ 10 }, n_sample{ 4 }, n_epoch{ 200 };

    if(argc > 4){
        n_layer = atoi(argv[1]);
        n_unit = atoi(argv[2]);
        n_epoch = atoi(argv[3]);
        n_sample = atoi(argv[4]);
    }
    /***************************************************************************************/
    float* input_weight = (float*)malloc((size_t)N_INPUT_UNIT * n_unit * sizeof(float));
    /* dense_weight[i][j][k] is the weight going from layer i unit j to layer i+1 unit k */
    float* dense_weight = (float*)malloc((size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float)); // A
    float* dense_bias = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* dense_layer = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* input_layer = (float*)malloc((size_t)N_INPUT_UNIT * sizeof(float));
    float* output_layer = (float*)malloc((size_t)n_output_unit * sizeof(float));
    float* output_weight = (float*)malloc((size_t)n_unit * n_output_unit * sizeof(float));
    /***************************************************************************************/
    float* dense_pre_activation = (float*)malloc((size_t)n_layer * n_unit * sizeof(float)); // Z
    float* output_pre_activation = (float*)malloc((size_t) n_output_unit * sizeof(float));
    /***************************************************************************************/
    float* delta_dense = (float*)malloc((size_t)n_layer * n_unit * sizeof(float));
    float* delta_output = (float*)malloc((size_t)n_output_unit * sizeof(float));
    /***************************************************************************************/
    /*********************************-INITIALIZATION-**************************************/
    /***************************************************************************************/
    initRandom(input_weight, N_INPUT_UNIT * n_unit);
    initRandom(dense_weight, (n_layer - 1) * n_unit * n_unit);
    initRandom(dense_bias, n_layer * n_unit);
    initRandom(output_weight, n_unit * n_output_unit);

    float *desired_output = (float *)malloc((size_t)n_output_unit * sizeof(float)); // Y

    float* dense_weight_temp = (float*)malloc((size_t)(n_layer - 1) * n_unit * n_unit * sizeof(float));
    float* output_weight_temp = (float*)malloc((size_t)n_unit * n_output_unit * sizeof(float));
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
    /*****************************-EPOCH PREPARATION-*********************************/
    /***************************************************************************************/
    int epoch_sample_size = dataset.test_images.size();   // the first epoch_sample_size samples to be studied in each epoch
    epoch_sample_size = 20;    // !!!!!!!!!!!!Use this line to change the content you want Neuro-Sama to study!!!!!!!!!!!!!!!!!!!!!!!!!
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
        for (size_t curr_batch = 0; curr_batch < epoch_sample_size / n_sample; curr_batch++)
        {

            // Create copies of the weights and biases which are constant during a batch
            copyArray(dense_weight_temp, dense_weight, (n_layer - 1) * n_unit * n_unit);
            copyArray(output_weight_temp, output_weight, n_unit * n_output_unit);
            copyArray(input_weight_temp, input_weight, N_INPUT_UNIT * n_unit);
            copyArray(dense_bias_temp, dense_bias, n_layer * n_unit);

            auto start = high_resolution_clock::now();
            float cost = 0.;
            /***************************************************************************************/
            /************************-ACCUMULATE ERRORS THROUGH SAMPLES-****************************/
            /***************************************************************************************/
            for (size_t sample_count = 0; sample_count < n_sample; sample_count++)
            {
                size_t curr_sample = epoch_samples.back();
                epoch_samples.pop_back();
                // Zeroing the output error array
                zeroArray(delta_output, n_output_unit);
                // Zeroing the delta_dense array
                zeroArray(delta_dense, n_layer * n_unit);

                // Enter the input layer
                for (size_t i = 0; i < N_INPUT_UNIT; i++)
                {
                    input_layer[i] = (float)dataset.test_images.at(curr_sample).at(i);
                }

                // Forward propagation through three regions
                sigmoidActivationForwardInput(dense_layer, dense_pre_activation, input_layer, n_unit, input_weight_temp, dense_bias_temp);
                sigmoidActivationForwardDense(dense_layer, dense_pre_activation, n_layer, n_unit, dense_weight_temp, dense_bias_temp);
                // Use sigmoidActivationForwardOutput for sigmoid output layer
                softmaxActivationForwardOutput(dense_layer, output_layer, output_weight_temp, output_pre_activation, n_output_unit,
                                               n_unit, n_layer);

                // update the desired output Y
                zeroArray(desired_output, n_output_unit);
                desired_output[(size_t)dataset.test_labels.at(curr_sample)] = 1.; // change test label into train label to have more data
                cost -= log(output_layer[(size_t)dataset.test_labels.at(curr_sample)]);
                // Get the output layer errors for the current sample
                getCatCrossEntropyDeltaOutput(output_layer, delta_output, desired_output, n_output_unit);
                // Start back propagating the full delta matrix using the output error for the current sample
                sigmoidBackPropagate(delta_dense, dense_weight_temp, dense_pre_activation, delta_output, output_weight_temp, n_layer,
                                     n_unit, n_output_unit);

                /***************************************************************************************/
                /************************-LEARNING THROUGH GRADIENT DESCENT-****************************/
                /***************************************************************************************/

                // Start gradient descent to change the weights and biases accumulated through each sample, order does not matter.
                // Note that we use output_weight_temp, dense_weight_temp, etc in each sample forward back propagation as
                // those matrices are not modified within each batch calculation. One batch is one learning experience for all the samples.
                gradientDescentOutputWeight(output_weight, delta_output, dense_layer, n_layer, n_unit, n_output_unit, n_sample);
                gradientDescentDenseWeight(dense_weight, delta_dense, dense_layer, n_layer, n_unit, n_sample);
                gradientDescentDenseBias(dense_bias, delta_dense, n_layer, n_unit, n_sample);
                gradientDescentInputWeight(input_weight, delta_dense, input_layer, n_unit, n_sample);
            }

            auto stop = high_resolution_clock::now();
            // Get duration. Substart timepoints to
            // get duration. To cast it to proper unit
            // use duration cast method
            auto duration = duration_cast<milliseconds>(stop - start);
            if (curr_batch > 1)
            { // First iter is warm up
                totalTime += (float)duration.count() / 1000.;
            }
            printf("In the %dth epoch, Neuro-Sama finished studing the %dth batch in %f seconds. The grind rate is %f, the cost is %f.\n", (int)curr_epoch, (int)curr_batch, 
                (float)duration.count() / (float)1000., (float)1000/(float)duration.count(), cost);
        }
    }

    /***************************************************************************************/
    /*********************************-TESTING RESULTS-*************************************/
    /***************************************************************************************/
    float avgTime = totalTime / (float)(n_epoch * epoch_sample_size / (float)n_sample - 1); 
    printf("Total time is %fs. The average time for each batch is %fs. Average grind rate is %f.\n", totalTime, avgTime, 1/avgTime);

    bool continue_test = true;
    string input;
    int curr_sample;
    while (continue_test)
    {
        cout<<"Enter the sample # you want to test: ";
        cin>>input;
        curr_sample = stoi(input);
        cout<<"\nThinking....\n";
        for(size_t i=0; i<N_INPUT_UNIT; i++){
            input_layer[i]=(float)dataset.test_images.at(curr_sample).at(i);;
        }

        // Forward propagation through three regions
        sigmoidActivationForwardInput(dense_layer, dense_pre_activation, input_layer, n_unit, input_weight, dense_bias);
        sigmoidActivationForwardDense(dense_layer, dense_pre_activation, n_layer, n_unit, dense_weight, dense_bias);
        softmaxActivationForwardOutput(dense_layer, output_layer, output_weight, output_pre_activation, 
                                    n_output_unit, n_unit, n_layer);

        printf("Result after learning for digit %d: ", dataset.test_labels.at(curr_sample));
        printArray(output_layer, n_output_unit);
        cout<<"Continue? (y/n): ";
        cin>>input;
        if(input != "y") continue_test = false;
    }
    

    /***************************************************************************************/
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
