# Serial Version

## Running with the mnist.h header to extract test/training image set
Need to have all the relevant training files in the "data" subdirectory, such as "data/train-images-idx3-ubyte"

## Makefile
In the makefile neuro_serial.cpp is compiled into an executable "serial_neuro", and without input parameters it will have preset parameters.

The order of the input parameters are: ./serial_neuro nl nh ne nb

## Change training set and size of training set
To change the set from the "test" set to "train" set, change ```dataset.test_images``` to ```dataset.training_images``` and ```dataset.test_labels``` to ```dataset.training_labels```.

Change ```epoch_sample_size``` to change the total amount of images to be studies.
