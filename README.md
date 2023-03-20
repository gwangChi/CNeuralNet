# Convolutional Neural Network for recoganizing digits in 728 pixels
## Compilation and running the program
Use ```make``` to generate the executables. To run the program, make sure you have the MNIST training data under the ```data/``` subdirectory.
## Training the neural net
Uses MNIST training images. You need to set the number of files to be used for each training epoch by changing the ```epoch_sample_size``` parameter in ```main()```.
e.g. Setting ```epoch_sample_size = 200``` will only let each epoch learn the first 200 images from the training set.
