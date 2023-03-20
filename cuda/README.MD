# CUDA version of Neuro
## Compilation and Running
Use make to compile to the gpu_neuro executable

Need all the relevant "ubyte" files in "data/" subdirectory!

## Change training set and size of training set
To change the set from the "test" set to "train" set, change ```dataset.test_images``` to ```dataset.training_images``` and ```dataset.test_labels``` to ```dataset.training_labels```.

Change ```epoch_sample_size``` to change the total amount of images to be studies.
