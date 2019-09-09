# Hands on Tensorflow 2.0

This repository gathers some pieces of code I did to get used to the new tensorflow 2.0 API.
It contains:
* 
* `biorules_segmentation.py`: a try at a segmentation based solely on biological rules.
* `chan_vese.py`: a tensorflow implementation of the Chan-Vese segmentation algorithm. Runs considerably faster than the sklearn's implementation on big samples.  
* `circles_cvae.py`: some work on CVAE, using a simple circle generator for practice.
* `custom_flow.py`: a failed try at designing a new flow based on biological rules.
* `cvae.py`: a class to build a VAE from scratch.
* `cycle_gan.py`: some utility functions to get a cycle gan running rapidly.
* `ellipses_to_cell_gan.py`: a try at generating cells-like mass cytometry samples, prior to usual supervised training.
* `encoder_decoder.py`: some functions and classes on encoder and decoder models and layers.
* `environment.yml`: the conda environment used. 
* `sample_train_loop.py`: a sample train loop which can be reused to train other models.