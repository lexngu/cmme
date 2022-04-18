# Computational Models of Musical Expectation – Comparison Environment
This Python3-library provides the environment to run models and to collect their output. 

Currently supported models: 
* PPM-Decay (Harrison et al., 2020) 
* D-REX (Skerritt-Davis & Elhilali, 2018; 2020; 2021a; 2021b)

## Requirements
Installed software and packages:
* Python 3.7, incl.
  * rpy2 v3.x
  * pymatbridge
  * pandas
  * numpy
  * scipy
  * jupyter
* MATLAB R2021b
* R v4.x

Notes for future work:
* Ensure usability in IPython environments?
* Use Matplotlib for visualization (instead of MATLAB plots)?
* Use "MATLAB engine" for interfacing with MATLAB (instead of pymatbridge)?
* Use Python type hints?

## Installation
* Install MATLAB
* Install Python and packages, e.g. inside a Conda environment (e.g. called "cmme": $ conda create -n cmme python=3.7 && conda activate cmme)
  * Install jupyter: $ conda install jupyter
  * Within, install dependencies: $ pip install rpy2 pandas numpy scipy zmq pymatbridge
  * Regarding pymatbridge, you may need to run further steps: https://github.com/arokem/python-matlab-bridge#building-the-pymatbridge-messenger-from-source 
* In R: Install ppm, dplyr, tidyr

## Use
Software design:
* The models are processed in their respective programming language: PPM-Decay in R, D-REX in MATLAB. 
* This environment creates input files (R: .csv, MATLAB: .mat) containing the necessary instructions and triggers the models' processing.
* After processing, the models' output files are parsed and aggregated for comparison. 

Python modules:
* **model_io** : This module generates path strings that are used by other modules to organize the locations of generated files (e.g., input parameters, output results, plot figures).
* **matlab_worker** : This module handles the communication with MATLAB through pymatbridge. In particular, it automatically starts and stops MATLAB instances and triggers D-REX's processing.
* **drex** : This module contains various type objects that represent all of D-REX's input parameters and output data. It also contains a builder.
* **ppm** : This module contains various typoe objects that represent all of PPM-Decay's input parameters and output data. It also contains a builder.
* **model_output_aggregator** : This module parses models' output and aggregates it in one single data frame.
* **model_output_plot** : This module uses the aggregated data frame and plots the data for visual comparison.

Examples:
* See Jupyter notebooks
