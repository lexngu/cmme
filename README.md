# Computational Models of Musical Expectation – Comparison Environment
This Python3-library provides an environment to run models, and to collect their output.

Currently supported models: 
* PPM-Decay (Harrison et al., 2020): Both "simple", and "decay" version. 
* D-REX (Skerritt-Davis & Elhilali 2018, 2020, 2021a, 2021b): All distributions (Gaussian, GMM, Lognormal, Poisson), multi-feature input.
* IDyOM (Pearce, 2005, 2018): Data import, parameterizing and running IDyOM, parsing results. 

https://github.com/lexngu/cmme-jupyter contains Jupyter notebooks, which show basic examples on how to use CMME.

## Requirements
CMME does not re-implement the models, but invokes them in their respective environment. Therefore, these softwares are required:

For running D-REX:
* MATLAB R2021b or newer

For running PPM-Decay:
* R v4.x

For running IDyOM:
* Steelbank Common Lisp (SBCL) v2.x

For the comparison environment:
* A Python 3 environment manager (e.g., Miniconda)
* Python v3.10 (v3.7+ may also work)

## Installation
* Install MATLAB, R, SBCL, and the virtual environment manager according to the official guides <br>(Note: On macOS, Homebrew/MacPorts may provide more recent precompiled releases of SBCL)
* If you want to also use Jupyter notebooks, clone the cmme-jupyter repository and its submodules: `$ git clone github.com/lexngu/cmme-jupyter.git --recursive`
* Open the terminal, and go to the directory containing CMME (`$ cd PATH_TO_CMME`).
  * Initalize a new Python environment (here, using Miniconda):
    * Create: `$ conda create -n cmme-env python=VERSION` <br>(VERSION must match the requirements of MATLAB (e.g., Python v3.10 for R2022b and newer, or Python v3.8 for R2020b and newer)
    * Activate: `$ conda activate cmme-env`
    * Install cmme's dependencies: `$ pip install -r requirements.txt`
    * Install matlab-engine within this Python environment (see: [Install MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))
    * Install a patched version of cl4py: `$ pip install git+https://github.com/lexngu/cl4py.git` <i>(The patched version implements means to capture the console output of SBCL for online interaction with IDyOM's database)</i>
* Open R, then: 
  * Install dplyr, tidyr: `install.packages(c("tidyr", "dplyr", "arrow"))`
  * Install PPM-Decay: `if (!require("devtools")) install.packages("devtools"); devtools::install_github("pmcharrison/ppm")`
* For the following step (initializing IDyOM's database), check and edit `cmme/cmme-comparison.ini` as needed.
* Run Python, then: 
  * Initalize IDyOM's database `from cmme.idyom.util import install_idyom; install_idyom()` <br>(This will use the variables IDYOM_ROOT_PATH and IDYOM_DATABASE_PATH inside cmme-comparison.ini)

## Examples
See Jupyter notebooks.