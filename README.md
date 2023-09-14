# Computational Models of Musical Expectation – Comparison Environment
This Python3-library provides an environment to run models, to collect their output, and to generate plots for visual inspection. 

Currently supported models: 
* PPM-Decay (Harrison et al., 2020): Both "simple", and "decay" version. 
* D-REX (Skerritt-Davis & Elhilali 2018, 2020, 2021a, 2021b): All distributions (Gaussian, GMM, Lognormal, Poisson), multi-feature input.
* IDyOM (Pearce, 2005, 2018): Data import, parameterizing and running IDyOM, parsing results. 

The Jupyter notebooks are part of another repository: https://github.com/lexngu/cmme-jupyter.

## Actual requirements
For running D-REX:
* MATLAB R2021b or newer

For running PPM-Decay:
* R v4.x

For running IDyOM:
* Steelbank Common Lisp (SBCL)

For the comparison environment:
* Any Python environment manager, e.g. Anaconda
* Python 3.10

## Installation
* Install MATLAB, R, SBCL, Anaconda according to official guides (Note: On macOS, you may want to install SBCL via Brew or MacPorts though)
* Clone the cmme-jupyter repository and its submodules: `$ git clone git@github.com:lexngu/cmme-jupyter --recursive`
* Change settings in `cmme/cmme-comparison.ini`, if needed.
* Open the terminal, and go to the directory containing the cloned repository (`$ cd path_to_cmme`).
  * Initalize a new Python environment (here, using Anaconda):
    * Create: `$ conda create -n cmme python=VERSION` (VERSION must match the requirements of MATLAB (e.g., for 2022a: 3.8-3.10))
    * Activate: `$ conda activate cmme`
    * Install jupyter: `$ pip install jupyter`
    * Install cmme's dependencies: `$ pip install -r requirements.txt`
    * Install matlab-engine within this Python environment (see: [Install MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))
    * Install cmme and cl4py: `$ pip install -e . res/packages/cl4py` (Note: this is a preliminary solution during development of CMME) 
* Open R, and run: 
  * Install dplyr, tidyr: `install.packages(c("tidyr", "dplyr", "arrow"))`
  * Install PPM-Decay model: `if (!require("devtools")) install.packages("devtools"); 
devtools::install_github("pmcharrison/ppm")`
* Run `$ python`, then: `import cmme.idyom.binding; cmme.idyom.binding.install_idyom(IDYOM_ROOT_PATH)` (replace IDYOM_ROOT_PATH with the desired value, e.g. `/Users/alexander/idyom/`)
* Update `cmme/cmme-comparison.ini`, if needed.

## Examples
See Jupyter notebooks.