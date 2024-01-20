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
The following instructions reflect our recommended way to setup CMME on macOS. You may adapt it to fit your requirements and/or technical environment.

Please install following their official guides:
* MATLAB R2023a or R2023b ([https://www.mathworks.com](https://www.mathworks.com))
* R v4.x ([https://cran.rstudio.com](https://cran.r-project.org))
* SBCL v2.x (We recommend installing Homebrew ([https://brew.sh](https://brew.sh)), then execute inside the terminal: `brew install sbcl`; for reference: [https://www.sbcl.org/](https://www.sbcl.org/))
* ZeroMQ (We recommend installing it via Homebrew: `brew install zmq`; for reference: [https://zeromq.org/download/](https://zeromq.org/download/))
* (Mini)conda ([https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/))

To finish setting up R, open R (we recommend launching R's command-line interface in the terminal, i.e. run `R` inside the terminal). Inside R's CLI:
* Install dplyr, tidyr, arrow: `install.packages(c("tidyr", "dplyr", "arrow"))`
* Install PPM-Decay: `if (!require("devtools")) install.packages("devtools"); devtools::install_github("pmcharrison/ppm")`
* Close the CLI: `q()`. If asked, don't save a workspace image.

To finish setting up SBCL, install Quicklisp. Therefore, download [https://beta.quicklisp.org/quicklisp.lisp](https://beta.quicklisp.org/quicklisp.lisp), then open SBCL (we recommend using the terminal, execute: `sbcl`). Then:
* Load Quicklisp, replace PATH (inside the double quotes) with the path to the downloaded file: `(load "FILE")`.
* You will be asked to run `(quicklisp-quickstart:install)`. Please do so.
* Close the CLI: `(quit)`.

Open the terminal at a directory, where you want the repository to be stored, then clone this repository and its submodules: `git clone --recurse-submodules https://github.com/lexngu/cmme.git`. Then open a terminal inside the repository's directory, and run:
* Initalize a new Python environment: `conda create -n cmme-env python=3.10`
* Activate it: `conda activate cmme-env`
* Install the "matlabengine" (see: [Install MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))
* Install CMME's dependencies:
  * First run `pip install pyzmq numpy`
  * Then run `pip install -r requirements.txt`
  * Install our patched version of cl4py: `pip install git+https://github.com/lexngu/cl4py.git` <i>(The patched version implements means to capture the console output of SBCL for online interaction with IDyOM's database)</i>

Finally, setup IDyOM's database:
* Check and edit `cmme/cmme-comparison.ini` in a text editor. Change R_HOME, MATLAB_PATH as needed, replace the username in IDYOM-ROOT and IDYOM_DATABASE with your user account's.
* Inside the terminal (with correctly activated Python environment) open a Python CLI: `python`. Then run:
 * `from cmme.idyom.util import install_idyom; install_idyom()` <br>(This will use the variables IDYOM_ROOT_PATH and IDYOM_DATABASE_PATH from cmme/cmme-comparison.ini)

## Examples
See our [Jupyter notebooks](https://github.com/lexngu/cmme-jupyter).
