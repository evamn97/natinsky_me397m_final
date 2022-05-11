 
# AFM Image Gradient Analysis
The is Eva Natinsky's ME397M-DA final project, which conducts gradient analysis on a set of AFM images. 

	Dataset: AFM scans of square pillars
            --> 20 um square
            --> 128 px resolution
            --> 22 total images


### Tips for running this script:

This project requires the ```pathos``` parallel processing library. It can be installed via ```pip``` with:

    pip install pathos

However, it is recommended to install it from the Github repository:

    pip install git+git://github.com/uqfoundation/pathos@master

If you're running a conda environment be sure to activate it and install both ```pip``` and ```git``` before installing ```pathos```. 

## Running this script on Stampede2@TACC:
```run_main.py``` requires the ```pipenv``` package manager to be installed, which is used instead of conda. 

To install ```pipenv``` on TACC, run:

    pip install --user pipenv

Navigate to your project directory and run: 

    pipenv install

This will create a pipenv virtual environment in the project directory and install dependencies listed in ```requirements.txt```. 
Note that this installs the ```PyPI``` version of ```pathos```, but on TACC this is much easier than installing the Github version, and it still works. 