# Visualizing Word Vector Biases

<hr>

### Requirements
The web interface is supported only for Chrome and Firefox.

Python 3.6+, pip

The following libraries are also required to run the code:
```
flask
sklearn
scipy
numpy
networkx
tqdm
```

To install these libraries using pip, use the following command in the terminal:
```
pip install flask scikit-learn scipy numpy networkx tqdm
```
To install these packages only for current user (or if you do not write access to the python installation on the machine):
```
pip install flask scikit-learn scipy numpy networkx tqdm --user
```

Alternately, you can also use conda to install the packages:
```
conda install flask scikit-learn scipy numpy networkx tqdm
```

### Installation
Clone this repository to your local machine, make sure the requirement are installed. 
Then navigate to the cloned repository and in the base directory, type the following
command in the terminal.
```shell script
git clone https://github.com/architrathore/visualizing-bias.git
cd <repo-location>
python -m flask run
```

### Common known issues and fixes

#### Error: Could  not locate a Flask application on command `python -m flask run`
If you get the following error, it might indicate that you are not in the correct directory. Open a terminal in the 
base directory of the cloned repo
```
Error: Could not locate a Flask application. You did not provide the
"FLASK_APP" environment variable, and a "wsgi.py" or "app.py" module
was not found in the current directory
```


#### Python version
The tool is written with Python 3.6+ support, and may/may not work with earlier versions of Python3.x. It will not 
work with Python 2.x

#### Missing libraries
Make sure you install the requirements before running the application. 

#### Application runs but UI is mangled in the browser
We have tested the tool on Firefox and Chrome. There are known issues of point labels not showing correctly in Safari.

#### I get the following error: "Something went wrong! Could not find the key 'xxxx'
This error means that one of the words (denoted by 'xxxx' above) in your provided word set was not found the 
vocabulary of the word vector embedding. Check the spelling, or use another common word instead. 

