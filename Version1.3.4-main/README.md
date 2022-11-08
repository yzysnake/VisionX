# Software version1.3.3

Update on Version1.3.2: added algorithm choice, stop training, calendar button

Part1: Setting Environment Install Anaconda: Go to Anaconda installation page: ​https://www.anaconda.com/distribution/#download-section

Follow the instructions on the following link to install Anaconda: https://docs.anaconda.com/anaconda/install/

One reason we recommend using Anaconda is that you can create virtual environments for your demand.

Each virtual environment will not affect each other.

Set virtual environment for ​CPU-only

TensorFlow: Run the following commands in terminal:

$ conda create -n tf_cpu tensorflow python=x.x (python part is optional) 

$ conda activate tf_cpu

For deactivate your virtual environment:

$ conda deactivate

Install required Modules:

In Anaconda and your virtual environment, using “​conda install​” to install all the package. (Modules includes: matplotlib, pandas, PIL, keras, sklearn, alpha_vantage)

$ conda install matplotlib

$ conda install python=3.6.4 // older version of python is necessary for installing older versions of pandas and tensorflow

$ conda install tensorflow=1.15.0

$ conda install pandas=0.24 // the newest version is not compatible

$ conda install pillow

$ conda install keras

$ conda install scikit-learn

$ conda install requests

$ pip install alpha_vantage

$ pip install tkcalendar

Part 2: Running the Program Run the Program file: Use the following commands in the terminal to run the main program file. Change directory to file path :

$ cd

Add a output folder: mkdir output // IMPORTANT!!!

Run file: $ python3

Example: $ cd /Users/jing/Desktop/Version1.1/ $ python3 Stock_Prediction_LIVE_ver1.1.py

Show User Interface: The interface contains a search bar for stock input. Input Stock Symbol: Example : “AAPL” for apple, “​GOOGL” for google, “AMZN” for amazon. Output: plot of 90- days stock price data.

Train the Model: Click “train the model” to train current model using stock data. Training Progress: The training progress takes about 10 minutes.

Input Predict Date and Click “Predict”: Input Date should follow the format: %Y-%m-%d (eg. 2019-09-05), and weekends and holidays are invalid date. Prediction can only be made for the next unknown closing price. Predictions further into the future is not allowed due to poor accuracy using the current model. The current model is a simple multi-layer LSTM. Click “Plot Chart” to Plot Chart: The output chart indicates predicted trends.

Adjust Percent and Click “Re-Plot”:

Click Proof and Show Proof:

Show actual closing price and accuracy.
