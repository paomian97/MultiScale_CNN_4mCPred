# MultiScale_CNN_4mCPred
Abstract：  
1.This project provides three py files, Framework_model, Predict_model and Data_generate respectively.  
2.With the Framework_model.py file, the structure of the model can be seen and training examples are provided.  
3.The Predict_model.py file shows the runs of our final model on the test set and the predicted results.  
4.The Data_generate.py file shows how we process the source dataset and stores the pre-processed data in the data_processed folder.  
5.The folder dataset holds the most original dataset.  
6.The folder model_weights holds our model weights.  

# Requirements
Python3  
Tensorflow>=2.0  
numpy==1.18.5  

# Guidance

## Show model test set results
### Step1:
Run the data_generate.py file by： Python Data_generate.py
### Step2:
Run the Predict_model.py file by: Python Predict_model.py

## If you want to train the model
### Step1:
Run the data_generate.py file by： python Data_generate.py
### Step2:
You can change the code in Framework_model.py as appropriate, or you can run it directly.
python Framework_model.py
