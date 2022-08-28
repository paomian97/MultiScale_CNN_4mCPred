# MultiScale_CNN_4mCPred
## Abstract  
This project provides a core code for mouse DNA N-4 methylation site prediction. As you can see, the project contains three py files and three folders. The following describes the meaning of each file one by one.    
1.With the Framework_model.py file, the structure of the model can be seen and training examples are provided.  
2.The Predict_model.py file shows the runs of our final model on the test set and the predicted results.  
3.The Data_generate.py file shows how we process the source dataset and stores the pre-processed data in the data_processed folder.  
4.The folder dataset holds the most original dataset.  
5.The folder model_weights holds our model weights.  
6.The folder data_processed holds our preprocessed data.  

# Requirements
* Python3  
* Tensorflow>=2.0  
* numpy==1.18.5 
* sklearn


# Guidance  

## Show model test set results  
### Step1:  
Run the data_generate.py file by： Python Data_generate.py
### Step2:  
Run the Predict_model.py file by:  Python Predict_model.py

## If you want to train the model  
### Step1:  
Run the data_generate.py file by:  
python Data_generate.py  
### Step2:  
You can change the code in Framework_model.py as appropriate, or you can run it directly.  
python Framework_model.py
