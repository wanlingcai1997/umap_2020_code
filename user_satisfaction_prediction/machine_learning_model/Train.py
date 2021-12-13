# Train.py
import numpy as np
import pandas as pandas

from Data import Data
import sklearn_model

class Train():
    
    @staticmethod
    def train():


        sklearn_model.model_train_and_evaluate(Data.model_name, Data.feature_vector, Data.conversation_labels, Data.cross_validation) 
        # sklearn_model.model_train_and_evaluate(Data.model_name, X_clf_new, Data.conversation_labels) 
        

        
        
