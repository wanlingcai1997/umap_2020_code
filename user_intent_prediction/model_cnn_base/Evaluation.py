# Train.py
import numpy as np
import pandas as pandas
from Data import Data
from BuildModel import BuildModel
import copy
import sys
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping

import sklearn.metrics as metrics

sys.path.append("..") 
from evaluation import hamming_score


def evaluation_score(label_test, predict_label):
    f1_micro=metrics.f1_score(label_test, predict_label, average='micro')
    hamm_loss=metrics.hamming_loss(label_test,predict_label)
    hamm_score= hamming_score(label_test, predict_label)
    accuracy = metrics.accuracy_score(label_test, predict_label)
    precision = metrics.precision_score(label_test, predict_label,average='micro') 
    recall=metrics.recall_score(label_test, predict_label,average='micro')
    zero_one_loss= metrics.zero_one_loss(label_test, predict_label)

    
    
    print("f1_micro :", round(f1_micro,4))
    print("hamming_loss :", round(hamm_loss,4))
    print("hamming_score :", round(hamm_score,4))
    print("accuracy :",  round(accuracy,4))
    print("precision :",  round(precision,4))
    print("recall :", round(recall,4))
    print("zero_one_loss :", round(zero_one_loss,4))

    return 

class Evaluation():
    
    @staticmethod
    def evaluate():

        predicted_test = Data.trained_model.predict(np.array(Data.x_test))

        for threshold in [0.5]:

            pred = copy.deepcopy(predicted_test)
            print(threshold)
    
            for i in range(pred.shape[0]):
                if len(np.where(pred[i] >= threshold)[0]) > 0:
                    pred[i][pred[i] >= threshold] = 1
                    pred[i][pred[i] < threshold] = 0
                else:
                    max_index = np.argmax(pred[i])
                    pred[i] = 0
                    pred[i][max_index] = 1

        evaluation_score(Data.y_test,pred)





        
        return