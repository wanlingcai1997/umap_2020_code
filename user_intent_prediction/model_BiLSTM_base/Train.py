# Train.py
import numpy as np
import pandas as pandas
from Data import Data
from BuildModel import BuildModel
import copy
import sys
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping


class Train():
    
    @staticmethod
    def train():
        
        # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model = BuildModel.BiLSTM_base_model()


        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

        print(Data.x_train.shape)
        history = model.fit(Data.x_train, Data.y_train,
            batch_size=128,
            epochs=100,
            callbacks=[es],
            validation_data=(Data.x_val, Data.y_val))
    
        Data.trained_model = model

        

        
        return