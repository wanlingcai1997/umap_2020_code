import numpy as np
import pandas as pd
from Data import Data
import json
import copy
from sklearn.preprocessing import MultiLabelBinarizer
import sys 
sys.path.append("..") 
import load_data
import helper
class ReadData(object):

    @staticmethod
    def readData():
        if Data.file_input_data != None :
            #------------------------------------------------
            helper.print_current_time()
            print("Read Data")
            #------------------------------------------------

            # Store data into a dictionary $D$
            Data.whole_annotation_data_dict = load_data.load_whole_annotation_data_dict(Data.file_input_data)            
            # Store store dictionary data into a DataFrame
            # Store intent labels
            # Store action labels
            Data.whole_utterance_dataframe, Data.intent_labels, Data.action_labels= load_data.load_utterance_data(Data.whole_annotation_data_dict)
            
            #------------------------------------------------            
            helper.print_current_time()
            print("Number of All Utterances: %d:" % len(Data.whole_utterance_dataframe))
            #------------------------------------------------

            # Obtain User Utterance Data and store into a DataFrame 
            Data.user_utterance_dataframe = Data.whole_utterance_dataframe[Data.whole_utterance_dataframe['role'] == 'seeker']
            
            
            #------------------------------------------------
            helper.print_current_time()
            print("Number of User Utterances: %d:" % len(Data.user_utterance_dataframe))
            #------------------------------------------------
        
            # Store intent/action labels into two dict with index for convenient use 
            Data.intent_label_indext_dict, Data.intent_indext_label_dict = load_data.construct_label_index_dict(Data.intent_labels)
            Data.action_label_indext_dict, Data.action_indext_label_dict = load_data.construct_label_index_dict(Data.action_labels)
            
            #------------------------------------------------            
            helper.print_current_time()
            print("Intent Labels (%d):" % len(Data.intent_labels), end=' ')
            print(Data.intent_labels)
            helper.print_current_time()
            print("Action Labels (%d):" % len(Data.action_labels), end=' ')
            print(Data.action_labels)
            #------------------------------------------------            

            
            # initialize label data for a more comfortable usage
            Data.labels = Data.user_utterance_dataframe[['labels']].values
            Data.labels = [label_list[0] for label_list in Data.labels]
            mlb = MultiLabelBinarizer(classes=Data.intent_labels)
            Data.labels = mlb.fit_transform(Data.labels) 
            helper.print_current_time()
            #------------------------------------------------            
            print("Binarize the multi-label for label data")
            print(mlb.get_params)
            #------------------------------------------------            
