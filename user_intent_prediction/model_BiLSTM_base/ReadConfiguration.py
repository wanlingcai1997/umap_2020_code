# ReadConfiguration.py
from Data import Data
import argparse

class ReadConfiguration(object):
    
    @staticmethod
    def readConfiguration():
        parser = argparse.ArgumentParser(description='Read some configuration.....')
        # -----------------------------------------------------------------------------------------------
        # ------                                   Data                                            ------
        # -----------------------------------------------------------------------------------------------

        parser.add_argument('--file_input_data', nargs='?', help='input annotation data')

        # -----------------------------------------------------------------------------------------------
        # ------                       Feature Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------

        # feature construction
        parser.add_argument('--content_features', type=int, default=1, help='whether or not to use content_features (0/1)')
        parser.add_argument('--structural_features', type=int, default=1, help='whether or not to use structural_features (0/1)')
        parser.add_argument('--sentiment_features', type=int, default=1, help='whether or not to use sentiment_features (0/1)')
        parser.add_argument('--conversational_features', type=int, default=1, help='whether or not to use sentiment_features (0/1)')
        
        # feature normalization
        parser.add_argument('--feature_normalization', type=int, default=0, help='whether or not to operate feature scaling')
        
        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------

        parser.add_argument('--model_name', nargs='?', help='input model name')
        parser.add_argument('--max_sequence_length', type=int, default=50, help='the maximum number of sequence length for training')
        parser.add_argument('--max_num_vocabulary', type=int, default=20000, help='the maximum number of vocabulary for tokenization')
        parser.add_argument('--embedding_selection', nargs='?', help='embedding vector from (1) word2vec or (2) Glove')
        parser.add_argument('--embedding_dimension', type=int, default=100, help='the number of convolutional unit in the convolutional layer')
        

        

        #  --------------------------------- Model Parameters -------------------------------------------
        parser.add_argument('--lstm_units', type=int, default=1024, help='the number of convolutional unit in the convolutional layer')
        parser.add_argument('--dropout_rate', type=float, default=0.6, help='the dropout rate in the dropout layer')
        parser.add_argument('--dense_units', type=int, default=256, help='the number of dense unit in the fully-connected layer')
        

        # -----------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------
        # ------                                    Evaluation                                     ------                                 
        # -----------------------------------------------------------------------------------------------

        # cross-validation
        parser.add_argument('--cross_validation', type=int, default=3, help='k-fold cross_validation')



      
        # ===============================================================================================
        # ====================          Store Configurations    =========================================                          
        # ===============================================================================================                          


        args = parser.parse_args()

        # -----------------------------------------------------------------------------------------------
        # ------                                   Data                                            ------
        # -----------------------------------------------------------------------------------------------
        Data.file_input_data = args.file_input_data


        # -----------------------------------------------------------------------------------------------
        # ------                       Feature Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        Data.content_features = args.content_features
        Data.structural_features = args.structural_features
        Data.sentiment_features = args.sentiment_features
        Data.conversational_features = args.conversational_features

        Data.feature_normalization = args.feature_normalization

        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        
        Data.model_name = args.model_name
        
        Data.max_sequence_length = args.max_sequence_length
        Data.max_num_vocabulary = args.max_num_vocabulary
        Data.embedding_selection = args.embedding_selection
        Data.embedding_dimension = args.embedding_dimension
       



        #  --------------------------------- Model Parameters -------------------------------------------
        Data.lstm_units = args.lstm_units
        Data.dropout_rate = args.dropout_rate
        Data.dense_units = args.dense_units
  

        # -----------------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------------
        # ------                                    Evaluation                                     ------                                 
        # -----------------------------------------------------------------------------------------------


        Data.cross_validation = args.cross_validation


        
        # ===============================================================================================
        # ====================          Print Configurations    =========================================                          
        # ===============================================================================================

      
        
        
        print ('-------------------------------------------')
        print ('------------ Configurations ---------------')
        print ('-------------------------------------------')

        print ('file_input_data: %s' % Data.file_input_data)
		
        print ('content_features: %d' % Data.content_features)
        print ('structural_features: %d' % Data.structural_features)
        print ('sentiment_features: %d' % Data.sentiment_features)
        print ('conversational_features: %d' % Data.conversational_features)


        print ('feature_normalization: %d' % Data.feature_normalization)
        
        
        print ('model_name: %s' % Data.model_name)
        print ('max_sequence_length: %s' % Data.max_sequence_length)
        print ('max_num_vocabulary: %s' % Data.max_num_vocabulary)
        print ('embedding_selection: %s' % Data.embedding_selection)
        print ('embedding_dimension: %s' % Data.embedding_dimension)
        
        print ('lstm_units: %s' % Data.lstm_units)
        print ('dropout_rate: %s' % Data.dropout_rate)
        print ('dense_units: %s' % Data.dense_units)
        

        print ('cross_validation: %d' % Data.cross_validation)
