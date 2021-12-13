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
        parser.add_argument('--num_previous_turns', type=int,  default=3, help='extract fixed num_previous_turns')

        # -----------------------------------------------------------------------------------------------
        # ------                       Feature Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------

        # feature construction
        parser.add_argument('--content_features', type=int, default=1, help='whether or not to use content_features (0/1)')
        parser.add_argument('--discourse_features', type=int, default=1, help='whether or not to use discourse_features (0/1)')
        parser.add_argument('--sentiment_features', type=int, default=1, help='whether or not to use sentiment_features (0/1)')
        parser.add_argument('--conversational_features', type=int, default=1, help='whether or not to use conversational_features (0/1)')
        parser.add_argument('--task_specific_features', type=int, default=1, help='whether or not to use task_specific_features (0/1)')
        
        # feature normalization
        parser.add_argument('--feature_normalization', type=int, default=0, help='whether or not to operate feature scaling')
        
        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------

        parser.add_argument('--model_name', nargs='?', help='input model name')

        #  --------------------------------- Model Parameters -------------------------------------------

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
        Data.num_previous_turns = args.num_previous_turns

        # -----------------------------------------------------------------------------------------------
        # ------                       Feature Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        Data.content_features = args.content_features
        Data.discourse_features = args.discourse_features
        Data.sentiment_features = args.sentiment_features
        Data.conversational_features = args.conversational_features
        Data.task_specific_features = args.task_specific_features


        Data.feature_normalization = args.feature_normalization

        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        
        Data.model_name = args.model_name

        #  --------------------------------- Model Parameters -------------------------------------------

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
        print ('num_previous_turns: %s' % Data.num_previous_turns)
		
        print ('content_features: %d' % Data.content_features)
        print ('discourse_features: %d' % Data.discourse_features)
        print ('sentiment_features: %d' % Data.sentiment_features)
        print ('conversational_features: %d' % Data.conversational_features)
        print ('task_specific_features: %d' % Data.task_specific_features)


        print ('feature_normalization: %d' % Data.feature_normalization)
        
    
        print ('model_name: %s' % Data.model_name)



        print ('cross_validation: %d' % Data.cross_validation)
