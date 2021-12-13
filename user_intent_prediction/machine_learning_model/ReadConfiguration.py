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
        parser.add_argument('--discourse_features', type=int, default=1, help='whether or not to use structural_features (0/1)')
        parser.add_argument('--sentiment_features', type=int, default=1, help='whether or not to use sentiment_features (0/1)')
        parser.add_argument('--conversational_features', type=int, default=1, help='whether or not to use sentiment_features (0/1)')
        parser.add_argument('--num_previous_turns', type=int,  default=3, help='fixed num_previous_turns')

        
        # feature normalization
        parser.add_argument('--feature_normalization', type=int, default=0, help='whether or not to operate feature scaling')
        
        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------

        parser.add_argument('--neural_model', type=int, help='is it a neural_model')    
        parser.add_argument('--algorithm_adaption', type=int, default=0,  help='is it a algorithm adaption method')
        parser.add_argument('--problem_transformation', type=int, default=1, help='is it a problem_transformation method')
        parser.add_argument('--problem_transformation_method', default='CC', nargs='?', help='what kind of problem_transformation method')
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


        # -----------------------------------------------------------------------------------------------
        # ------                       Feature Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        Data.content_features = args.content_features
        Data.discourse_features = args.discourse_features
        Data.sentiment_features = args.sentiment_features
        Data.conversational_features = args.conversational_features
        Data.num_previous_turns = args.num_previous_turns

        Data.feature_normalization = args.feature_normalization

        # -----------------------------------------------------------------------------------------------
        # ------                         Model Related Configuration                               ------                                 
        # -----------------------------------------------------------------------------------------------
        
        Data.neural_model = args.neural_model
        Data.algorithm_adaption = args.algorithm_adaption
        Data.problem_transformation = args.problem_transformation
        Data.problem_transformation_method = args.problem_transformation_method
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
		
        print ('content_features: %d' % Data.content_features)
        print ('discourse_features: %d' % Data.discourse_features)
        print ('sentiment_features: %d' % Data.sentiment_features)
        print ('conversational_features: %d' % Data.conversational_features)
        print ('num_previous_turns: %s' % Data.num_previous_turns)
		
        

        print ('feature_normalization: %d' % Data.feature_normalization)
        
        
        print ('neural_model: %d' % Data.neural_model)
        print ('algorithm_adaption: %d' % Data.algorithm_adaption)
        print ('problem_transformation: %d' % Data.problem_transformation)
        print ('problem_transformation_method: %s' % Data.problem_transformation_method)
        print ('model_name: %s' % Data.model_name)



        print ('cross_validation: %d' % Data.cross_validation)
