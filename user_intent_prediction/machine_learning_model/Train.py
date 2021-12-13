# Train.py
from Data import Data
import sklearn_model

class Train():
    
    @staticmethod
    def train():
        
        X = Data.feature_vector
        Y = Data.labels
        # print(Data.labels)

        if Data.algorithm_adaption == 1:
            if Data.model_name == 'ML_kNN':
                sklearn_model.ML_kNN(X, Y, Data.cross_validation)
            if Data.model_name == 'BR_kNN':
                sklearn_model.BR_kNN(X, Y, Data.cross_validation)
        if Data.problem_transformation == 1:
            if Data.problem_transformation_method == 'BR':
                sklearn_model.BR_model(X, Y, Data.model_name, Data.cross_validation)
            if Data.problem_transformation_method == 'LP':
                sklearn_model.LP_model(X, Y, Data.model_name, Data.cross_validation)
            if Data.problem_transformation_method == 'CC':
                sklearn_model.CC_model(X, Y, Data.model_name, Data.cross_validation)

        
