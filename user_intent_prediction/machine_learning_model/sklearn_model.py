import numpy as np
from skmultilearn.adapt import MLkNN
from skmultilearn.adapt import BRkNNaClassifier

from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pylab as plt
from matplotlib import pyplot

from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import time
import evaluation
import sys
sys.path.append("..") 
import helper
# import evaluation as evaluate2




def cross_validation_score(best_model, X, label, cv):
    print(best_model)
    hamming_loss = metrics.make_scorer(metrics.hamming_loss)
    hamming_score = metrics.make_scorer(evaluation.hamming_score)
    zero_one_loss = metrics.make_scorer(metrics.zero_one_loss)
    cohen_kappa_score = metrics.make_scorer(evaluation.cohen_kappa_score)
    scoring = {'f1_micro':'f1_micro', 'hamming_loss': hamming_loss, 'hamming_score':hamming_score, 'accuracy':'accuracy', 'precision':'precision_micro',  'recall': 'recall_micro', \
        'zero_one_loss':zero_one_loss, 'cohen_kappa_score': cohen_kappa_score, 'f1_macro':'f1_macro', 'precision_macro':'precision_macro',  'recall_macro': 'recall_macro', }
    # scoring = {'f1_micro':'f1_micro', 'hamming_loss': hamming_loss, 'accuracy':'accuracy', 'precision':'precision_micro',  'recall': 'recall_micro', 'zero_one_loss':zero_one_loss}
    scores = cross_validate(best_model, X, label, cv=cv, scoring=scoring, return_train_score=True)
    
    print(scores)

    print("f1_micro :", abs(scores['test_f1_micro'].mean()))
    print("hamming_loss :", abs(scores['test_hamming_loss'].mean()))
    print("hamming_score :", abs(scores['test_hamming_score'].mean()))
    print("accuracy :", abs(scores['test_accuracy'].mean()))
    print("precision :", abs(scores['test_precision'].mean()))
    print("recall :", abs(scores['test_recall'].mean()))
    print("zero_one_loss :", abs(scores['test_zero_one_loss'].mean()))
    print("cohen_kappa_score :", abs(scores['test_cohen_kappa_score'].mean()))
    print("f1_macro :", abs(scores['test_f1_macro'].mean()))
    print("precision_macro :", abs(scores['test_precision_macro'].mean()))
    print("recall_macro :", abs(scores['test_recall_macro'].mean()))

    return 

# Algortihm Adaption (1) - Multi-label K Nearest Neighbours

def ML_kNN (X, label, cv):

   

    print(" ==============================================")
    print(" ML_kNN")
    print(" ==============================================")

    parameters = {'k': range(1,10), 's': [0.1,0.3,0.5,0.7,0.9]}
    score = 'accuracy'

    start=time.time()

    # X = np.row_stack((X_train, X_test))
    # label = np.row_stack((label_train, label_test))

    classifier = GridSearchCV(MLkNN(), parameters, scoring=score, cv=cv)
    classifier.fit(X, label)
    
    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', classifier.best_params_, 'best score: ', classifier.best_score_)

   
    # set up best model
    best_MLkNN = MLkNN(k= classifier.best_params_['k'], s=classifier.best_params_['s'] )
    
    # evaluation
    cross_validation_score(best_MLkNN, X, label, cv)
    
    return 



# Algorithm Adaption (2) - Binary Relevance k-Nearest Neighbours

def BR_kNN(X, label, cv):
    
    print(" ==============================================")
    print(" BR_kNN")
    print(" ==============================================")


    parameters = {'k': range(1,10)}
    score = 'accuracy'

    start=time.time()

    classifier = GridSearchCV(BRkNNaClassifier(), parameters, scoring=score, cv=cv)
    classifier.fit(X, label)

    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', classifier.best_params_, 'best score: ',classifier.best_score_)


    # set up best model
    best_BRkNN = BRkNNaClassifier(k= classifier.best_params_['k'])

    # evaluation
    cross_validation_score(best_BRkNN, X, label, cv)
    
    return 

def match_classifier (model_name):
    classifier_and_parameters = None
    
    if model_name == 'LR':
        classifier_and_parameters = {
            'classifier': [LogisticRegression()],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'classifier__max_iter': [10, 100, 1000],
            'classifier__penalty': ['l2', 'l1']
        }
    if model_name == 'NB_M':
        classifier_and_parameters = {
            'classifier': [MultinomialNB()],
            'classifier__alpha': [0.2,0.4,0.6,0.8,1]
        }
    if model_name == 'LinearSVC':
        classifier_and_parameters = {
            'classifier': [LinearSVC()],
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'classifier__max_iter': [10, 100, 1000, 2000]
        }
    if model_name == 'MLP':
        classifier_and_parameters = {
            'classifier': [MLPClassifier(early_stopping=True)],
            # 'classifier__hidden_​​layer_sizes': [(64, 16), (128, 16), (256, 16), (512, 16)],
            # 'classifier__hidden_​​layer_sizes': [64, 128, 256, 512],
            # 'classifier__hidden_​​layer_sizes': (64,),
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'classifier__max_iter': [10, 100, 1000, 2000],
        }


    if model_name == 'DT':
        classifier_and_parameters = {
            'classifier': [DecisionTreeClassifier(class_weight='balanced')],
            'classifier__max_depth': [5, 10, 20, 30, 40, None],
            'classifier__min_samples_split': [2, 5, 10, 15, 20]
        }
    if model_name == 'RF':
        classifier_and_parameters = {
            'classifier': [RandomForestClassifier(class_weight='balanced')],
            'classifier__n_estimators': [50, 100, 200],
            "classifier__max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15, 20],
            'classifier__min_samples_split': [2, 5, 10, 15, 20]
        }
    if model_name == 'AdaBoost':
        classifier_and_parameters = {
            'classifier': [AdaBoostClassifier()],
            'classifier__n_estimators': [50, 100, 200],
            "classifier__learning_rate"    : [0.001, 0.01, 0.1, 1] ,
        }
    # if model_name == 'GBDT':
    #     classifier_and_parameters = {
    #         'classifier': [GradientBoostingClassifier()],
    #         'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #         'classifier__max_iter': [10, 100, 1000],
    #         'classifier__penalty': ['l1', 'l2']
    #     }
    if model_name == 'XGBoost':
        classifier_and_parameters = {
            'classifier': [xgb.XGBClassifier(n_jobs=-1)],      
            'classifier__n_estimators': [50, 100, 200],
            "classifier__max_depth": [ 3,  5, 10, 15]
        }
    return classifier_and_parameters

# Problem Transformation (1) - Binary Relevance 
def BR_model(X, label, model_name, cv):
    print(" ==============================================")
    print(" Binary Relevance Model")
    print(" ==============================================")

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size= 0.2)

    classifier_and_parameters = match_classifier(model_name)

    model_to_tune = BinaryRelevance()
    
    hamming_score = metrics.make_scorer(evaluation.hamming_score)

    start=time.time()

    model_tuned = GridSearchCV(model_to_tune, classifier_and_parameters, scoring=hamming_score)
    print(model_tuned)

    # Find the best hyper paramters in the training data.
    model_tuned.fit(X_train, y_train)
    # y_pred = model_tuned.predict(X_test)
    # print(confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1)))

    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', model_tuned.best_params_)
    print('best hamming score: ',model_tuned.best_score_)

    

    # use the model with tuned parameters
    best_model = BinaryRelevance(classifier=model_tuned.best_params_['classifier'])

    # cross-validatation evaluation
    cross_validation_score(best_model, X, label, cv)
    return


# Problem Transformation (2) -  Label Powerset
def LP_model(X, label, model_name, cv):

    print(" ==============================================")
    print(" Label Powerset - Model")
    print(" ==============================================")

    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size= 0.2)

    classifier_and_parameters = match_classifier(model_name)

    model_to_tune = LabelPowerset()
    
    hamming_score = metrics.make_scorer(evaluation.hamming_score)
    
    start=time.time()

    model_tuned = GridSearchCV(model_to_tune, classifier_and_parameters, scoring=hamming_score)
    print(model_tuned)

    # Find the best hyper paramters in the training data.
    model_tuned.fit(X_train, y_train)

    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', model_tuned.best_params_)
    print('best hamming score: ',model_tuned.best_score_)


    

    # use the model with tuned parameters
    best_model = LabelPowerset(classifier=model_tuned.best_params_['classifier'])

    # cross-validatation evaluation
    cross_validation_score(best_model, X, label, cv)
    return

  

# Problem Transformation (2) -  Classifier Chain
def CC_model(X, label, model_name, cv):

    print(" ==============================================")
    print("  Classifier Chain - Model")
    print(" ==============================================")


    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size= 0.2)

    classifier_and_parameters = match_classifier(model_name)

    model_to_tune = ClassifierChain()
    
    hamming_score = metrics.make_scorer(evaluation.hamming_score)

    start=time.time()

    model_tuned = GridSearchCV(model_to_tune, classifier_and_parameters, scoring=hamming_score)
    print(model_tuned)

    # Find the best hyper paramters in the training data.
    model_tuned.fit(X_train, y_train)

    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', model_tuned.best_params_)
    print('best hamming score: ',model_tuned.best_score_)

    

    # use the model with tuned parameters
    best_model = ClassifierChain(classifier=model_tuned.best_params_['classifier'])

    # cross-validatation evaluation
    cross_validation_score(best_model, X, label, cv)
    return


def obtain_feature_importance(best_model, X, label ,cv):
    
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size= 0.2)

    # fit model no training data

    best_model.fit(X_train, y_train)

    # prediction
    y_pred = best_model.predict(X_test)

    evaluation.evaluation_score(y_test,y_pred)

    # plot feature importance


    xgb_model = best_model.get_params()['classifier']

    feature_importance_score = xgb_model.feature_importances_.tolist()
    feature_importance_score_dict = {}
    for index in range(len(feature_importance_score)):
        f_name = 'f%d' % index
        feature_importance_score_dict[f_name] = feature_importance_score[index]
    print("feature number: %d" % len(X_train[0]))
    print("feature importance number: %d" % len(feature_importance_score))
    feature_importance_score_dict = helper.get_sorted_dict(feature_importance_score_dict)
    helper.store_result_json_file('xgboost_model_feature_importance_scores.json', feature_importance_score_dict)
    
    # plot
    # plot_importance(xgb_model)
    # pyplot.show()

    plot_importance(xgb_model, max_num_features=25, importance_type="gain", title= 'Feature Importance (Gain)')
    fig = plt.gcf()
    plt.savefig('result_analysis/xgboost_model_feature_importance_gain_max_25.pdf')
    plt.show()

    plot_importance(xgb_model, max_num_features=25, importance_type="weight", title= 'Feature Importance (Weight)')
    fig = plt.gcf()
    fig.savefig('result_analysis/xgboost_model_feature_importance_gain_max_25.pdf')
    plt.show()


    plot_importance(xgb_model, max_num_features=25, importance_type="cover", title= 'Feature Importance (Cover)')
    fig = plt.gcf()
    fig.savefig('result_analysis/xgboost_model_feature_importance_gain_max_25.pdf')
    plt.show()


    
        
    