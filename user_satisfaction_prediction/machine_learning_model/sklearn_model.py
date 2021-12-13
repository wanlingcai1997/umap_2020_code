import numpy as np

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
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
import sklearn.metrics as metrics
import time
import evaluation

def cross_validation_score(best_model, X, label, cv):
    print(best_model)
    scoring = {'f1':'f1', 'accuracy':'accuracy', 'precision':'precision',  'recall': 'recall', }
    scores = cross_validate(best_model, X, label, cv=cv, scoring=scoring, return_train_score=True)
    
    print(scores)

    print("f1 :", abs(scores['test_f1'].mean()))
    print("accuracy :", abs(scores['test_accuracy'].mean()))
    print("precision :", abs(scores['test_precision'].mean()))
    print("recall :", abs(scores['test_recall'].mean()))

    return 

def model_train_and_evaluate(model_name, X, Y, cv):
    #----------------------------------------------------------
    #------------ Step 1: Dataset Split -----------------------
    #----------------------------------------------------------
    # Split dataset into Train, Test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    assert len(x_train) + len(x_test) == len(X)
    assert len(y_train) + len(y_test) == len(Y)


    #----------------------------------------------------------
    #------------ Step 2: Select the Model --------------------
    #----------------------------------------------------------

    classifier = None
    parameter_grid = {}
    if model_name == 'LR':
        classifier = LogisticRegression()
        parameter_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [10, 100,500, 1000],
            'penalty': ['l2', 'l1']
        }
    if model_name == 'NB_M':
        classifier = MultinomialNB()
    if model_name == 'LinearSVC':
        classifier = LinearSVC()
        parameter_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'max_iter': [10, 100, 500, 1000]
        }

    if model_name == 'MLP':
        classifier = MLPClassifier()
        parameter_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [ 100, 500, 1000],
        }
    if model_name == 'DT':
        classifier = DecisionTreeClassifier()
        parameter_grid = {
            'max_depth': [ 3, 5,  10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }
    if model_name == 'RF':
        classifier = RandomForestClassifier()
        parameter_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [ 3, 5,  10, 15, 20],
            'min_samples_split': [2, 5, 10]
        }

    if model_name == 'XGBoost':
        classifier = xgb.XGBClassifier()
        parameter_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [ 3, 5, 10, 15, 20],
        }
    if model_name == 'KNN':
        # KNN model requires you to specify n_neighbors,
        # the number of points the classifier will look at to determine what class a new point belongs to
        classifier = KNeighborsClassifier()
        parameter_grid = {
            'n_neighbors': [5, 10, 20, 30]
        }
    if model_name == 'SVM':
        classifier = SVC()
    if model_name == 'AdaBoost':
        classifier = AdaBoostClassifier()
    if model_name == 'GBDT':
        classifier = GradientBoostingClassifier()
    if model_name == 'NB_G':
        classifier = GaussianNB()

    print (classifier.get_params)
    
    

    #----------------------------------------------------------
    #-------- Step 3: Train (fit) the Model -------------------
    #----------------------------------------------------------

    start=time.time()

    classifier_tuned = GridSearchCV(classifier, parameter_grid, scoring='f1')
    print(classifier_tuned)

    # Find the best hyper paramters in the training data.
    classifier_tuned.fit(x_train, y_train)
    
    print('training time taken: ',round(time.time()-start,0),'seconds')
    print('best parameters :', classifier_tuned.best_params_)
    print('best estimator :', classifier_tuned.best_estimator_)
    print('best F1 score: ',classifier_tuned.best_score_)


    #----------------------------------------------------------
    #---------------- Step 4: Evaluation ----------------------
    #----------------------------------------------------------
    prediction_train = classifier_tuned.predict(x_train)
    print("accuracy_train :",  metrics.accuracy_score(prediction_train, y_train))
    prediction_test = classifier_tuned.predict(x_test)
    print("accuracy_test :",  metrics.accuracy_score(prediction_test, y_test))
    # But Confusion Matrix and Classification Report give more details about performance
    print(metrics.confusion_matrix(prediction_test, y_test))
    print(metrics.classification_report(prediction_test, y_test))


    #----------------------------------------------------------
    #---------------- Step 5: Set up the best model -----------
    #----------------------------------------------------------

    cross_validation_score(classifier_tuned.best_estimator_, X, Y, cv)
