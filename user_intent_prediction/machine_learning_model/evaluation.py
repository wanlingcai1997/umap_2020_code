import numpy as np
import math
import sklearn.metrics as metrics

from scipy.sparse import csr_matrix

# def evaluation_score(label_test, predict_label):
#     f1_micro=metrics.f1_score(label_test, predict_label, average='micro')
#     hamm=metrics.hamming_loss(label_test,predict_label)
#     accuracy = metrics.accuracy_score(label_test, predict_label)
#     precision = metrics.precision_score(label_test, predict_label, average='micro') 
#     # f1=metrics.f1_score(label_test, predict_label)
#     recall=metrics.recall_score(label_test, predict_label,average='micro')

#     print('F1-score:',round(f1_micro,4))
#     print('Hamming Loss:',round(hamm,4))
#     print("accuracy :", round(accuracy, 4))
#     print("precision :", round(precision, 4))
#     # print("f1 :",  round(f1, 4))
#     print("recall :", round(recall, 4))
#     return 
def evaluation_score(label_test, predict_label):
    f1_micro=metrics.f1_score(label_test, predict_label, average='micro')
    hamm=metrics.hamming_loss(label_test,predict_label)
    hamm_score = hamming_score(label_test,predict_label)
    accuracy = metrics.accuracy_score(label_test, predict_label)
    precision = metrics.precision_score(label_test, predict_label, average='micro') 
    # f1=metrics.f1_score(label_test, predict_label)
    recall=metrics.recall_score(label_test, predict_label,average='micro')
    cohen = cohen_kappa_score(label_test,predict_label)

    print('F1-score:',round(f1_micro,4))
    print('Hamming Loss:',round(hamm,4))
    print('Hamming Score:',round(hamm_score,4))
    print("accuracy :", round(accuracy, 4))
    print("precision :", round(precision, 4))
    # print("f1 :",  round(f1, 4))
    print("recall :", round(recall, 4))
    print("cohen_kappa_score :", round(cohen, 4))

    
def f1(y_true, y_pred):
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0])
        set_pred = set( np.where(y_pred[i])[0] )
        
        correct_preds += len(set_true & set_pred)
        total_preds += len(set_pred)
        total_correct += len(set_true)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return p, r, f1

def cohen_kappa_score(y_true, y_pred):
    cohen_kappa_score_for_each_class = []

    y_true = np.array(y_true) # since y_true is numpy.matrix
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_pred, np.matrix): 
        y_pred = y_pred.toarray().astype(int) # since y_pre is scipy.sparse.lil_matrix 
   
    # print(y_true.shape)

    for i in range(y_true.shape[1]):
        y_true_i = y_true[:,i]
        y_pred_i = y_pred[:,i]
        # print(metrics.accuracy_score(y_true_i, y_pred_i))
        cohen_kappa_score_i = metrics.cohen_kappa_score(y_true_i, y_pred_i, labels=[1, 0])
        # print(cohen_kappa_score_i)
        if math.isnan(cohen_kappa_score_i) == False:
            cohen_kappa_score_for_each_class.append(cohen_kappa_score_i)
    # print(cohen_kappa_score_for_each_class)
    return np.mean(cohen_kappa_score_for_each_class)

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    y_true = np.array(y_true) # since y_true is numpy.matrix
    if not isinstance(y_pred, np.ndarray) or not isinstance(y_pred, np.matrix): 
        y_pred = y_pred.toarray() # since y_pre is scipy.sparse.lil_matrix 
   
    for i in range(y_true.shape[0]):
        #print(y_true[i])
        #print(y_pred[i])

        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)



if __name__ == "__main__":

    y_true = np.array([[0,1,0],
                   [0,1,1],
                   [1,0,1],
                   [0,0,1]])

    y_pred = np.array([[0,1,1],
                   [0,1,1],
                   [0,1,0],
                   [0,0,0]])

    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) # 0.375 (= (0.5+1+0+0)/4)
    print('Cohen_kappa_score: {0}'.format(cohen_kappa_score(y_true, y_pred))) # 0.375 (= (0.5+1+0+0)/4)
    
    # Subset accuracy
    # 0.25 (= 0+1+0+0 / 4) --> 1 if the prediction for one sample fully matches the gold. 0 otherwise.
    print('Subset accuracy: {0}'.format(metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)))

    # Hamming loss (smaller is better)
    # $$ \text{HammingLoss}(x_i, y_i) = \frac{1}{|D|} \sum_{i=1}^{|D|} \frac{xor(x_i, y_i)}{|L|}, $$
    # where
    #  - \\(|D|\\) is the number of samples  
    #  - \\(|L|\\) is the number of labels  
    #  - \\(y_i\\) is the ground truth  
    #  - \\(x_i\\)  is the prediction.  
    # 0.416666666667 (= (1+0+3+1) / (3*4) )
    print('Hamming loss: {0}'.format(metrics.hamming_loss(y_true, y_pred))) 