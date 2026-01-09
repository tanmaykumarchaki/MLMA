from evaluation import evaluate_model as eval_model
import numpy as np

def threshold_decision( y_pred, threshold = 0.5):

    """
    Converting regression predictions to binary decisions based on a threshold value.

    """
    return(y_pred >= threshold).astype(int)

def evaluate_decisions( y_true , y_pred, threshold):
    decision = threshold_decision(y_pred, threshold)
    y_true_decision = (y_true >= threshold).astype(int)

    TP =np.sum((decision == 1) & (y_true_decision == 1)) # True Positives
    TN =np.sum((decision == 0) & (y_true_decision == 0)) # True Negatives
    FP =np.sum((decision == 1) & (y_true_decision == 0)) # False Positives
    FN =np.sum((decision == 0) & (y_true_decision == 1)) # False Negatives

    return TP, TN , FP, FN

def decision_pipeline(threshold, model, x_test, y_test):
    y_test, y_pred = eval_model(model, x_test, y_test)
    TP, TN, FP, FN = evaluate_decisions(y_test, y_pred, threshold)

    print("\nDecision Analysis")
    print('-'*20)
    print(f"Threshold: {threshold}")
    print(f"True Positives : {TP}")
    print(f"False Positives : {FP}")
    print(f"True Negatives : {TN}")
    print(f"False Negatives : {FN}")
    print('-'*20)


#example 
from evaluation import *
x , y = generate_data(200)
x_train, x_test, y_train, y_test = train_test_split_func(x,y)
model = normalequation()
model.fit(x_train, y_train)

y_test, y_pred = eval_model(model, x_test, y_test)
decision_pipeline(4.5, model , x_test, y_test)
