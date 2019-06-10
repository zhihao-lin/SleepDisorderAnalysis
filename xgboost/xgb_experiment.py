import numpy as np
import pandas as pd
import os 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from util import *
from label_handler import LabelHandler

import xgboost as xgb
# targets 
# SLQ300 - Usual sleep time on weekdays or workdays
# SLQ310 - Usual wake time on weekdays or workdays
# SLD012 - Sleep hours
# SLQ030 - How often do you snore?
# SLQ040 - How often do you snort or stop breathing
# SLQ050 - Ever told doctor had trouble sleeping?
# SLQ120 - How often feel overly sleepy during day?
# DPQ030 - Trouble sleeping or sleeping too much

def evaluate(model, train_data, target_data):
    acc = cross_val_score(model, train_data, target_data, cv= 5)
    auc = cross_val_score(model, train_data, target_data, cv= 5, scoring= 'roc_auc')
    
    print('========================')
    print('ACC: {:.3f} (+/- {:.3f})'.format(acc.mean(), acc.std()))
    print('AUC: {:.3f} (+/- {:.3f})'.format(auc.mean(), auc.std()))
    return acc, auc

def select_feature(model, train_data, target_data, num= 50):
    model.fit(train_data, target_data)
    importances = model.feature_importances_
    
    priority = np.argsort(importances)[::-1][:num]
    feature = np.array(train_data.columns[priority])
    return feature

def main():
    model = xgb.XGBClassifier()
    label_handler = get_all_handler()

    symbols = ['DPQ010', 'DPQ020', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100']
    target_data = get_2015_sleep_data(target= 'DPQ030')
    train_data, target_data = get_2015_Questionnaire_data(target_data, symbols)
    evaluate(model, train_data, target_data)

    features = select_feature(model, train_data, target_data, 20)
    print(features)

if __name__ == '__main__':
    main()