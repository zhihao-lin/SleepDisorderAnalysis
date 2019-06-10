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

def select_feature(model, train_data, target_data, num= 10):
    model.fit(train_data, target_data)
    importances = model.feature_importances_
    print('========================')
    print(importances)
    priority = np.argsort(importances)[-num:]
    print(priority)
    print(importances[priority])
    feature = train_data.columns[priority]
    print(feature)

def main():
    label_handler = get_all_handler()
    # cat = label_handler.get_categories()[1]
    # symbols = label_handler.get_symbols_by_category(cat)
    symbols = []

    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Questionnaire_data(target_data, symbols)

    model = RandomForestClassifier(n_estimators= 20, max_depth= 10, random_state= 0)
    # select_feature(model, train_data, target_data)
    evaluate(model, train_data, target_data)

if __name__ == '__main__':
    main()