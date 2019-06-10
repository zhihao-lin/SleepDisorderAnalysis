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
    return acc.mean(), auc.mean()

def select_feature(model, train_data, target_data, num= 50):
    model.fit(train_data, target_data)
    importances = model.feature_importances_
    
    priority = np.argsort(importances)[::-1][:num]
    feature = np.array(train_data.columns[priority])
    return feature

def main():
    target = 'DPQ030'
    model = RandomForestClassifier(n_estimators= 200, max_depth= 10, random_state= 0)
    label_handler = get_all_handler()
    error_messages = []
    results = []
    all_categories = label_handler.get_categories()
    
    for category in all_categories:
        try:
            symbols = label_handler.get_symbols_by_category(category)
            target_data = get_2015_sleep_data(target= target)
            train_data, target_data = get_2015_all(target_data, symbols)
            acc, auc = evaluate(model, train_data, target_data)
            info = '{} | ACC: {} | AUC: {}'.format(category, acc, auc)
            results.append(info)

        except Exception as e:
            info = '{} | Error: {}'.format(category, e)
            error_messages.append(info)

    print('============== Results ===============')
    for result in results:
        print(result)
    print('============== Errors  ===============')
    for error in error_messages:
        print(error)

if __name__ == '__main__':
    main()