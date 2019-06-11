import numpy as np
import pandas as pd
import os 
from matplotlib import pyplot as plt
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from util import *
from label_handler import LabelHandler
from pdpbox import pdp, info_plots
import shap

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

def most_important_features(model, train_data, target_data, num= 50):
    model.fit(train_data, target_data)
    importances = model.feature_importances_
    
    priority = np.argsort(importances)[::-1][:num]
    features = np.array(train_data.columns[priority])
    scores = importances[priority]
    return features, scores

def plot_feature_scores(features, scores, name, info= None):
    data_table = [['Feature', 'Score - {}'.format(info)]]
    for i in range(len(features)):
        pair = [features[i], scores[i]]
        data_table.append(pair)
    table = ff.create_table(data_table)
    py.plot(table, filename= name)

def plot_single_feature_impact(model, data, feature, img_name):
    pdp_dist = pdp.pdp_isolate(model, dataset= data, model_features= data.columns, feature= feature)
    pdp.pdp_plot(pdp_dist, feature)
    plt.savefig(img_name)

def analyze_by_category():
    target = 'SLQ050'
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

def analyze_features():
    target = 'SLQ040'
    model = model = RandomForestClassifier(n_estimators= 200, max_depth= 10, random_state= 0)
    DPQ030_categories = ['Cardiovascular Health (CDQ_I)', 'Current Health Status (HSQ_I)', 'Disability (DLQ_I)', 'Hospital Utilization & Access to Care (HUQ_I)',
                        'Medical Conditions (MCQ_I)', 'Mental Health - Depression Screener (DPQ_I)']
    
    SLQ050_categories = ['Audiometry (AUQ_I)', 'Blood Pressure & Cholesterol (BPQ_I)', 'Cardiovascular Health (CDQ_I)', 'Diabetes (DIQ_I)',
                        'Disability (DLQ_I)', 'Hospital Utilization & Access to Care (HUQ_I)', 'Income (INQ_I)', 'Medical Conditions (MCQ_I)',
                        'Mental Health - Depression Screener (DPQ_I)', 'Physical Activity (PAQ_I)', 'Weight History (WHQ_I)', 
                        'Demographic Variables and Sample Weights (DEMO_I)', 'Standard Biochemistry Profile (BIOPRO_I)']

    DPQ030_categories = ['Mental Health - Depression Screener (DPQ_I)']

    label_handler = get_all_handler()
    symbols = label_handler.get_symbols_by_categories(DPQ030_categories)
    symbols = []
    target_data = get_2015_sleep_data(target)
    train_data, target_data = get_2015_all(target_data, symbols)
    acc, auc = evaluate(model, train_data, target_data)
    info = '| ACC: {:.3f} | AUC: {:.3f}'.format(acc, auc)

    features, scores = most_important_features(model, train_data, target_data, 50)
    print(' ------ Most important Feature ------')
    for f in features:
        print(f)
    
    plot_feature_scores(features, scores, 'SLQ040_all', info= info)

def test():
    target = 'SLQ030'
    model = model = RandomForestClassifier(n_estimators= 200, max_depth= 10, random_state= 0)

    category = 'Body Measures (BMX_I)'
    label_handler = get_all_handler()
    symbols = label_handler.get_symbols_by_category(category)
    target_data = get_2015_sleep_data(target)
    train_data, target_data = get_2015_all(target_data, symbols)

    evaluate(model, train_data, target_data)
    model.fit(train_data, target_data)
    target_feautre = 'Weight (kg)'
    plot_single_feature_impact(model, train_data, target_feautre, 'test.png')

def main():
    analyze_features()

if __name__ == '__main__':
    import sys
    mode = sys.argv[1]
    if mode == 'main':
        print('-- MAIN --')
        main()
    else:
        print('-- TEST --')
        test()