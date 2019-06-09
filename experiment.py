import numpy as np
import pandas as pd
import os 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from util import *
from label_handler import LabelHandler

targets = ['SLQ300','SLQ310','SLD012','SLQ030','SLQ040','SLQ050','SLQ120']
# SLQ300 - Usual sleep time on weekdays or workdays
# SLQ310 - Usual wake time on weekdays or workdays
# SLD012 - Sleep hours
# SLQ030 - How often do you snore?
# SLQ040 - How often do you snort or stop breathing
# SLQ050 - Ever told doctor had trouble sleeping?
# SLQ120 - How often feel overly sleepy during day?

def train(model, train_data, target_data):
    scores = cross_val_score(model, train_data, target_data, cv= 5)
    print('Accuracy: {} (+/- {})'.format(scores.mean(), scores.std() ** 2))



def test():
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    cat = label_handler.get_categories()[1]
    # symbols = label_handler.get_symbols_by_category(cat)
    symbols = []

    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Questionnaire_data(target_data, symbols)

    print(target_data)
    

if __name__ == '__main__':
    test()