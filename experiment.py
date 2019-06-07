import numpy as np
import pandas as pd
import os 
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
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

def test():
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    cat = label_handler.get_categories()[1]
    # symbols = label_handler.get_symbols_by_category(cat)
    symbols = []

    target_data = get_2015_sleep_data(target= 'SLQ050')

    train_data, target_data = get_2015_Questionnaire_data(target_data, symbols)
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, target_data, test_size = .2, random_state=0) 
    model = RandomForestClassifier(n_estimators= 20, max_depth=10, random_state= 0)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    valid_score = model.score(x_valid, y_valid)

    print('Train Acc: {}'.format(train_score))
    print('Valid Acc: {}'.format(valid_score))

if __name__ == '__main__':
    test()