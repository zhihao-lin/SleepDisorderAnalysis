import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import *

def main():
    data = get_2015_Quesitonaire_data()
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    target_feature = label_handler.get_content_by_symbol('SLQ050')
	#SLQ310：0.98/0.97
	#SLQ050：0.90/0.77
    target = data[target_feature].astype('int')
    target[target <  2] = 0
    target[target >= 2] = 1
    data = data.drop([target_feature], axis= 1)
    data = process_nan(data)
    x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size = .2, random_state=0) 
    model = RandomForestClassifier(n_estimators= 20, max_depth=10, random_state= 0)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    valid_score = model.score(x_valid, y_valid)

    print('================================')
    print('Target feature:', target_feature)
    print('Training score: ', train_score)
    print('Training score: ', valid_score)

    

if __name__ == '__main__':
    main()