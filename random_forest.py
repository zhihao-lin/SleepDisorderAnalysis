import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import *
from analyze import *
from label_handler import LabelHandler
import os

def main():
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    os.makedirs('../analyze_files', exist_ok=True)
    f = open('../analyze_files/acc.txt', 'w')
    #feature_selected = ['ALQ','DUQ','SMQ','DPQ','PAQ'] #choose some data
    #feature_selected = []  #use all data
    category = label_handler.get_categories()
    for cat in range(0,len(category),1):
        try:
            os.makedirs('../analyze_files/'+category[cat], exist_ok=True)
            feature_selected = label_handler.get_symbols_by_category(category[cat])
            print('feature_selected: ',feature_selected)
            data = get_2015_Questionnaire_data(feature_selected)
            target_feature = label_handler.get_content_by_symbol('SLQ050')
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
            print('=======analyze=======')
            generate_tree_png(model.estimators_[0], data.columns, target_feature,'../analyze_files/'+category[cat]+'/tree.png')
            permutation_importance(model, x_valid, y_valid, '../analyze_files/'+category[cat]+'/permutation_importance.csv')
            #partial_dependence_plot('Avg # alcoholic drinks/day - past 12 mos', model, x_valid, data.columns,'../analyze_files/partial_dependence_plot.png')
            shap_plot(model, x_valid, '../analyze_files/'+category[cat]+'/')
            f.write('================================'+"\n")
            f.write('Target feature:'+str(target_feature)+"\n")
            f.write(category[cat]+' and '+category[cat+1]+"\n")
            f.write('feature_selected:'+str(feature_selected)+"\n")
            f.write('--------------------------------'+"\n")
            f.write('Training score: '+str(train_score)+"\n")
            f.write('Training score: '+str(valid_score)+"\n")
            
        except Exception as e: 
            f.write('########Warn########'+"\n")
            f.write(str(e)+"\n")
            f.write('########Warn########'+"\n")
            pass
        continue
if __name__ == '__main__':
    main()
    

