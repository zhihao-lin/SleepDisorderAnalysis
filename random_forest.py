import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from util import *
from analyze import *
from label_handler import LabelHandler
import os

def main(mode):
    
    #load LabelHandler
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    
    #different mode
    if mode == 0:
        path = '../analyze_mode_0/'
        category=['']
        #use all data
        feature_selected = []  
    elif mode == 1:
        path = '../analyze_mode_1/'
        category = label_handler.get_categories()
    elif mode == 2:
        path = '../analyze_mode_2/'
        category=['']
        feature_selected = ['ALQ130','ALQ151','BPQ080','BPQ020','BPQ070','CDQ001','CDQ010',
                            'CBD091','CBD121','HUQ010','HSQ590','DED120','DIQ180','DIQ170',
                            'DLQ150','DLQ140','DUQ200','HIQ270','HUQ051','HUQ090','INQ030',
                            'MCQ160a','DPQ030','DPQ040','DPQ020','PFQ051','PFQ049','SXD031',
                            'SMQ858','SMQ856','WHQ150','WHD050']
    elif mode == 3:
        path = '../analyze_mode_3/'
        category=['']
        feature_selected = ['HUQ051','DPQ030','PFQ049','DLQ140',
                            'DLQ150','HUQ090','CDQ001','PFQ051',
                            'WHD050','BPQ070','DIQ170']
    #make ../analyze_mode_x & acc.txt
    os.makedirs(path, exist_ok=True)
    f = open(path+'acc.txt', 'w')
    print(path)
    #train
    for cat in category:
        try:
            os.makedirs(path+cat, exist_ok=True)
            if mode == 1:
                feature_selected = label_handler.get_symbols_by_category(cat)
            print('feature_selected: ',feature_selected)
            target_data = get_2015_sleep_data(target= 'SLQ050')
            target_feature = label_handler.get_content_by_symbol('SLQ050')
            train_data, target_data = get_2015_Questionnaire_data(feature_selected, target_data)

            x_train, x_valid, y_train, y_valid = train_test_split(train_data, target_data, test_size = .2, random_state=0) 
            model = RandomForestClassifier(n_estimators= 20, max_depth=10, random_state= 0)
            model.fit(x_train, y_train)
            train_score = model.score(x_train, y_train)
            valid_score = model.score(x_valid, y_valid)
            print('================================')
            print('Target feature:', target_feature)
            print('Training score: ', train_score)
            print('Training score: ', valid_score)
            print('=======analyze=======')
            generate_tree_png(model.estimators_[0], train_data.columns, target_feature,path+cat+'/tree.png')
            permutation_importance(model, x_valid, y_valid, path+cat+'/permutation_importance.csv')
            #partial_dependence_plot('Avg # alcoholic drinks/day - past 12 mos', model, x_valid, data.columns,'../analyze_files/partial_dependence_plot.png')
            shap_plot(model, x_valid, path+cat+'/')
            f.write('================================'+"\n")
            f.write('Target feature:'+str(target_feature)+"\n")
            f.write(cat+"\n")
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
    main(mode = 3) 
    #mode = 0 :select all
    #mode = 1 :every category
    #mode = 2 :choose mode1 mean(SHAP value) > 0.25 
    #mode = 3 :choose mode2 mean(SHAP value) top 14 except ['DPQ040','CDQ010','BPQ020']

