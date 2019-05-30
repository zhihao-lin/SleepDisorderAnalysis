import numpy as np
import pandas as pd
from label_handler import LabelHandler

### Get data ##
def select_feature(data,feature_list):
    if feature_list == []:
        return data
    index = []
    for i in range(len(feature_list)):
        feature_list[i] = feature_list[i][0:3]

    col_name = ['SLQ','SLD','SEQ','Unn']+feature_list
    for i in range(len(data.columns)):
        if data.columns[i][0:3] not in col_name :
            index.append(i)
    data = data.drop(data.columns[index],axis=1)
    print('Remove '+str(len(index))+'col | Remain'+str(data.shape[1])+'col')
    return data

def filter_data(csv, column_threshold= 0.5, row_threshold= 0.5):
    raw_shape = csv.shape
    drop_columns = []
    drop_rows = []
    
    data_num = len(csv)
    for feature in csv.columns:
        missing_num = csv[feature].isna().sum()
        if missing_num/data_num > column_threshold:
            drop_columns.append(feature)
    csv = csv.drop(drop_columns, axis= 1)

    feature_num = len(csv.columns)
    for index in range(data_num):
        missing_num = csv.iloc[index].isna().sum()
        if missing_num/feature_num > row_threshold:
            drop_rows.append(index)
    csv = csv.drop(drop_rows, axis = 0)

    print('Dropped column: {}/{}'.format(len(drop_columns), raw_shape[1]))
    print('Dropped row: {}/{}'.format(len(drop_rows), raw_shape[0]))
    return csv

def process_nan(csv):
    # split categorical data & replace nan with median if numerical data
    for feature in csv.columns:
        possible_values = np.unique(csv[feature])
        if len(possible_values) < 10: # categorical
            # print('== {} =='.format(feature))
            csv[feature][csv[feature] == max(possible_values)] = np.nan
            csv[feature] = csv[feature].astype('object')
        else: # numerical
            # print('** {} **'.format(feature))
            csv[feature] = csv[feature].fillna(max(possible_values))
            median = np.median(csv[feature][csv[feature] != max(possible_values)])
            csv[feature][csv[feature] == max(possible_values)] = median
            csv[feature] = csv[feature].astype('float')
    csv = pd.get_dummies(csv)
    return csv

def normalize_time(raw_array):
    normalized = []
    for raw_data in raw_array: 
        if ':' not in raw_data:
            normalized.append(0)
            continue
        hour, minute = str(raw_data).split(':')
        hour, minute = int(hour[2:]), int(minute[:-1])  
        if hour > 12:
            time_diff = ((23 - hour)*60 + (60 - minute)) * (-1)
            normalized.append(time_diff)
        else:
            time_diff = hour * 60 + minute
            normalized.append(time_diff)
    return normalized

def get_2015_Quesitonaire_data( feature_selected,
                                csv= 'data_preprocess/Questionnaire.csv', 
                                label= 'data/2015-2016/Questionnaire.txt'):
    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)
    data = filter_data(raw_csv) # Remove features and data for too much mmissing
    
    #Select features (default:['SLQ','SLD','SEQ','Unn'])
    data = select_feature(data,feature_selected)

    # Replace feautre names with meaningful contents, and remove unknowns
    columns = data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    data.columns = contents
    data = data.drop(noresults, axis= 1)
    object_feature = []
    for i in range(len(data.dtypes)):
        if data.dtypes[i] == 'object':
            object_feature.append(data.columns[i])
    # Remove Cigaratte feature, too many emptybyte string and aren't caught 
    # 05/22:Use select_feature() to solve this problem   
    if feature_selected == []:
       data = data.drop(object_feature[-2:], axis= 1) 

    # Convert time : e.g. b'23:00' -> -60
    data[object_feature[0]] = normalize_time(data[object_feature[0]])
    data[object_feature[1]] = normalize_time(data[object_feature[1]])

    return data

## TEST ##
def test_get_data():
    data = get_2015_Quesitonaire_data()
    data = process_nan(data)
    print(data.columns)

if __name__ == '__main__':
    test_get_data()