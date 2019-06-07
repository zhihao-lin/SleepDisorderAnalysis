import numpy as np
import pandas as pd
from label_handler import LabelHandler

### Get data ##
def select_feature(data, symbol_list):
    if symbol_list == []:
        return data

    symbol_list += ['SEQN']
    symbol_list = [symbol for symbol in symbol_list if(symbol in data.columns)]
    data = data[symbol_list]
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

def align_data_with_target(train_data, target_data):
    target = target_data.columns[1]
    all_data = pd.merge(train_data, target_data, how= 'inner', on= 'SEQN')
    train_data = all_data.drop(target, axis= 1)
    target_data = all_data[target]
    return train_data, target_data

def process_nan(csv):
    # split categorical data & replace nan with median if numerical data
    features_to_discard = []
    for feature in csv.columns:
        try:
            possible_values = np.unique(csv[feature][~np.isnan(csv[feature])])
        except:
            print(' == Fail to process follow ==')
            print(csv[feature][:10])
            features_to_discard.append(feature)
            continue

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
    
    if len(features_to_discard) > 0:
        csv = csv.drop(features_to_discard, axis= 1)

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

def get_2015_sleep_data(csv= 'data_preprocess/Sleep.csv', 
                        label= 'data/2015-2016/Questionnaire.txt', target= None):
    
    data = pd.read_csv(csv)
    label_handler = LabelHandler(label)
    data = data.drop('Unnamed: 0', 1)
    
    if target:
        data = data[['SEQN', target]]
        data = filter_data(data, 1, 0)
        return data

    columns = data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    data.columns = contents
    
    # Convert time : e.g. b'23:00' -> -60
    data[contents[1]] = normalize_time(data[contents[1]])
    data[contents[2]] = normalize_time(data[contents[2]])
    return data

def get_2015_Questionnaire_data(target_data, symbol_list= [],
                                csv= 'data_preprocess/Questionnaire.csv', 
                                label= 'data/2015-2016/Questionnaire.txt'):
    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)
    
    train_data = select_feature(raw_csv, symbol_list)
    # Remove features and data for too much mmissing
    train_data = filter_data(train_data) 
    # Align train data and target data according to 'SEQN'
    train_data, target_data = align_data_with_target(train_data, target_data)
    
    # Replace feautre names with meaningful contents, and remove unknowns
    columns = train_data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    train_data.columns = contents
    train_data = train_data.drop(noresults, axis= 1)
    
    # Remove Cigaratte feature, too many emptybyte string and aren't caught 
    if symbol_list == []:
        cigarette_feature = ['Cig 12-digit Universal Product Code-UPC', 'Cigarette Brand/sub-brand']
        train_data = train_data.drop(cigarette_feature, axis= 1) 

    train_data = process_nan(train_data)
    return train_data, target_data

def get_2015_Demorgraphics_data(target_data, symbol_list= [], 
                                csv= 'data_preprocess/Demographics.csv',
                                label= 'data/2015-2016/Demographics.txt'):
    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)

    train_data = select_feature(raw_csv, symbol_list)
    train_data = filter_data(train_data)
    train_data, target_data = align_data_with_target(train_data, target_data)

    columns = train_data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    train_data.columns = contents
    if noresults:
        train_data = train_data.drop(noresults, axis= 1)

    train_data = process_nan(train_data)
    return train_data, target_data

def get_2015_Examination_data(target_data, symbol_list= [], 
                        csv= 'data_preprocess/Examination.csv',
                        label= 'data/2015-2016/Examination.txt'):
    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)

    train_data = select_feature(raw_csv, symbol_list)
    train_data = filter_data(train_data)
    train_data, target_data = align_data_with_target(train_data, target_data)

    columns = train_data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    train_data.columns = contents
    if noresults:
        train_data = train_data.drop(noresults, axis= 1)

    train_data = process_nan(train_data)
    return train_data, target_data

def get_2015_Laboratory_data(target_data, symbol_list= [], 
                        csv= 'data_preprocess/Laboratory.csv',
                        label= 'data/2015-2016/Laboratory.txt'):

    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)

    train_data = select_feature(raw_csv, symbol_list)
    train_data = filter_data(train_data)
    train_data, target_data = align_data_with_target(train_data, target_data)

    columns = train_data.columns
    contents, noresults = label_handler.symbols_to_contents(columns)
    train_data.columns = contents
    if noresults:
        train_data = train_data.drop(noresults, axis= 1)

    train_data = process_nan(train_data)
    return train_data, target_data 

## TEST ##
def test_get_sleep():
    data = get_2015_sleep_data(target = 'SLQ120')
    print(data)

def test_get_questionnaire():
    label_handler = LabelHandler('data/2015-2016/Questionnaire.txt')
    cat = label_handler.get_categories()[1]
    symbols = label_handler.get_symbols_by_category(cat)
    # symbols = []

    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Questionnaire_data(target_data, symbols)
    print(train_data)
    print(target_data)

def test_get_demorgraphics():
    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Demorgraphics_data(target_data)
    
    print(train_data.columns)
    print(train_data.head())

def test_get_examination():
    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Examination_data(target_data)

    for column in train_data.columns:
        print(train_data[column][:10])
        print(' ---------- ')

def test_get_laboratory():
    target_data = get_2015_sleep_data(target= 'SLQ050')
    train_data, target_data = get_2015_Laboratory_data(target_data)

    # for column in train_data.columns:
    #     print(train_data[column][:10])
    #     print(' ---------- ')
    print(train_data)

if __name__ == '__main__':
    test_get_laboratory()