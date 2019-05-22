import numpy as np
import pandas as pd

class LabelHandler():
    def __init__(self, path):
        self.main_categories = {}
        category_name = None
        file = open(path, 'r')
        for line in file:
            if line == '':
                continue
            elif line[-1] == '\n':
                line = line[:-1]
            
            if line[0] == '=':
                category_name = line[2:]
                self.main_categories[category_name]= {}
            else:
                split_index = line.index('-')
                symbol, content = line[:split_index - 1], line[split_index + 2:]
                symbol = symbol.upper()
                self.main_categories[category_name][symbol] = content

    def show_all(self):
        for category in self.main_categories:
            print('\n = {} = '.format(category))
            for symbol, content in self.main_categories[category].items():
                print('{} - {}'.format(symbol, content))

    def get_categories(self):
        categories = list(self.main_categories.keys())
        return categories
    
    def get_contents_by_category(self, category):
        contents = list(self.main_categories[category].values())
        return contents
    
    def get_content_by_symbol(self, symbol):
        symbol = symbol.upper()
        for category in self.main_categories:
            if symbol in self.main_categories[category]:
                return self.main_categories[category][symbol]
        return None

    def symbols_to_contents(self, symbols):
        contents = []
        noresult_symbol = []
        for i, symbol in enumerate(symbols):
            content = self.get_content_by_symbol(symbol)
            if not content:
                print('Warn: Symbol {} cant be found'.format(symbol))
                contents.append(symbol)
                noresult_symbol.append(symbol)
            else:
                contents.append(content)
        return contents, noresult_symbol

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
            csv[feature][csv[feature] == max(possible_values)] = np.nan
            csv[feature] = csv[feature].astype('object')
        else: # numerical
            median = np.median(csv[feature][csv[feature] != max(possible_values)])
            csv[feature][csv[feature] == max(possible_values)] = median
            csv[feature][csv[feature] == np.nan] = median
            csv[feature] = csv[feature].astype('float64')
    csv = pd.get_dummies(csv)
    return csv

def get_2015_Quesitonaire_data( csv= 'data_preprocess/Questionnaire.csv', 
                                label= 'data/2015-2016/Questionnaire.txt'):
    raw_csv = pd.read_csv(csv)
    label_handler = LabelHandler(label)
    data = filter_data(raw_csv) # Remove features and data for too much mmissing

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
    data = data.drop(object_feature[-2:], axis= 1) 
    # Convert time : e.g. b'23:00' -> -60
    data[object_feature[0]] = normalize_time(data[object_feature[0]])
    data[object_feature[1]] = normalize_time(data[object_feature[1]])
    return data

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


def test_labelhandler():
    path = 'data/2015-2016/Questionnaire.txt'
    handler = LabelHandler(path)
    df = pd.read_csv('data_preprocess/Questionnaire.csv')
    labels = df.columns
    contents, noresult_id = handler.symbols_to_contents(labels)
    print(len(contents))
    print(noresult_id)

def test_get_data():
    data = get_2015_Quesitonaire_data()
    data = process_nan(data)
    print(data)

if __name__ == '__main__':
    test_get_data()