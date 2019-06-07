import numpy as np
import pandas as pd

class LabelHandler():
    def __init__(self, path= None):
        self.main_categories = {}
        if path:
            self.read(path)

    def read(self, path):
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
    
    def get_symbols_by_category(self, category):
        symbols = list(self.main_categories[category])
        return symbols

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

def test_labelhandler():
    path = 'data/2015-2016/Questionnaire.txt'
    handler = LabelHandler(path)
    df = pd.read_csv('data_preprocess/Questionnaire.csv')
    labels = df.columns
    contents, noresult_id = handler.symbols_to_contents(labels)
    print(len(contents))
    print(noresult_id)

if __name__ == '__main__':
    test_labelhandler()