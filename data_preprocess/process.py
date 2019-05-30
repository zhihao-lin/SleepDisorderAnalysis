import numpy as np
import pandas as pd

def extract_sleep_data():
    questionnaire_path = 'Questionnaire.csv'
    sleep_path = 'Sleep.csv'

    questionnaire = pd.read_csv(questionnaire_path)
    # sleep = questionnaire.iloc[:, 1:9]
    # sleep.to_csv(sleep_path)
    sleep_label = ['Unnamed: 0', 'SLQ300', 'SLQ310', 'SLD012', 'SLQ030', 'SLQ040', 'SLQ050', 'SLQ120']
    
    questionnaire = questionnaire.drop(sleep_label, 1)
    questionnaire.to_csv('Questionnaire_clean.csv')

if __name__ == '__main__':
    extract_sleep_data()