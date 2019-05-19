import numpy as np
import pandas as pd

import sys

mypath = sys.argv[1]
Ques_data = pd.read_csv(mypath)

feature_filter_thresh = 0.5
index_filter_thresh = 0.5

print("Origin Ques_data.shape:",Ques_data.shape)

# delete SEQN number
Ques_data = Ques_data.drop([Ques_data.columns[0],Ques_data.columns[1]], axis=1)
print(Ques_data.head())
print(Ques_data.dtypes.value_counts())

print("\nDeleting object type data ... ...\n",Ques_data.select_dtypes(include="object").columns)
# access only float type data
Ques_data = Ques_data.select_dtypes(include="float")
print(Ques_data.shape)

# get Sleep Label
Label = Ques_data.iloc[:,:5] #ignore SLQ300 SLQ310
print("\nSleep disorder features:",Label.shape)
print(Label.columns)

# filter feature with more than threshold missing 


delete_feature_list = []
for feature in Ques_data.columns:
    counts = Ques_data[feature].isna().sum()
    if (counts/Ques_data.shape[0]) > feature_filter_thresh:
        delete_feature_list.append(feature)

print("Number of feature with more than half NaN:",len(delete_feature_list))
Ques_data = Ques_data.drop(delete_feature_list, axis=1)

delete_index_list = []
for index in range(Ques_data.shape[0]):
    counts = Ques_data.iloc[index].isna().sum()
    if (counts/Ques_data.shape[1]) > index_filter_thresh:
        delete_index_list.append(index)

print("Number of index with more than half NaN:",len(delete_index_list))
Ques_data = Ques_data.drop(delete_index_list, axis=0)
print("Filtering data ... ...")

print("\nShape of current data: {}".format(Ques_data.shape))


#fill_values = {}
#for feature in Ques_data.columns:
#    fill_values[feature] = Ques_data[feature].max()

Ques_data = Ques_data.apply(lambda column: column.fillna(column.max()), axis=0)

import matplotlib.pyplot as plt
import seaborn as sns #for plotting
from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
import shap #for SHAP values
from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproducibility
from sklearn.preprocessing import MinMaxScaler, Imputer

#imputer = Imputer(strategy=np.max())

#Ques_data = imputer.fit_transform(Ques_data)

for i in range(5):
    Ques_data.iloc[:,i] = Ques_data.iloc[:,i].astype('int32')

for i in range(5):
    target = Label.columns[i]
    print("\nTraining target:",target)
    print("Target distribution:\n",Ques_data[target].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(Ques_data.drop(target, 1), Ques_data[target], test_size = .2, random_state=10)

    model = RandomForestClassifier(max_depth=10,min_samples_split=5)
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)
    y_pred_quant = model.predict_proba(X_test)[:, 1]
    y_pred_bin = model.predict(X_test)

    print("Accuracy:",(y_predict==y_test).sum()/y_test.shape[0])
    conf_matrix = confusion_matrix(y_test, y_pred_bin)
    print("Confusion matrix:\n",conf_matrix)

print("Validation number:",y_test.shape[0])
#total=sum(sum(confusion_matrix))

#sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
#print('Sensitivity : ', sensitivity )

#specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
#print('Specificity : ', specificity)