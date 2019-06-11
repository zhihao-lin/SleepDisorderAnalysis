import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#for generate tree
from sklearn.tree import export_graphviz
import pydot
from subprocess import call
#for purmutation importance
import eli5 
from eli5.sklearn import PermutationImportance
#for partial plots
from pdpbox import pdp, info_plots 
#for SHAP values
import shap 


#sudo apt-get install graphviz
def generate_tree_png(estimators, feature_names, class_names, path):
    export_graphviz(estimators, out_file='tree.dot', 
                    feature_names = feature_names,
                    class_names = class_names,
                    rounded = True, proportion = False, 
                    precision = 2, filled = True)

    (graph,) = pydot.graph_from_dot_file('tree.dot')
    os.remove("tree.dot")
    
    graph.write_png(path)
    print('generate '+path)

def permutation_importance(model, val_X, val_y, path):
    perm = PermutationImportance(model, random_state=1).fit(val_X, val_y)
    Table = eli5.explain_weights_df(perm, feature_names = val_X.columns.tolist())
    Table.to_csv(path)
    print('generate '+path)

#sudo apt-get install python3.6-dev
#pip3 install pdpbox
def  partial_dependence_plot(feat_name, model, X_test, base_features, path):
    pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=base_features, feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.savefig(path)
    print('generate '+path)
    plt.close()

def shap_plot(model, X_test, path):
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False   )
    plt.savefig(path+'shap_plot_1.png', bbox_inches='tight')
    print('generate '+path+'shap_plot_1.png')
    plt.close()
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig(path+'shap_plot_2.png', bbox_inches='tight')
    print('generate '+path+'shap_plot_2.png')
    plt.close()
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[1,:].astype(float))
    shap.initjs()
    
    shap.force_plot(explainer.expected_value[1], shap_values[1],
                    X_test.iloc[1,:].astype(float), show=False,
                    matplotlib=True, text_rotation=60)
    plt.savefig(path+'shap_plot_3.png', bbox_inches='tight')
    print('generate '+path+'shap_plot_3.png')
    plt.close()
    '''

#   Add confusion_matrix(sensitivity, specificity) & ROC &AUC

from sklearn.metrics import confusion_matrix as cfm
from sklearn.metrics import roc_curve, auc 

def confusion_matrix(y_valid, y_pred):
    confusion_matrix = cfm(y_valid, y_pred)
    sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
    specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    print("sensitivity = {}/({}+{}) = {}".format(confusion_matrix[0,0],
    confusion_matrix[0,0],
    confusion_matrix[1,0],
    sensitivity))
    print("specificity = {}/({}+{}) = {}".format(confusion_matrix[1,1],
    confusion_matrix[1,1],
    confusion_matrix[0,1],
    specificity))
    # return confusion_matrix, sensitivity, specificity

def plot_ROC(y_valid, y_pred_quant, path):
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_quant)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.savefig(path+'ROC', bbox_inches='tight')
    plt.close()
    print("Auc:",auc(fpr, tpr))
    return auc(fpr, tpr)
"""     
    0.90 - 1.00 = excellent
    0.80 - 0.90 = good
    0.70 - 0.80 = fair
    0.60 - 0.70 = poor
    0.50 - 0.60 = fail """

