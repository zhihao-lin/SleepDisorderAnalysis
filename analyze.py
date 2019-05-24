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