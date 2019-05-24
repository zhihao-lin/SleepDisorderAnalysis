import pandas as pd
import numpy as np
import os
from sklearn.tree import export_graphviz
import pydot

#sudo apt-get install graphviz
def generate_tree_png(estimator,feature_names,class_names,path):
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = class_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    (graph,) = pydot.graph_from_dot_file('tree.dot')
    os.remove("tree.dot")
    graph.write_png(path)