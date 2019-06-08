from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
import numpy as np
from scipy.stats import uniform

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
        scores, np.mean(scores), np.std(scores)))


def report_best_scores(results, scoring_function, n_top=3):
    print("\n\nHyperparameter searching")
    
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score({scoring_function}): {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate],
                scoring_function=scoring_function,))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def Hyperparameter_searching(xgb_model, X, y):

    params = {
        'gamma': uniform(0, 0.5),
        'n_estimators': range(80, 200, 4),
        'learning_rate': np.linspace(0.01, 2, 20),
        'max_depth': range(2, 13, 1),
        'min_child_weight': range(1, 9, 1),
        'subsample': np.linspace(0.7, 0.9, 20),
        'colsample_bytree': np.linspace(0.5, 0.98, 10),
        'reg_alpha': [0.05, 0.1, 1, 2, 3], 
        'reg_lambda': [0.05, 0.1, 1, 2, 3]
    }
    scoring_function = 'roc_auc'

    # n_jobs=-1 -> use all CPU 
    # n_iter -> number of parameter sampled
    # cross validation to find the model with best "scoring_function"
    # default 3-fold cross validation,

    search = RandomizedSearchCV(xgb_model, param_distributions=params,
                                random_state=42, n_iter=500, verbose=1, cv=3, scoring=scoring_function, n_jobs=-1, return_train_score=True)
    search.fit(X, y)

    # display n_top model
    report_best_scores(search.cv_results_, scoring_function, n_top=3)

    # return the best model on cross validation
    return search.best_estimator_
