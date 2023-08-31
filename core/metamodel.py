import sys 
sys.path.append("../")

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sn
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score
import umap
import sklearn
import pandas as pd 
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import catboost as cb 
import torch.nn as nn 

from .utils import (find_non_numerical_columns,categorize_columns,evaluate_cross_validation,
                   convert_df_to_numeric_df,scale_columns,evaluate_model,optimize_xgb,optimize_catboost)

from sklearn.linear_model import LinearRegression





class MetaModel(nn.Module):

    def __init__(self,linear_subset_size=1000):
        self.linear_subset_size = linear_subset_size

    def train(self,X_train, y_train,X_test,y_test):
        self.xgb_model = optimize_xgb(X_train, y_train,X_test,y_test)
        self.catboost_model = optimize_catboost(X_train, y_train,X_test,y_test)

        catboost_preds = self.catboost_model.predict(X_train[-self.linear_subset_size :])
        xgboost_preds = self.xgb_model.predict(X_train[-self.linear_subset_size :])

        meta_features = np.column_stack((catboost_preds, xgboost_preds))
        target_values = y_train[-self.linear_subset_size :]

        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features, target_values)


    def predict(self,X) :
        catboost_pred = self.catboost_model.predict(X)
        xgboost_pred = self.xgb_model.predict(X)
        meta_features_new = np.array([catboost_pred, xgboost_pred]).reshape(-1, 2)
        return self.meta_model.predict(meta_features_new)
