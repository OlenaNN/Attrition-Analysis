import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import xgboost as xgb
import warnings
import data_processing
import data_io

warnings.filterwarnings('ignore')
attrition = pd.read_csv('input/WA_Fn-UseC_-HR-Employee-Attrition.csv')
numerical_data , categorical_data, data_description = data_processing.generate_data_description(attrition)

