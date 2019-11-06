import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.metrics import precision_score, roc_auc_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
import xgboost as xgb
import warnings


warnings.filterwarnings('ignore')

attrition = pd.read_csv('input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

def generate_data_description(data):
    description = pd.DataFrame(index=['observations(rows)', 'percent missing', 'dtype', 'range'])
    numerical = []
    categorical = []
    for col in data.columns:
        obs = data[col].size
        p_nan = round(data[col].isna().sum()/obs, 2)
        num_nan = f'{p_nan}% ({data[col].isna().sum()}/{obs})'
        dtype = 'categorical' if data[col].dtype == object else 'numerical'
        numerical.append(col) if dtype == 'numerical' else categorical.append(col)
        rng = f'{len(data[col].unique())} labels' if dtype == 'categorical' else f'{data[col].min()}-{data[col].max()}'
        description[col] = [obs, num_nan, dtype, rng]

    numerical.remove('EmployeeCount')
    numerical.remove('StandardHours')
    return numerical, categorical, description

def data_preparation(data, categorical_features):
    # Define a dictionary for the target mapping
    target_map = {'Yes':1, 'No':0}
    # Use the pandas apply method to numerically encode our attrition target variable
    data["Attrition_numerical"] = data["Attrition"].apply(lambda x: target_map[x])

    lgb_data = attrition.copy()
    lgb_dummy = pd.get_dummies(lgb_data[categorical], drop_first=True)
    lgb_data = pd.concat([lgb_dummy, lgb_data], axis=1)
    lgb_data.drop(columns = categorical, inplace=True)
    lgb_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)

y_df = lgb_data['Attrition'].reset_index(drop=True)
x_df = lgb_data.drop(columns='Attrition')
train_x, test_x, train_y, test_y = train_test_split(x_df, y_df, test_size=0.20)