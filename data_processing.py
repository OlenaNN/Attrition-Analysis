import pandas as pd


import warnings


warnings.filterwarnings('ignore')

#attrition = pd.read_csv('input/WA_Fn-UseC_-HR-Employee-Attrition.csv')

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
    processed_data = data.copy()
    dummy = pd.get_dummies(processed_data[categorical_features], drop_first=True)
    processed_data = pd.concat([dummy, processed_data], axis=1)
    processed_data.drop(columns = categorical_features, inplace=True)
    processed_data.rename(columns={'Attrition_Yes': 'Attrition'}, inplace=True)
    return processed_data

