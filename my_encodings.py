from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np

def dummy_encode(train_data: pd.DataFrame, test_data:pd.DataFrame, categories):    
    train_data = train_data.copy()
    test_data = test_data.copy()

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False) # initialize encoder

    # train
    encoded_train_data = encoder.fit_transform(train_data[categories]) # fit and transform data via encoder 
    new_cols = encoder.get_feature_names_out(categories)
    train_data[new_cols] = encoded_train_data 
    train_data = train_data.drop(columns=categories)

    # test
    encoded_test_data = encoder.transform(test_data[categories]) # fit and transform data via encoder 
    test_data[new_cols] = encoded_test_data 
    test_data = test_data.drop(columns=categories)

    return train_data, test_data

def ordinal_encode(train_data: pd.DataFrame, test_data:pd.DataFrame, column, category_order):  
    train_data = train_data.copy()
    test_data = test_data.copy()
    
    encoder = OrdinalEncoder(categories=[category_order], handle_unknown="use_encoded_value", unknown_value=np.nan) # initialize encoder with category order; handle unknowns
    
    # train
    encoded_train_data = encoder.fit_transform(train_data[[column]])
    new_col = encoder.get_feature_names_out([column])[0] 
    train_data[new_col] = encoded_train_data # update df with encoded values 
    train_data = train_data.drop(columns=[column])

    # test
    encoded_test_data = encoder.transform(test_data[[column]])
    test_data[new_col] = encoded_test_data
    test_data = test_data.drop(columns=[column]) 

    return train_data, test_data

