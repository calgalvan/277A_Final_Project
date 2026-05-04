from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd

def dummy_encode(data: pd.DataFrame, categories):    

    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False) # initialize encoder
    encoded_data = encoder.fit_transform(data[categories]) # fit and transform data via encoder 

    new_cols = encoder.get_feature_names_out()
    data[new_cols] = encoded_data 

    return data.drop(columns=categories) # drop from df after encoding 

def ordinal_encode(data: pd.DataFrame, column, category_order):  

    encoder = OrdinalEncoder(categories=[category_order], handle_unknown="use_encoded_value", unknown_value=np.nan) # initialize encoder with category order; handle unknowns
    encoded_data = encoder.fit_transform(data[[column]])

    new_col = encoder.get_feature_names_out() 
    data[new_col] = encoded_data # update df with encoded values 

    return data 

