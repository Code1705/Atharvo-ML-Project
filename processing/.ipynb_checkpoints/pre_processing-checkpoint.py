import pandas as pd
from sklearn.preprocessing import LabelEncoder

def pre_process_data(data):
    """
    Function to preprocess the input dataset:
    1. Encodes categorical variables.
    2. Ensures that the data is in the correct format for prediction models.
    """

    # List of categorical columns that need to be encoded
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Apply label encoding to all categorical columns
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Return the preprocessed data
    return data
