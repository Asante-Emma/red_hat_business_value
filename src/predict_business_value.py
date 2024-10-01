import pandas as pd
from keras.models import load_model
import joblib

def align_columns(df, model_columns):
    """
    Align the DataFrame's columns with the model's expected columns.
    Add missing columns with a default value of 0 and ensure the correct order of columns.
    """
    # Identify missing columns that need to be added
    missing_cols = set(model_columns) - set(df.columns)

    # Add missing columns in the correct order with a default value of 0
    for col in model_columns:
        if col not in df.columns:
            df.insert(model_columns.index(col), col, 0)

    # Reorder the DataFrame to match the model's expected column order
    df = df[model_columns]

    return df


def make_predictions(file_path, model, scaler, model_columns):
    """
    Load data, align columns, scale data, make predictions, and return DataFrame with predictions.
    """
    # Load the data from CSV
    data = pd.read_csv(file_path, compression='zip')

    # Align the data columns with the model's expected columns
    data_aligned = align_columns(data, model_columns)

    # Scale the data
    data_scaled = scaler.transform(data_aligned)

    # Make predictions
    predictions = model.predict(data_scaled)

    # Convert predictions to binary classes
    predictions_class = (predictions > 0.5).astype("int32")

    # Add predictions to DataFrame
    data_aligned['Predicted_Outcome'] = predictions_class

    # Return the DataFrame with predictions
    return data_aligned

if __name__ == '__main__':

    # Set up require column names
    preprocessed_df = pd.read_csv('../data/processed/processed_data.csv.zip', compression='zip')
    preprocessed_df.drop(columns=['outcome'], inplace=True)
    columns = preprocessed_df.columns.to_list()

    # Load the saved model
    model = load_model('../models/red_hat_model.keras')

    # Load the scaler
    scaler = joblib.load('../models/standard_scaler.pkl')

    # Model columns
    model_columns = columns

    # Test data file path
    file_path = '../data/interim/testing_data.csv.zip'

    # Make predictions
    predicted_data = make_predictions(file_path=file_path, model=model, scaler=scaler, model_columns=model_columns)
    print(predicted_data)