from flask import Flask, request, jsonify
import pandas as pd
from make_predictions import make_predictions  # Import your prediction function
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load the saved model and scaler
model = load_model('../models/red_hat_model.keras')
scaler = joblib.load('../models/standard_scaler.pkl')

# Define the expected model columns
model_columns = ['char_1', 'char_10', 'char_11', 'char_12', 'char_13', 'char_14', 'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20', 'char_21', 'char_22', 'char_23', 'char_24', 'char_25',
                'char_26', 'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32', 'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38', 'activity_type_labeled', 'group_1_labeled',
                'activity_category_type 2', 'activity_category_type 3', 'activity_category_type 4', 'activity_category_type 5', 'activity_category_type 6', 'activity_category_type 7', 'char_2_type 2',
                'char_2_type 3', 'char_3_type 10', 'char_3_type 11', 'char_3_type 12', 'char_3_type 13', 'char_3_type 14', 'char_3_type 15', 'char_3_type 16', 'char_3_type 17', 'char_3_type 18', 'char_3_type 19',
                'char_3_type 2', 'char_3_type 20', 'char_3_type 21', 'char_3_type 22', 'char_3_type 23', 'char_3_type 24', 'char_3_type 25', 'char_3_type 26', 'char_3_type 27', 'char_3_type 28', 'char_3_type 29',
                'char_3_type 3', 'char_3_type 30', 'char_3_type 31', 'char_3_type 32', 'char_3_type 33', 'char_3_type 34', 'char_3_type 35', 'char_3_type 36', 'char_3_type 37', 'char_3_type 38', 'char_3_type 39',
                'char_3_type 4', 'char_3_type 40', 'char_3_type 41', 'char_3_type 42', 'char_3_type 44', 'char_3_type 5', 'char_3_type 6', 'char_3_type 7', 'char_3_type 8', 'char_3_type 9', 'char_4_type 10', 'char_4_type 11',
                'char_4_type 12', 'char_4_type 13', 'char_4_type 14', 'char_4_type 15', 'char_4_type 16', 'char_4_type 17', 'char_4_type 18', 'char_4_type 19', 'char_4_type 2', 'char_4_type 20', 'char_4_type 21', 'char_4_type 22',
                'char_4_type 23', 'char_4_type 24', 'char_4_type 25', 'char_4_type 3', 'char_4_type 4', 'char_4_type 5', 'char_4_type 6', 'char_4_type 7', 'char_4_type 8', 'char_4_type 9', 'char_5_type 2', 'char_5_type 3', 'char_5_type 4',
                'char_5_type 5', 'char_5_type 6', 'char_5_type 7', 'char_5_type 8', 'char_5_type 9', 'char_6_type 2', 'char_6_type 3', 'char_6_type 4', 'char_6_type 5', 'char_6_type 6', 'char_6_type 7', 'char_7_type 10', 'char_7_type 11',
                'char_7_type 12', 'char_7_type 13', 'char_7_type 14', 'char_7_type 15', 'char_7_type 16', 'char_7_type 17', 'char_7_type 18', 'char_7_type 19', 'char_7_type 2', 'char_7_type 20', 'char_7_type 21', 'char_7_type 22', 'char_7_type 23',
                'char_7_type 24', 'char_7_type 25', 'char_7_type 3', 'char_7_type 4', 'char_7_type 5', 'char_7_type 6', 'char_7_type 7', 'char_7_type 8', 'char_7_type 9', 'char_8_type 2', 'char_8_type 3', 'char_8_type 4', 'char_8_type 5', 'char_8_type 6',
                'char_8_type 7', 'char_8_type 8', 'char_9_type 2', 'char_9_type 3', 'char_9_type 4', 'char_9_type 5', 'char_9_type 6', 'char_9_type 7', 'char_9_type 8', 'char_9_type 9', 'activity_day_of_week', 'activity_month', 'activity_year', 'day_of_week', 'month', 'year']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Save the data to a temporary CSV file
        temp_file_path = 'temp_data.csv'
        df.to_csv(temp_file_path, index=False, compression='zip')

        # Call the make_predictions function
        predicted_data = make_predictions(
            file_path=temp_file_path,
            model=model,
            scaler=scaler,
            model_columns=model_columns
        )

        # Extract the predictions from the DataFrame
        predictions = predicted_data['Predicted_Outcome'].tolist()

        # Return predictions as JSON
        return jsonify({'predictions': predictions})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
