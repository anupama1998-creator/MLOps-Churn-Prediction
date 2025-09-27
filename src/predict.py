import os
import json
import joblib
import pandas as pd

# Global variable to hold the model.
# This is initialized once in the init() function.
model = None

def init():
    """
    This function is called when the service is initialized.
    It loads the model from the path where Azure ML mounts it.
    """
    global model
    
    # The environment variable AZUREML_MODEL_DIR points to the directory
    # where the registered model files are downloaded.
    # MLflow typically saves the model file as 'model.pkl'.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")


def run(raw_data):
    """
    This function is called for every inference request.
    It expects the raw_data to be a JSON string.
    """
    print(f"Received request: {raw_data}")
    
    try:
        # The standard input format is a JSON string with a "data" key,
        # which contains a list of records (dictionaries).
        data = json.loads(raw_data)["data"]
        
        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Use the loaded model to make predictions
        predictions = model.predict(df)
        
        # Get prediction probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[:, 1]
        else:
            # If not, create a list of Nones as placeholders
            probabilities = [None] * len(predictions)

        # Format the result as a list of dictionaries, one for each input record
        result = [
            {"prediction": int(pred), "probability": float(prob) if prob is not None else None}
            for pred, prob in zip(predictions, probabilities)
        ]
        
        # Return the list of predictions as a JSON response
        return result

    except Exception as e:
        # If an error occurs, return the error message in a structured way
        error = str(e)
        print(f"Error: {error}")
        return {"error": error, "message": "Failed to process request."}
