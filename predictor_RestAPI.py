'''RESTful Test Evaluator Utilizing Flask'''

import sys
import json
from flask import Flask

from config import HELP_FILE, PREDICTOR_NAME, SUPPORTED_REQUEST_FORMATS, SUPPORTED_RESPONSE_FORMATS
from error_checking_functions import *
from schema_validation import *
from evo2_utils import *
from predictor_content_handler import decode_request, encode_response

# --- Flask App and Central Error Handler ---
app = Flask(__name__)
# One of these works to maintain order when using jsonify()
app.config["JSON_SORT_KEYS"] = False
app.json.sort_keys = False

def create_error_response(error_key, messages, status_code):
    """ 
    Formats error response into a standardized JSON structure.
    
    Args:
        error_key (str): The category of the error (e.g. 'bad_prediction_request', 'prediction_request_failed').
        messages (list or str): A list of error message strings or a single message.
        status_code (int): Standard HTTP error status code based on the error.
    
    Returns:
        dict: A dictionary formatted for the standardized JSON error response.
    """
    if not isinstance(messages, list):
        messages = [str(messages)]
    error_payload = {"error": [{error_key: msg} for msg in messages]}
    print(error_payload)
    return error_payload, status_code

@app.errorhandler(APIError)
def handle_api_error(error):
    """This single handler catches all of our custom API errors."""
    # Get raw payload and status code
    payload, status_code = create_error_response(error.error_key, error.message, error.status_code)
    
    return encode_response(
        payload, 
        status_code=status_code,
        isError=True,
        predictor_name=PREDICTOR_NAME)
    

@app.after_request
def after_request_callback(response):
    """This function runs after each request is processed."""
    print(f"\n--- Sending predictions back to Evaluator. ---")
    print(f"--- Request Complete. {PREDICTOR_NAME} Predictor is listening on http://{predictor_ip}:{predictor_port} ---\n")
    return response

# --- API Endpoints ---
@app.route('/formats', methods=['GET'])
def formats_endpoint():
    """Provides the Predictor's supported formats"""
    supported_fmts = {
        "predictor_supported_request_formats": SUPPORTED_REQUEST_FORMATS,
        "predictor_supported_response_formats": SUPPORTED_RESPONSE_FORMATS
    }
    try:
        return encode_response(
            supported_fmts,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error serializing supported format for /format endpoint: {e}")

@app.route('/help', methods=['GET'])
def help_endpoint():
    """Provides the Predictor's help/metadata information."""
    try:
        with open(HELP_FILE, 'r') as f:
            help_data = json.load(f)
        return encode_response(
            help_data,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    except Exception as e:
        raise ServerError(f"Error reading help file: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """The main endpoint for receiving sequences and returning predictions."""
    
    try:
        #Decode incoming request, using the headers or JSON default
        evaluator_request = decode_request(SUPPORTED_REQUEST_FORMATS)
            
        # Validate the payload using the imported function
        # These functions will raise an APIError on failure,
        # which will be caught automatically by @app.errorhandler
        validate_request_payload(evaluator_request)

        # Preprocess the data using the imported function
        sequences = preprocess_data(evaluator_request)
        readout_type = evaluator_request.get('readout')
        prediction_ranges = evaluator_request.get('prediction_ranges')
        if prediction_ranges is not None:
            print("Prediction ranges will be used to return the request")

        if readout_type in ["track"]:
            json_return = {
                'predictor_name': PREDICTOR_NAME,
                "bin_size": 1,
                # Prediction task is an array of objects for all requested tasks
                "prediction_tasks": [],
            }

        if readout_type in ["point"]:
            json_return = {
                'predictor_name': PREDICTOR_NAME,
                # Prediction task is an array of objects for all requested tasks
                "prediction_tasks": [],
            }

        # Loop through all the prediction tasks
        for prediction_task in evaluator_request["prediction_tasks"]:
            #TO DO if scale is the same for all requested tasks only make 1 prediction
            request_type = prediction_task["type"]
            cell_type = prediction_task["cell_type"]
            scale_requested = prediction_task.get("scale", None)
            
            task_prediction, scale_actual = predict_evo2(sequences, readout_type, scale_requested, prediction_ranges)
            # Create structured response for the evaluator
            current_prediction_task = {
                "name": prediction_task["name"],
                "type_requested": request_type,
                "type_actual": request_type,  # Evo2 can return predictions for any request
                "cell_type_requested": cell_type,
                "cell_type_actual": cell_type,  # Evo2 can return predictions for any request
                "species_requested": prediction_task["species"],
                "species_actual": prediction_task["species"],
                "scale_prediction_requested": scale_requested,
                "scale_prediction_actual": scale_actual,
                "predictions": task_prediction,
            }

            # Append results for current prediction task to the main JSON object
            json_return["prediction_tasks"].append(current_prediction_task)

        return encode_response(
            json_return,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS)
    
    except Exception as e:
        # If it's already an APIError, re-raise it for the handler
        if isinstance(e, APIError):
            raise e
        # Otherwise, wrap the unknown error in a ServerError
        raise ServerError(f"An unexpected internal error occurred: {e}.")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print(f"Invalid arguments! Arguments must have: <container image/python script> <ip_address> <port>")
    #     sys.exit(1)
        
    # predictor_ip = sys.argv[1]
    # predictor_port = int(sys.argv[2])
    # app.run(host=predictor_ip, port=predictor_port)

    with open('/scratch/iluthra/evaluator_message_more_complex.json', 'r') as f:
        evaluator_request = json.load(f)

    print(evaluator_request)
    #validate_request_payload(evaluator_request)
    sequences = preprocess_data(evaluator_request)
    readout_type = evaluator_request.get('readout')
    prediction_ranges = evaluator_request.get('prediction_ranges')

    for prediction_task in evaluator_request["prediction_tasks"]:
        scale_requested = prediction_task.get("scale", None)
        task_prediction, scale_actual = predict_evo2(sequences, readout_type, scale_requested, prediction_ranges)
        # print(f"Task: {prediction_task['name']}")
        # print(f"Scale: {scale_actual}")
        # print(f"Predictions: {task_prediction}")