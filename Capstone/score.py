import json
import joblib
import numpy as np
import os
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"account_age_days": pd.Series([0.0], dtype="float32"), "transaction_amt": pd.Series([0.0], dtype="float32"), "transaction_adj_amt": pd.Series([0.0], dtype="float32"), "historic_velocity": pd.Series([0.0], dtype="float32"), "ip_address": pd.Series(["example_value"], dtype="object"), "user_agent": pd.Series(["example_value"], dtype="object"), "email_domain": pd.Series(["example_value"], dtype="object"), "phone_number": pd.Series(["example_value"], dtype="object"), "billing_city": pd.Series(["example_value"], dtype="object"), "billing_postal": pd.Series([0.0], dtype="float32"), "billing_state": pd.Series(["example_value"], dtype="object"), "card_bin": pd.Series([0.0], dtype="float32"), "currency": pd.Series(["example_value"], dtype="object"), "cvv": pd.Series(["example_value"], dtype="object"), "signature_image": pd.Series(["example_value"], dtype="object"), "transaction_type": pd.Series(["example_value"], dtype="object"), "transaction_env": pd.Series(["example_value"], dtype="object"), "EVENT_TIMESTAMP": pd.Series(["2000-1-1"], dtype="datetime64[ns]"), "applicant_name": pd.Series(["example_value"], dtype="object"), "billing_address": pd.Series(["example_value"], dtype="object"), "merchant_id": pd.Series(["example_value"], dtype="object"), "locale": pd.Series(["example_value"], dtype="object"), "tranaction_initiate": pd.Series(["example_value"], dtype="object"), "days_since_last_logon": pd.Series([0.0], dtype="float32"), "inital_amount": pd.Series([0.0], dtype="float32")})
output_sample = np.array(["example_value"])
method_sample = StandardPythonParameterType("predict")

# Called when the service is loaded
def init():
    global model
    # Get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)
    

# Called when a request is received
@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data)
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        if isinstance(result, pd.DataFrame):
            result = result.values
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    # Get the input data as a numpy array
    
    #data = np.array(json.loads(raw_data)['data'])
    #print('np '+data)
    #df = pd.DataFrame(raw_data['data'])
    # Get a prediction from the model
    #predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    #return predictions.tolist()