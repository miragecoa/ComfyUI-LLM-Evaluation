from sklearn.metrics import accuracy_score, f1_score
import json
from .utils import AlwaysEqualProxy
any_type = AlwaysEqualProxy("*")

class F1ScoreNode:
    def __init__(self):
        pass

    OUTPUT_NODE = True
    @classmethod
    def IS_CHANGED(cls, model_name):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),  # Path to the JSON file
                "true_value_key": ("STRING", {"default": "answer"}),  # Key for true values
                "generated_response_key": ("STRING", {"default": "GeneratedResponse"}),  # Key for generated responses
            },
            "optional": {
                "flow": (any_type,),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("f1_score",)
    FUNCTION = "calculate_f1_score"
    CATEGORY = "Metrics"

    def calculate_f1_score(self, file_path, true_value_key, generated_response_key, flow):
        try:
            # Load the JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Ensure the data is a list
            if not isinstance(data, list):
                return ("Error: JSON data must be a list.",)

            # Extract true values and generated responses
            true_values = [item.get(true_value_key, "").strip() for item in data]
            generated_responses = [item.get(generated_response_key, "").strip() for item in data]
            print("True:",true_values)
            print("\nGenerated:", generated_responses)

            # Calculate F1 score (assuming binary classification; adjust average as needed)
            f1 = f1_score(true_values, generated_responses, average='micro')
            return (f1,)
        except FileNotFoundError:
            return ("Error: The specified file was not found.",)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON format in file.",)
        except Exception as e:
            return (f"Error calculating F1 score: {e}",)


class AccuracyNode:
    def __init__(self):
        pass

    OUTPUT_NODE = True
    @classmethod
    def IS_CHANGED(cls, model_name):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),  # Path to the JSON file
                "true_value_key": ("STRING", {"default": "answer"}),  # Key for true values
                "generated_response_key": ("STRING", {"default": "GeneratedResponse"}),  # Key for generated responses
            },
            "optional": {
                "flow": (any_type,),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("accuracy",)
    FUNCTION = "calculate_accuracy"
    CATEGORY = "Metrics"

    def calculate_accuracy(self, file_path, true_value_key, generated_response_key, flow):
        try:
            # Load the JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Ensure the data is a list
            if not isinstance(data, list):
                return ("Error: JSON data must be a list.",)

            # Extract true values and generated responses
            true_values = [item.get(true_value_key, "").strip() for item in data]
            generated_responses = [item.get(generated_response_key, "").strip() for item in data]

            # Calculate accuracy
            acc = accuracy_score(true_values, generated_responses, normalize=False)
            return (acc,)
        except FileNotFoundError:
            return ("Error: The specified file was not found.",)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON format in file.",)
        except Exception as e:
            return (f"Error calculating accuracy: {e}",)
