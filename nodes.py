from .config import config_key, config_path, current_dir_path, load_api_keys
from .functions import Chat
import openai

class LLMAPIChat:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("CUSTOM", {"forceInput": True}),  # Required model input
                "user_input": ("STRING", {"forceInput": True}),  # Required user input
                "temperature": ("FLOAT", {"default": 0.7}),
                "max_length": ("INT", {"default": 150}),
            },
            "optional": {
                "imgbb_api_key": ("STRING", {"default": ""}),
                "is_tools_in_sys_prompt": ("STRING", {"default": "disable"}),  # Options: "enable" or "disable"
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("response", "updated_history")
    FUNCTION = "generate_response"
    CATEGORY = "LLM Evaluation/api"

    def generate_response(
            self,
            model,
            user_input,
            temperature=0.7,
            max_length=150,
            history=None,
            tools=None,
            images=None,
            imgbb_api_key="",
            is_tools_in_sys_prompt="disable",
    ):
        if model is None or user_input is None:
            return ("Error: Model and user input are required.", history)

        # Initialize history if not provided
        if history is None:
            history = []

        try:
            # Call the send method on the Chat instance with the user input and additional parameters
            response_content, updated_history = model.send(
                user_prompt=user_input,
                temperature=temperature,
                max_length=max_length,
                history=history,
                tools=tools,
                images=images,
                imgbb_api_key=imgbb_api_key,
                is_tools_in_sys_prompt=is_tools_in_sys_prompt,
            )

            # Return the response and the updated history
            return (response_content, updated_history)

        except Exception as e:
            return (f"Error generating response: {e}", history)


class LLMAPILoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ("STRING", {"default": "gpt-4o-mini"}),
            },
            "optional": {
                "base_url": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "is_ollama": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CUSTOM",)
    RETURN_NAMES = ("model",)

    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "LLM Evaluation/api"

    def chatbot(self, model_name, base_url=None, api_key=None, is_ollama=False):
        if is_ollama:
            openai.api_key = "ollama"
            openai.base_url = "http://127.0.0.1:11434/v1/"
        else:
            api_keys = load_api_keys(config_path)
            if api_key != "":
                openai.api_key = api_key
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.api_key = api_keys.get("api_key")
            elif api_keys.get("openai_api_key") != "":
                openai.api_key = api_keys.get("openai_api_key")
            if base_url != "":
                openai.base_url = base_url
            elif model_name in config_key:
                api_keys = config_key[model_name]
                openai.base_url = api_keys.get("base_url")
            elif api_keys.get("base_url") != "":
                openai.base_url = api_keys.get("base_url")
            if openai.api_key == "":
                return ("请输入API_KEY",)
            if openai.base_url != "":
                if openai.base_url[-1] != "/":
                    openai.base_url = openai.base_url + "/"

        chat = Chat(model_name, openai.api_key, openai.base_url)
        return (chat,)




class LoadFileNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "example.txt"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load_file"
    CATEGORY = "LLM Evaluation"

    def load_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return (file.read(),)
        except FileNotFoundError:
            return (f"Error: File '{path}' not found.",)
        except Exception as e:
            return (f"Error reading file: {e}",)


import json

class SelectItemByIndexNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "index": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_item",)
    FUNCTION = "select_item_by_index"
    CATEGORY = "LLM Evaluation"

    def select_item_by_index(self, text, index):
        if text is None or index is None:
            return ("Error: Both text and index are required.",)

        try:
            # Parse the JSON text
            data = json.loads(text)

            # Check if the data is a list and index is within bounds
            if isinstance(data, list) and 0 <= index < len(data):
                item = data[index]
                return (json.dumps(item),)  # Output as JSON string
            else:
                return (f"Error: Index {index} is out of bounds or data is not a list.",)

        except json.JSONDecodeError:
            return ("Error: Invalid JSON format.",)
        except Exception as e:
            return (f"Error processing data: {e}",)


import json

class SelectItemByKeyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_value",)
    FUNCTION = "select_item_by_key"
    CATEGORY = "LLM Evaluation"

    def select_item_by_key(self, text, key):
        if text is None or key is None:
            return ("Error: Both text and key are required.",)

        try:
            # Parse the JSON text
            data = json.loads(text)

            # If the data is a list, treat each item as a dictionary and get the value by key
            if isinstance(data, list):
                values = [item.get(key, f"Key '{key}' not found") for item in data if isinstance(item, dict)]
                return (json.dumps(values),)  # Return all matching values as a list

            # If data is a dictionary, directly access the key
            elif isinstance(data, dict):
                if key in data:
                    return (json.dumps(data[key]),)
                else:
                    return (f"Error: Key '{key}' not found in object.",)

            return ("Error: JSON data must be a list or object.",)

        except json.JSONDecodeError:
            return ("Error: Invalid JSON format.",)
        except Exception as e:
            return (f"Error processing data: {e}",)



