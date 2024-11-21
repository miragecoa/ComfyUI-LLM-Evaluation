import openai
import json
from .utils import AlwaysEqualProxy
any_type = AlwaysEqualProxy("*")
import subprocess
import threading
import time
import re
import sys
from tqdm import tqdm
import os
from huggingface_hub import snapshot_download, login
import folder_paths
import torch


class JsonResultGenerator:
    def __init__(self):
        pass

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always consider this node as changed to ensure it runs
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),  # Path to the JSON file
                "model_type": ("STRING", {"default": "pretrained"}),  # Model type
                "model_dtype": ("STRING", {"default": "torch.float16"}),  # Model data type
                "model_name": ("STRING", {"default": "Qwen/Qwen2-7B-Instruct"}),  # Model name
                "model_sha": ("STRING", {"default": "main"}),  # Model SHA
                "task_name": ("STRING", {"default": ""}),  # Task name for results
            },
            "optional": {
                "metric1_name": ("STRING", {"default": "F1"}),
                "metric1_score": ("FLOAT", {"forceInput": True}),
                "metric2_name": ("STRING", {"default": "Acc"}),
                "metric2_score": ("FLOAT", {"forceInput": True}),
                "metric3_name": ("STRING", {"default": ""}),
                "metric3_score": ("FLOAT", {"forceInput": True}),
                "metric4_name": ("STRING", {"default": ""}),
                "metric4_score": ("FLOAT", {"forceInput": True}),
                "metric5_name": ("STRING", {"default": ""}),
                "metric5_score": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("json_file_path", "json_content",)
    FUNCTION = "initialize_json"
    CATEGORY = "LLM Evaluation"

    def initialize_json(
        self, file_path, model_type, model_dtype, model_name, model_sha,
        task_name, metric1_name="", metric1_score=0.0, metric2_name="",
        metric2_score=0.0, metric3_name="", metric3_score=0.0,
        metric4_name="", metric4_score=0.0, metric5_name="", metric5_score=0.0
    ):
        try:
            # Initialize json_data with the given config
            json_data = {
                "config": {
                    "model_type": model_type,
                    "model_dtype": model_dtype,
                    "model_name": model_name,
                    "model_sha": model_sha
                },
                "results": {}
            }

            # If the file already exists, load its content
            if os.path.exists(file_path):
                with open(file_path, "r") as json_file:
                    existing_data = json.load(json_file)
                    # Preserve the existing config
                    json_data["config"] = existing_data.get("config", json_data["config"])
                    # Copy over existing results
                    json_data["results"] = existing_data.get("results", {})

            # Prepare the new task results
            if task_name:
                if task_name not in json_data["results"]:
                    json_data["results"][task_name] = {}

                # Add metrics only if names are provided
                if metric1_name and metric1_score:
                    json_data["results"][task_name][metric1_name] = metric1_score
                if metric2_name and metric2_score:
                    json_data["results"][task_name][metric2_name] = metric2_score
                if metric3_name and metric3_score:
                    json_data["results"][task_name][metric3_name] = metric3_score
                if metric4_name and metric4_score:
                    json_data["results"][task_name][metric4_name] = metric4_score
                if metric5_name and metric5_score:
                    json_data["results"][task_name][metric5_name] = metric5_score

            # Convert the JSON data to a string
            json_content = json.dumps(json_data, indent=4)

            # Save the JSON to the specified file path
            with open(file_path, "w") as json_file:
                json_file.write(json_content)

            return (file_path, json_content)
        except Exception as e:
            return (f"Error creating or updating JSON file: {e}", f"Error creating or updating JSON content: {e}")


class LLMLocalLoader:
    def __init__(self):
        self.id = hash(str(self))
        self.device = ""
        self.dtype = ""
        self.model_name_or_path = ""
        self.model = ""
        self.tokenizer = ""
        self.is_locked = False

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always consider this node as changed to ensure it runs
        return float("NaN")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name_or_path": ("STRING", {"default": ""}),
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {
                        "default": "auto",
                    },
                ),
                "dtype": (
                    ["float32", "float16","bfloat16", "int8", "int4"],
                    {
                        "default": "float32",
                    },
                ),
                "is_locked": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "CUSTOM",
        "CUSTOM",
    )
    RETURN_NAMES = (
        "model",
        "tokenizer",
    )

    FUNCTION = "chatbot"

    # OUTPUT_NODE = False

    CATEGORY = "LLM Evaluation"

    def chatbot(self, model_name_or_path, device, dtype, is_locked=True):
        self.is_locked = is_locked
        if self.is_locked == False:
            setattr(LLMLocalLoader, "IS_CHANGED", LLMLocalLoader.original_IS_CHANGED)
        else:
            # 如果方法存在，则删除
            if hasattr(LLMLocalLoader, "IS_CHANGED"):
                delattr(LLMLocalLoader, "IS_CHANGED")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        if (
            self.device != device
            or self.dtype != dtype
            or self.model_name_or_path != model_name_or_path
            or is_locked == False
        ):
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            # 对于 CPU 和 MPS 设备，不需要清空 CUDA 缓存
            elif self.device == "cpu" or self.device == "mps":
                gc.collect()
            self.model = ""
            self.tokenizer = ""
            self.model_name_or_path = model_name_or_path
            self.device = device
            self.dtype = dtype
            model_kwargs = {
                'device_map': device,
            }

            if dtype == "float16":
                model_kwargs['torch_dtype'] = torch.float16
            elif dtype == "bfloat16":
                model_kwargs['torch_dtype'] = torch.bfloat16
            elif dtype in ["int8", "int4"]:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=(dtype == "int8"), load_in_4bit=(dtype == "int4"))

            config = AutoConfig.from_pretrained(model_name_or_path, **model_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

            if config.model_type == "t5":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_kwargs)
            elif config.model_type in ["gpt2", "gpt_refact", "gemma", "llama", "mistral", "qwen2", "chatglm"]:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")
            self.model = self.model.eval()
        return (
            self.model,
            self.tokenizer,
        )

    @classmethod
    def original_IS_CHANGED(s):
        # 生成当前时间的哈希值
        hash_value = hashlib.md5(str(datetime.datetime.now()).encode()).hexdigest()
        return hash_value

class ClearVRAM:
    def __init__(self):
        pass

    OUTPUT_NODE = True  # Marks this node as an output node

    @classmethod
    def IS_CHANGED(cls, *args):
        # Always consider this node as changed to ensure it runs
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CONTROL",),  # A flow control input to connect with other nodes
            },
        }

    RETURN_TYPES = ("FLOW_CONTROL",)
    RETURN_NAMES = ("flow",)
    FUNCTION = "clear_vram"
    CATEGORY = "LLM Evaluation"

    def clear_vram(self, flow):
        try:
            # Clear PyTorch GPU cache
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("VRAM usage cleared successfully.")
            return (flow,)  # Return the flow control to continue execution
        except Exception as e:
            print(f"Error clearing VRAM: {e}")
            return (flow,)  # Return the flow control even if there's an error


class StringPatternEnforcer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),  # Input text to modify
                "pattern": ("STRING", {"default": "."}),  # Pattern to enforce or remove
                "position": (["left", "right"], {"default": "right"}),  # Position to enforce or remove the pattern
                "action": (["enforce", "remove"], {"default": "enforce"}),  # Action to perform
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_string",)
    FUNCTION = "modify_string"
    CATEGORY = "LLM Evaluation"

    def modify_string(self, text, pattern, position, action):
        try:
            if position == "left":
                if action == "enforce":
                    # Add the pattern to the left if it does not exist
                    if not text.startswith(pattern):
                        text = pattern + text
                elif action == "remove":
                    # Remove the pattern from the left if it exists
                    text = re.sub(f"^{re.escape(pattern)}", "", text)
            elif position == "right":
                if action == "enforce":
                    # Add the pattern to the right if it does not exist
                    if not text.endswith(pattern):
                        text = text + pattern
                elif action == "remove":
                    # Remove the pattern from the right if it exists
                    text = re.sub(f"{re.escape(pattern)}$", "", text)

            return (text,)
        except Exception as e:
            return (f"Error: {e}",)


class StringCombiner:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),  # First input string
                "string2": ("STRING", {"default": ""}),  # Second input string
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_string",)
    FUNCTION = "combine_strings"
    CATEGORY = "LLM Evaluation"

    def combine_strings(self, string1, string2):
        try:
            # Concatenate the two input strings
            combined_string = string1 + string2
            return (combined_string,)
        except Exception as e:
            return (f"Error: {e}",)


class StringScraper:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),  # Input text from which to remove the pattern
                "pattern": ("STRING", {"default": ""}),  # Regex pattern to remove from the string
            },
            "optional": {
                "left_trim": (["enable", "disable"], {"default": "enable"}),  # Option to trim left spaces
                "right_trim": (["enable", "disable"], {"default": "enable"}),  # Option to trim right spaces
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modified_string",)
    FUNCTION = "scrape_and_remove"
    CATEGORY = "LLM Evaluation"

    def scrape_and_remove(self, text, pattern, left_trim="enable", right_trim="enable"):
        try:
            # Use regex to remove the pattern from the text
            modified_string = re.sub(pattern, '', text)

            # Apply optional left and right trimming
            if left_trim == "enable":
                modified_string = modified_string.lstrip()
            if right_trim == "enable":
                modified_string = modified_string.rstrip()

            return (modified_string,)
        except Exception as e:
            return (f"Error: {e}",)




class WriteToJson:
    def __init__(self):
        pass

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, text, file_path, mode):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": ""}),  # Text to write to JSON
                "file_path": ("STRING", {"default": "output.json"}),  # Path to the JSON file
                "mode": ("STRING", {"choices": ["append", "overwrite"], "default": "append"}),  # Mode selection
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "write_to_json"
    CATEGORY = "LLM Evaluation"

    def write_to_json(self, text, file_path, mode):
        try:
            # Convert the text to a Python object (assuming it's a JSON-like string)
            data_to_write = json.loads(text)

            # Check if the file exists and read existing data if in append mode
            if mode == "append" and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
                    if isinstance(existing_data, list) and isinstance(data_to_write, list):
                        data_to_write = existing_data + data_to_write
                    elif isinstance(existing_data, list):
                        data_to_write = existing_data + [data_to_write]
                    elif isinstance(data_to_write, list):
                        data_to_write = [existing_data] + data_to_write
                    else:
                        data_to_write = [existing_data, data_to_write]

            # Write the data to the file (overwrite or append mode)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data_to_write, file, indent=4)

            return (file_path,)

        except json.JSONDecodeError:
            return ("Error: Invalid JSON format in text input.",)
        except Exception as e:
            return (f"Error writing to JSON file: {e}",)


class DeleteFile:
    def __init__(self):
        pass

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, any, filepath):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": (any_type,),
                "filepath": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "delete_file"
    CATEGORY = "LLM Evaluation"

    def delete_file(self, any, filepath):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"File {filepath} deleted successfully.")
            else:
                print(f"File {filepath} does not exist.")

            # Return the flow input unchanged to allow for chaining
            return (any,)

        except Exception as e:
            print(f"Error deleting file: {e}")
            return (any,)


class DownloadHuggingFaceModel:
    def __init__(self):
        pass

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model_name, token):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": "meta-llama/Llama-3.2-1B"}),
                "token": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_path",)
    FUNCTION = "download_model"
    CATEGORY = "LLM Evaluation/Download"

    def download_model(self, model, token):
        try:
            # Log in to Hugging Face using the provided token
            if token:
                login(token=token)

            # Extract model name and define model path
            model_name = model.rsplit('/', 1)[-1]
            model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)

            # Check if the model already exists, otherwise download
            if not os.path.exists(model_path):
                print(f"Downloading model to: {model_path}")
                snapshot_download(
                    repo_id=model,
                    local_dir=model_path,
                    local_dir_use_symlinks=False
                )

            # Return the path to the model
            return (model_path,)

        except Exception as e:
            return (f"Error downloading model: {e}",)




class PullOllamaModel:
    def __init__(self):
        self.progress = ""

    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model_name, token):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama3.2"}),  # User input for the model name
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status_message", "model_name")
    FUNCTION = "pull_model"
    CATEGORY = "LLM Evaluation/Download"

    @classmethod
    def IS_CHANGED(cls, model_name):
        # Return NaN to force the node to always be considered as changed
        return float("NaN")

    def pull_model(self, model_name):
        self.progress = "Starting to pull model...\n"

        try:
            # Construct the command
            command = f"ollama pull {model_name}"
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Initialize tqdm progress bar
            progress_bar = tqdm(total=100, desc="Pulling Model", unit="%")

            # Read the output line by line and update the progress bar
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    # Update the progress bar based on the percentage found
                    percentage = self._extract_percentage(output)
                    if percentage is not None:
                        progress_bar.n = percentage
                        progress_bar.refresh()

            # Wait for the command to finish and close the progress bar
            process.wait()
            progress_bar.close()

            if process.returncode == 0:
                self.progress += "\nSuccessfully pulled model."
            else:
                self.progress += f"\nError: {process.stderr.read()}"

        except Exception as e:
            self.progress += f"\nError executing command: {e}"

        # Return the final status message and model name
        return (self.progress, model_name)

    def _extract_percentage(self, output):
        # Extract the percentage from the command output
        match = re.search(r"(\d+)%", output)
        if match:
            return int(match.group(1))
        return None


class UpdateLLMResultToJson:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),  # Path to the JSON file
                "index": ("INT", {"forceInput": True}),  # Index of the element to update
                "key": ("STRING", {"default": "GeneratedResponse"}),  # Key to update
                "text": ("STRING", {"forceInput": True}),  # New text value
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_message",)
    FUNCTION = "update_json_file_by_index"
    CATEGORY = "LLM Evaluation"

    def update_json_file_by_index(self, file_path, index, key, text):
        try:
            # Load the JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Ensure the data is a list
            if not isinstance(data, list):
                return ("Error: JSON data must be a list.",)

            # Check if the index is within bounds
            if index < 0 or index >= len(data):
                return (f"Error: Index {index} is out of bounds.",)

            # Update the specified key with the given text
            if isinstance(data[index], dict):
                data[index][key] = text
            else:
                return (f"Error: Element at index {index} is not a dictionary.",)

            # Save the updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

            return ("JSON file successfully updated.",)

        except FileNotFoundError:
            return ("Error: The specified file was not found.",)
        except json.JSONDecodeError:
            return ("Error: Invalid JSON format in file.",)
        except Exception as e:
            return (f"Error processing file: {e}",)


class MathOperationNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("INT", {"forceInput": True}),  # First number (force input)
                "b": ("INT", {"forceInput": True}),  # Second number (force input)
                "operation": (
                    "STRING",
                    {
                        "default": "add",
                        "choices": ["add", "subtract", "multiply", "divide"]
                    }
                ),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "perform_operation"
    CATEGORY = "LLM Evaluation"

    def perform_operation(self, a, b, operation):
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return ("Error: Division by zero.",)
                result = a / b
            else:
                return ("Error: Invalid operation.",)

            return (result,)
        except Exception as e:
            return (f"Error: {e}",)


class JSONToListNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_string": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("converted_list",)
    FUNCTION = "convert_to_list"
    CATEGORY = "LLM Evaluation"

    def convert_to_list(self, json_string):
        try:
            # Attempt to load the string as JSON
            data = json.loads(json_string)
            if isinstance(data, list):
                return (data,)  # Return the list
            else:
                return ("Error: JSON string must represent a list.",)
        except json.JSONDecodeError as e:
            return (f"Error parsing JSON: {e}",)



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
    MAX_KEYS = 5  # You can adjust this to set the maximum number of keys allowed

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Create dynamic key inputs
        key_inputs = {
            f"key{i}": ("STRING", {"default": ""}) for i in range(1, cls.MAX_KEYS + 1)
        }

        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "optional": key_inputs
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_value",)
    FUNCTION = "select_item_by_keys"
    CATEGORY = "LLM Evaluation"

    def select_item_by_keys(self, text, **kwargs):
        if text is None:
            return ("Error: Text input is required.",)

        try:
            # Parse the JSON text
            data = json.loads(text)
            keys = [kwargs.get(f"key{i}") for i in range(1, self.MAX_KEYS + 1) if kwargs.get(f"key{i}")]

            if not keys:
                return ("Error: At least one key must be provided.",)

            # Collect values based on keys
            collected_values = []

            if isinstance(data, list):
                for key in keys:
                    values = [item.get(key, f"Key '{key}' not found") for item in data if isinstance(item, dict)]
                    collected_values.extend(values)
            elif isinstance(data, dict):
                for key in keys:
                    value = data.get(key, f"Key '{key}' not found")
                    collected_values.append(value)
            else:
                return ("Error: JSON data must be a list or object.",)

            # Combine all values into a single string, separated by newline
            combined_value = "\n".join(str(value) for value in collected_values)
            return (combined_value,)

        except json.JSONDecodeError:
            return ("Error: Invalid JSON format.",)
        except Exception as e:
            return (f"Error processing data: {e}",)




