from .nodes import JsonResultGenerator, LLMLocalLoader, ClearVRAM, StringPatternEnforcer, StringCombiner, StringScraper, WriteToJson, DeleteFile, DownloadHuggingFaceModel, PullOllamaModel, UpdateLLMResultToJson, LoadFileNode, SelectItemByIndexNode, SelectItemByKeyNode, JSONToListNode, MathOperationNode
from .metrics import F1ScoreNode, AccuracyNode


NODE_CLASS_MAPPINGS = {
    "LoadFileNode": LoadFileNode,
    "SelectItemByIndexNode": SelectItemByIndexNode,
    "SelectItemByKeyNode": SelectItemByKeyNode,
    "JSONToListNode": JSONToListNode,
    "MathOperationNode": MathOperationNode,
    "F1ScoreNode": F1ScoreNode,
    "AccuracyNode": AccuracyNode,
    "UpdateLLMResultToJson": UpdateLLMResultToJson,
    "PullOllamaModel": PullOllamaModel,
    "DownloadHuggingFaceModel": DownloadHuggingFaceModel,
    "DeleteFile": DeleteFile,
    "WriteToJson": WriteToJson,
    "StringScraper": StringScraper,
    "StringCombiner": StringCombiner,
    "StringPatternEnforcer": StringPatternEnforcer,
    "ClearVRAM": ClearVRAM,
    "LLMLocalLoader": LLMLocalLoader,
    "JsonResultGenerator": JsonResultGenerator,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadFileNode": "Load File",
    "SelectItemByIndexNode": "Select Item by Index",
    "SelectItemByKeyNode": "Select Item by Key",
    "JSONToListNode": "JSONToListNode",
    "MathOperationNode": "MathOperationNode",
    "F1ScoreNode": "F1ScoreNode",
    "AccuracyNode": "AccuracyNode",
    "UpdateLLMResultToJson": "UpdateLLMResultToJson",
    "PullOllamaModel": "PullOllamaModel",
    "DownloadHuggingFaceModel": "DownloadHuggingFaceModel",
    "DeleteFile": "DeleteFile",
    "WriteToJson": "WriteToJson",
    "StringScraper": "StringScraper",
    "StringCombiner": "StringCombiner",
    "StringPatternEnforcer": "StringPatternEnforcer",
    "ClearVRAM": "ClearVRAM",
    "LLMLocalLoader": "LLMLocalLoader",
    "JsonResultGenerator": "JsonResultGenerator",

}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


'''
There are a few rules to understand here:

In the first line, the word right after the dot must be your script name.

After import in the first line, you must have the list of all the classes you defined in your python script, separated with commas.

If you have multiple scripts, you must write that line several times, one per script.


On the contrary, for the other lines, you only need one line.

The node class mappings line must include all the class mappings you defined in your script. Just copy-paste them and separate them with a comma.

The display name mappings works similarly.

These two lines are mandatory in the init file. Note that they overwrite whatever was defined in the original script. So as soon as your custom node is in its own folder, you must define the node/class name pairs in the init file.

You should never touch the “all line”.
'''


