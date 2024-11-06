import base64
import io
import json
import re
import requests
from PIL import Image
import numpy as np
from openai import OpenAI, AzureOpenAI

class Chat:
    def __init__(self, model_name, apikey, baseurl) -> None:
        self.model_name = model_name
        self.apikey = apikey
        self.baseurl = baseurl

    def send(
        self,
        user_prompt,
        temperature,
        max_length,
        history,
        tools=None,
        is_tools_in_sys_prompt="disable",
        images=None,
        imgbb_api_key="",
        **extra_parameters,
    ):
        try:
            is_azure=False
            if images is not None:
                if imgbb_api_key == "" or imgbb_api_key is None:
                    imgbb_api_key = api_keys.get("imgbb_api")
                if imgbb_api_key == "" or imgbb_api_key is None:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        img_json.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}
                        })
                    user_prompt = img_json
                else:
                    img_json = [{"type": "text", "text": user_prompt}]
                    for image in images:
                        i = 255.0 * image.cpu().numpy()
                        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        url = "https://api.imgbb.com/1/upload"
                        payload = {"key": imgbb_api_key, "image": img_str}
                        response = requests.post(url, data=payload)
                        if response.status_code == 200:
                            result = response.json()
                            img_url = result["data"]["url"]
                            img_json.append({
                                "type": "image_url",
                                "image_url": {"url": img_url}
                            })
                        else:
                            return "Error: " + response.text
                    user_prompt = img_json

            # 将history中的系统提示词部分如果为空，就剔除
            for i in range(len(history)):
                if history[i]["role"] == "system" and history[i]["content"] == "":
                    history.pop(i)
                    break
            if "o1" in self.model_name:
                # 将history中的系统提示词部分的角色换成user
                for i in range(len(history)):
                    if history[i]["role"] == "system":
                        history[i]["role"] = "user"
                        history.append({"role": "assistant", "content": "好的，我会按照你的指示来操作"})
                        break
            openai_client = OpenAI(
                    api_key= self.apikey,
                    base_url=self.baseurl,
                )
            if "openai.azure.com" in self.baseurl:
                # 获取API版本
                api_version = self.baseurl.split("=")[-1].split("/")[0]
                # 获取azure_endpoint
                azure_endpoint = "https://"+self.baseurl.split("//")[1].split("/")[0]
                azure = AzureOpenAI(
                    api_key= self.apikey,
                    api_version=api_version,
                    azure_endpoint=azure_endpoint,
                )
                openai_client = azure
            new_message = {"role": "user", "content": user_prompt}
            history.append(new_message)
            print(history)
            if "o1" in self.model_name:
                if tools is not None:
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        tools=tools,
                        **extra_parameters,
                    )
                    while response.choices[0].message.tool_calls:
                        assistant_message = response.choices[0].message
                        response_content = assistant_message.tool_calls[0].function
                        print("正在调用" + response_content.name + "工具")
                        print(response_content.arguments)
                        results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                        print(results)
                        history.append(
                            {
                                "tool_calls": [
                                    {
                                        "id": assistant_message.tool_calls[0].id,
                                        "function": {
                                            "arguments": response_content.arguments,
                                            "name": response_content.name,
                                        },
                                        "type": assistant_message.tool_calls[0].type,
                                    }
                                ],
                                "role": "assistant",
                                "content": str(response_content),
                            }
                        )
                        history.append(
                            {
                                "role": "tool",
                                "tool_call_id": assistant_message.tool_calls[0].id,
                                "name": response_content.name,
                                "content": results,
                            }
                        )
                        response = openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            **extra_parameters,
                        )
                        print(response)
                elif is_tools_in_sys_prompt == "enable":
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        **extra_parameters,
                    )
                    response_content = response.choices[0].message.content
                    # 正则表达式匹配
                    pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                    while re.search(pattern, response_content, re.DOTALL) != None:
                        match = re.search(pattern, response_content, re.DOTALL)
                        tool = match.group(1)
                        parameters = match.group(2)
                        json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                        print("正在调用" + tool + "工具")
                        parameters = json.loads("{" + parameters + "}")
                        results = dispatch_tool(tool, parameters)
                        print(results)
                        history.append({"role": "assistant", "content": json_str})
                        history.append(
                            {
                                "role": "user",
                                "content": "调用"
                                + tool
                                + "工具返回的结果为："
                                + results
                                + "。请根据工具返回的结果继续回答我之前提出的问题。",
                            }
                        )
                        response = openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=history,
                            **extra_parameters,
                        )
                        response_content = response.choices[0].message.content
                else:
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        **extra_parameters,
                    )
                    print(response)
            elif tools is not None:
                response = openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    tools=tools,
                    max_tokens=max_length,
                    **extra_parameters,
                )
                while response.choices[0].message.tool_calls:
                    assistant_message = response.choices[0].message
                    response_content = assistant_message.tool_calls[0].function
                    print("正在调用" + response_content.name + "工具")
                    print(response_content.arguments)
                    results = dispatch_tool(response_content.name, json.loads(response_content.arguments))
                    print(results)
                    history.append(
                        {
                            "tool_calls": [
                                {
                                    "id": assistant_message.tool_calls[0].id,
                                    "function": {
                                        "arguments": response_content.arguments,
                                        "name": response_content.name,
                                    },
                                    "type": assistant_message.tool_calls[0].type,
                                }
                            ],
                            "role": "assistant",
                            "content": str(response_content),
                        }
                    )
                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": assistant_message.tool_calls[0].id,
                            "name": response_content.name,
                            "content": results,
                        }
                    )
                    try:
                        response = openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            **extra_parameters,
                        )
                        print(response)
                    except Exception as e:
                        print("tools calling失败，尝试使用function calling" + str(e))
                        # 删除history最后两个元素
                        history.pop()
                        history.pop()
                        history.append(
                            {
                                "role": "assistant",
                                "content": str(response_content),
                                "function_call": {
                                    "name": response_content.name,
                                    "arguments": response_content.arguments,
                                },
                            }
                        )
                        history.append({"role": "function", "name": response_content.name, "content": results})
                        response = openai_client.chat.completions.create(
                            model=self.model_name,
                            messages=history,
                            tools=tools,
                            temperature=temperature,
                            max_tokens=max_length,
                            **extra_parameters,
                        )
                        print(response)
                while response.choices[0].message.function_call:
                    assistant_message = response.choices[0].message
                    function_call = assistant_message.function_call
                    function_name = function_call.name
                    function_arguments = json.loads(function_call.arguments)
                    print("正在调用" + function_name + "工具")
                    results = dispatch_tool(function_name, function_arguments)
                    print(results)
                    history.append(
                        {
                            "role": "assistant",
                            "content": str(function_call),
                            "function_call": {"name": function_name, "arguments": function_arguments},
                        }
                    )
                    history.append({"role": "function", "name": function_name, "content": results})
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        tools=tools,
                        temperature=temperature,
                        max_tokens=max_length,
                        **extra_parameters,
                    )
                response_content = response.choices[0].message.content
                print(response)
            elif is_tools_in_sys_prompt == "enable":
                response = openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    **extra_parameters,
                )
                response_content = response.choices[0].message.content
                # 正则表达式匹配
                pattern = r'\{\s*"tool":\s*"(.*?)",\s*"parameters":\s*\{(.*?)\}\s*\}'
                while re.search(pattern, response_content, re.DOTALL) != None:
                    match = re.search(pattern, response_content, re.DOTALL)
                    tool = match.group(1)
                    parameters = match.group(2)
                    json_str = '{"tool": "' + tool + '", "parameters": {' + parameters + "}}"
                    print("正在调用" + tool + "工具")
                    parameters = json.loads("{" + parameters + "}")
                    results = dispatch_tool(tool, parameters)
                    print(results)
                    history.append({"role": "assistant", "content": json_str})
                    history.append(
                        {
                            "role": "user",
                            "content": "调用"
                            + tool
                            + "工具返回的结果为："
                            + results
                            + "。请根据工具返回的结果继续回答我之前提出的问题。",
                        }
                    )
                    response = openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=history,
                        temperature=temperature,
                        max_tokens=max_length,
                        **extra_parameters,
                    )
                    response_content = response.choices[0].message.content
            else:
                response = openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=history,
                    temperature=temperature,
                    max_tokens=max_length,
                    **extra_parameters,
                )
            response_content = response.choices[0].message.content
            history.append({"role": "assistant", "content": response_content})
        except Exception as ex:
            response_content = str(ex)
        return response_content, history