from openai import OpenAI
from PIL import Image
import json
import os
import base64
import omegaconf

conf = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
print(conf)

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_qwen_vl_response(base64_image, prompt, model_type="qwen-vl-plus"):
    """调用通义千问的图文大模型"""
    # https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-vl-plus-api
    client = OpenAI(
        api_key = conf.qwen["api-key"],
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model = model_type,
        messages = [
            {
              "role": "user",
              "content": [
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                  }
                },
                {
                  "type": "text",
                  "text": prompt
                }
              ]
            }
        ],
        max_tokens=1024,
        temperature=1.0,
        )
    # print(completion.model_dump_json())
    json_res = json.loads(completion.model_dump_json())
    content = json_res["choices"][0]["message"]["content"]
    return {
        "text": content,
        "json": completion.model_dump_json()
    }

def get_qwen_response(prompt, model_type="qwen-plus"):
    """调用通义千问的文本大模型"""
    client = OpenAI(
        api_key = conf.qwen["api-key"],
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",

    )
    completion = client.chat.completions.create(
        model=model_type,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}],
        max_tokens=255,
        temperature=1.0,
        
        )
    json_res = json.loads(completion.model_dump_json())
    print(json_res)
    content = json_res["choices"][0]["message"]["content"]
    # message_info = dict(completion.choices[0].message)
    return {
        "text": content,
        "json": completion.model_dump_json(),
        "thinking": completion.choices[0].message.reasoning_content if "reasoning_content" in json_res["choices"][0]["message"] else "",

    }




class Malio_Merge_Tags_By_Onerformer_ADE20K:
    """"合并SEG的标签结果"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "style_image_tags":("LIST",),
                "style_image_tags_percentage":("LIST",),
                "image_tags":("LIST",),
                "image_tags_percentage":("LIST",),
                "min_percentage": ("FLOAT", dict(default=0.01, min=0, max=1, step=0.01)),
                "max_nums": ("INT", dict(default=5, min=1, max=15, step=1, tooltip="最多保留的标签数量")),
            }
        }

    RETURN_TYPES = ( "LIST", "LIST", "STRING")
    RETURN_NAMES = ("merge_tags", "merge_tags_percentage", "merge_tags_str")
    FUNCTION = "merge_tags"

    def merge_tags(self, style_image_tags, style_image_tags_percentage, image_tags, image_tags_percentage, min_percentage=0.01, max_nums=5):
        """合并两张图片的标签, 取交集"""
        merge_tags = []
        merge_tags_percentage = []

        for tag, percentage in zip(style_image_tags, style_image_tags_percentage):
            # 取交集, 并且过滤掉小于min_percentage的标签
            if tag in image_tags and percentage > min_percentage:
                tag_index = image_tags.index(tag)
                if image_tags_percentage[tag_index] > min_percentage:
                    merge_tags.append(tag)
                    merge_tags_percentage.append(image_tags_percentage[tag_index])
        
        merge_tags = merge_tags[:max_nums]
        merge_tags_percentage = merge_tags_percentage[:max_nums]

        if len(merge_tags) == 0:
            merge_tags_str = ""
        else:
            merge_tags_str = ",".join(merge_tags)

        

        return (merge_tags, merge_tags_percentage, merge_tags_str)


class Malio_LLM_By_Qwen_VL:
    """调用Qwen的VL模型进行LLM"""

    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_base64":("STRING",),
                "model_type":(["qwen-vl-plus", "qwen-vl-plus-0809", "qwen-vl-max"], dict(default="qwen-vl-plus")),
                "llm_prompt": ("STRING", dict(tooltip="输入的文本")),
                "replace_text_1": ("STRING", dict(tooltip="将prompt中的标签替换为replace_text")),
                "replace_flag_1": ("STRING", dict(tooltip="替换prompt中的标志")),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ( "STRING", "STRING", "STRING")
    RETURN_NAMES = ("llm_answer", "llm_prompt", "llm_info")
    FUNCTION = "predict"

    def predict(self, image_base64, model_type, llm_prompt, replace_text_1, replace_flag_1):
        # if str(replace_text_1).strip() == "":
        #     print(f"调用Qwen的VL模型进行LLM, replace_text_1为空, 文件路径为{__file__}")
        #     return ("", llm_prompt, "")

        if str(replace_flag_1).strip() != "":
            # 替换标签
            llm_prompt = llm_prompt.replace(replace_flag_1, replace_text_1)
        # else:
        #     llm_prompt = str(llm_prompt).strip() + " " + replace_text_1

        if str(image_base64).strip() == ""  or str(llm_prompt).strip() == "":
            print(f"调用Qwen的VL模型进行LLM, 输入为空, 文件路径为{__file__}")
            return (replace_text_1, llm_prompt, "")

        
        llm_answer = get_qwen_vl_response(image_base64, llm_prompt, model_type=model_type)
        print(f"调用Qwen的VL模型进行LLM,请求的prompt为: {llm_prompt}")
        print(f"调用Qwen的VL模型进行LLM, 返回结果为: {llm_answer['text']}")
        print(f"请求响应为: {llm_answer['json']}")
        
        return (llm_answer["text"], llm_prompt, llm_answer["json"])

        

class Malio_LLM_Answer:
    """调用LLM大语言模型进行回答"""

    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type":(["qwen-plus", "qwen-turbo", "qwen-max"], dict(default="qwen-plus")),
                "llm_prompt": ("STRING", dict(tooltip="输入的文本")),
                "replace_text_1": ("STRING", dict(tooltip="将prompt中的标签替换为replace_text_1")),
                "replace_flag_1": ("STRING", dict(tooltip="等待替换标志replace_flag_1", default="replace_flag_1")),
                "replace_text_2": ("STRING", dict(tooltip="将prompt中的标签替换为replace_text_2", default="")),
                "replace_flag_2": ("STRING", dict(tooltip="等待替换标志replace_flag_2", default="replace_flag_2")),
                "replace_text_3": ("STRING", dict(tooltip="将prompt中的标签替换为replace_text_3", default="")),
                "replace_flag_3": ("STRING", dict(tooltip="等待替换标志replace_flag_3", default="replace_flag_3")),
                "replace_text_4": ("STRING", dict(tooltip="将prompt中的标签替换为replace_text_4", default="")),
                "replace_flag_4": ("STRING", dict(tooltip="等待替换标志replace_flag_4", default="replace_flag_4")),
            }
        }
    
    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    RETURN_TYPES = ( "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("llm_answer", "llm_prompt", "llm_info", "other_info")
    FUNCTION = "predict"

    def predict(self, model_type, llm_prompt, replace_text_1, replace_flag_1, replace_text_2="", replace_flag_2="", replace_text_3="", replace_flag_3="", replace_text_4="", replace_flag_4=""):
        # ----------------- 1. 替换标签 -----------------
        if str(replace_flag_1).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_1, replace_text_1)
        if str(replace_flag_2).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_2, replace_text_2)
        if str(replace_flag_3).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_3, replace_text_3)
        if str(replace_flag_4).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_4, replace_text_4)

        # ----------------- 2. 调用模型 -----------------
        try:
            llm_answer = get_qwen_response(llm_prompt, model_type=model_type)
            # print(f"调用Qwen的{model_type}模型进行LLM, 返回结果为: {llm_answer['text']}")
            # print(f"请求响应为: {llm_answer['json']}")
            
            return (llm_answer["text"], llm_prompt, llm_answer["json"], "调用正常")
        except Exception as e:
            print(f"调用Qwen的{model_type}模型进行LLM, 出现异常: {e} , 文件路径为{__file__}")
            return ("", llm_prompt, "", f"调用Qwen的{model_type}模型进行LLM, 出现异常: {e} , 文件路径为{__file__}")
        
