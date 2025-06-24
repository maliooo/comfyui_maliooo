import os
import omegaconf
import httpx
from volcenginesdkarkruntime import Ark
import time

conf = omegaconf.OmegaConf.load(os.path.join(os.path.dirname(__file__), "config.yaml"))
print(conf)


def get_ark_response(prompt, model_id, timeout=60*3, model_type=""):
    """调用火山的文本大模型"""

    if model_type == "DeepSeek-R1":
        timeout = 60*8

    t1 = time.time()
    client = Ark(    
        # The output time of the reasoning model is relatively long. Please increase the timeout period.
        api_key = conf.ARK["api-key"],
        timeout=httpx.Timeout(timeout=timeout),
    )

    print(f"----- 调用火山ARK模型-{model_type}, standard request -----")  # 不是流式请求
    completion = client.chat.completions.create(
        model = model_id,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    print(f"----- 调用火山ARK模型-{model_type}, 耗时: {(time.time()-t1):.2f}秒 -----\n")

    # print(completion.choices[0].message.reasoning_content)  # deepseek-R1 模型有这个字段
    # print("-"*100)
    # print(completion.choices[0].message.content)
    message_info = dict(completion.choices[0].message)

    return {
        "thinking": completion.choices[0].message.reasoning_content if "reasoning_content" in message_info else "",
        "text": completion.choices[0].message.content if "content" in message_info else "",
        "json": completion.model_dump_json()
    }



class Malio_ARK_LLM_Answer:
    """调用LLM大语言模型进行回答"""
    # 用于将火山ARK 输入的 model_type 转换为 model_id
    model_type_2_model_id = {
        "DeepSeek-V3" : "ep-20250205114920-z92xz",
        "DeepSeek-R1" : "ep-20250205101806-drpzs",
        "DeepSeek-R1-Distill-Qwen-32B" : "ep-20250212163446-9h7nn",
        "moonshot-V1" : "ep-20240530075612-fcqvj",
        "Doubao-1.5-pro-32k": "ep-20250212163653-9xd65"
    }

    def __init__(self) -> None:
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        model_type_list = list(Malio_ARK_LLM_Answer.model_type_2_model_id.keys())
        return {
            "required": {
                "model_type":(model_type_list, dict(default=model_type_list[0])),
                "model_id": ("STRING", dict(tooltip="模型ID", default="")),
                "llm_prompt": ("STRING", dict(tooltip="输入的文本")),
                "fallback": ("STRING", dict(tooltip="调用失败返回的回答", default="")),
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

    RETURN_TYPES = ( "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("llm_answer", "llm_prompt", "llm_info", "other_info", "llm_thinking")
    FUNCTION = "predict"

    def predict(self, model_type, model_id, llm_prompt, fallback, replace_text_1, replace_flag_1, replace_text_2="", replace_flag_2="", replace_text_3="", replace_flag_3="", replace_text_4="", replace_flag_4=""):
        # -----------------  1. 替换标签  -----------------
        if str(replace_flag_1).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_1, replace_text_1)
        if str(replace_flag_2).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_2, replace_text_2)
        if str(replace_flag_3).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_3, replace_text_3)
        if str(replace_flag_4).strip() != "":
            llm_prompt = llm_prompt.replace(replace_flag_4, replace_text_4)


        # -----------------  2. 调用火山ARK的模型  -----------------
        if model_id == "":
            model_id = self.model_type_2_model_id[model_type]
        try:
            llm_answer = get_ark_response(llm_prompt, model_id=model_id, model_type=model_type)
            # print(f"调用火山ARK的{model_type}模型进行LLM, 返回结果为: {llm_answer['text']}")
            # print(f"请求响应为: {llm_answer['json']}")
            
            return (llm_answer["text"], llm_prompt, llm_answer["json"], "调用正常", llm_answer["thinking"])
        except Exception as e:
            print(f"调用火山ARK的{model_type}模型进行LLM, 出现异常: {e} , 文件路径为{__file__}")
            return (
                fallback, 
                llm_prompt, 
                "", 
                f"调用火山ARK的{model_type}模型进行LLM, 出现异常: {e} , 文件路径为{__file__}",
                ""
            )