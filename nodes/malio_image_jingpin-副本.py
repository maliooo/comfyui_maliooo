# 竞品网站图片生成
import base64
import datetime
import http.client
import io
import json
import os
import random
import re
import time

import numpy as np
import requests
import torch
from PIL import Image
from tqdm import tqdm
import omegaconf
from urllib3.exceptions import InsecureRequestWarning

from .components.fields import Field
from .utils import get_comfyui_images, tensor2pil

# 禁用不安全请求的警告
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class Maliooo_Get_SuappImage:
    """Load an image from the given URL"""

    # 来源：\custom_nodes\comfy_mtb\nodes\image_processing.py

    def __init__(self):
        yaml_path = r"/data/ai_draw_data/suapp_config.yaml"
        config = omegaconf.OmegaConf.load(yaml_path)
        print("读取配置文件成功")
        print("suapp.authentication: ", config.suapp.authentication)
        print("-------------------")
        print("suapp.cookie: ", config.suapp.cookie)
        print("-------------------")
        self.authentication = config.suapp.authentication
        self.cookie = config.suapp.cookie

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "需要suapp处理的底图"}),
                "场景": Field.combo(["家装", "工装", "建筑", "景观"]),
                "style": Field.combo(['新中式', '现代', '极简', '奶油风', '轻奢', '法式', '侘寂风', '美式', '中式', '简欧', '北欧', '日式', '原木风']),
                "正向提示词": ("STRING", {"default": ""}),
                "authentication": ("STRING", {"default": "", "tooltip": "默认为空,会读config文件"}),
                "cookie": ("STRING", {"default": "", "tooltip": "默认为空,会读config文件"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = (
        "IMAGE",
        "场景-风格"
    )
    FUNCTION = "get_suapp_image"
    CATEGORY = "🐼malio/竞品"

    def get_suapp_image(
        self,
        images,
        style: str = "",
        场景: str = "",
        authentication: str = "",
        cookie: str = "",
        正向提示词: str = "",

    ):
        """返回suapp生成的图片的url"""

        print(f"开始进行suapp生成图片， 传入图片数量：{len(images)}， 场景：{场景}， 风格：{style}， 正向提示词：{正向提示词}")

        p_prompt = 正向提示词


        # ================== 1. 构建请求的信息 ==================
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 SketchUp Pro/23.1 (Mac) Safari/537.36"
        current_time = datetime.datetime.now()
        # Set the headers as provided in the curl request
        headers = {
            "authentication": authentication if authentication else self.authentication,
            "user-agent": user_agent,
            "cookie": cookie if cookie else self.cookie,
            "origin": "https://ai.sketchupbar.com",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "cors",
            "sec-fetch-dest": "empty",
            "referer": "https://ai.sketchupbar.com/public/index.html?r={}".format(
                current_time.second
            ),
            "Content-Type": "application/json",
            "sec-ch-ua-platform": "Windows",

        }
        print(f"请求头headers: {headers}")


        # znzmo : ['新中式' '现代' '极简' '奶油风' '轻奢' '法式' '侘寂风' '美式' '中式' '简欧' '北欧' '日式' '原木风']
        # SUAPP飞书对应文档： https://hr7ljf0dhp.feishu.cn/sheets/HaogszO3ShWCBYtSLJvce0nonFg?sheet=TXD8LY
        znzmo_2_suapp = {
            "家装-新中式": "中国风",
            "家装-现代": "现代风",
            "家装-极简": "色彩风",
            "家装-奶油风": "奶油风",
            "家装-轻奢": "轻奢风",
            "家装-法式": "古典风",
            "家装-侘寂风": "侘寂风",
            "家装-美式": "古典风",
            "家装-中式": "中国风",
            "家装-简欧": "古典风",
            "家装-北欧": "现代风",
            "家装-日式": "原木风",
            "家装-原木风": "原木风",
            "家装-工业风": "工业风",
            "工装-办公空间": "办公空间",
            "工装-餐饮空间": "餐厅空间",
            "工装-酒店民宿": "酒店空间",
            "工装-商业空间": "商业空间",
            "工装-展厅": "展厅空间",
            "建筑-城乡规划": "城市透视",
            "建筑-居住建筑": "别墅建筑",
            "建筑-商业建筑": "商业建筑",
            "建筑-办公建筑": "办公建筑",
            "建筑-乡村建筑": "乡村建筑",
            
        }

        

        render_style_dict = {
            "商业建筑": "Architecture",
            "中式建筑": "ArchitectureCN",
            "别墅建筑": "ArchiVilla",
            "乡村建筑": "ArchiRural",
            "工业建筑": "ArchiIndustrial",
            "教育建筑": "ArchiEDU",
            "办公建筑": "ArchiOffice",
            "住宅建筑": "ArchiResidential",
            "酒店建筑": "ArchiHotel",
            "观演建筑": "ArchiTheatrical",
            "城市透视": "UrbanPerspective",
            "城市鸟瞰": "UrbanAerial",
            "总平面图": "MasterPlan",
            "现代风": "InteriorDesign",
            "奶油风": "InteriorCream",
            "侘寂风": "InteriorWabi",
            "中国风": "InteriorCN",
            "工业风": "InteriorIndustrial",
            "轻奢风": "InteriorLuxury",
            "暗黑风": "InteriorGray",
            "原木风": "InteriorWood",
            "色彩风": "InteriorColor",
            "古典风": "InteriorNeoclassical",
            "中古风": "InteriorRetro",
            "乡村风": "InteriorRural",
            "异域风": "InteriorExotic",
            "赛博风": "InteriorCyber",
            "彩平图": "ColorFloorPlan",
            "办公空间": "InteriorOffice",
            "餐厅空间": "InteriorRestaurant",
            "酒店空间": "InteriorHotel",
            "商业空间": "InteriorCommercial",
            "车站空间": "InteriorStation",
            "幼儿园空间": "InteriorKids",
            "酒吧空间": "InteriorBar",
            "婚礼空间": "InteriorWedding",
            "图书馆空间": "InteriorLibrary",
            "展厅空间": "InteriorExhibition",
            "健身房空间": "InteriorGYM",
            "舞台空间": "InteriorAuditorium",
            "公园景观": "LandscapePark",
            "园区景观": "LandscapeDesign",
            "游乐场景观": "LandscapePlayground",
            "庭院景观": "LandscapeCourtyard",
            "大门景观": "LandscapeGate",
            "桥梁景观": "LandscapeBridge",
            "手工模型": "ManualModel",
            "建筑马克笔": "ArchiMarker",
            "景观马克笔": "LandscapeMarker",
            "室内马克笔": "InteriorMarker",
            "建筑手绘": "ArchiSketch",
            "草图手绘": "SimpleSketch",
            "绘画艺术": "PaintingArt",
            "扁平插画": "Illustration",
            "古风彩绘": "ColorPainting",
        }


        render_style = render_style_dict[znzmo_2_suapp[style]]
        print(f"suapp生成图片，风格：{style},场景：{场景} , suapp的风格：{znzmo_2_suapp[style]}, suapp风格代码：{render_style}")

        try:
            print(f"开始生成SUAPP图片, 传入图片数量：{len(images)}")
            print(f"正向提示词：{p_prompt}")
            print(f"images_0.shape: {images[0].shape}")
            print(f"images.shape: {images.shape}")
        except Exception as e:
            print(f"打印suapp函数中出错：{e}")


        # ================== 2. 遍历每一张图片 ==================
        suapp_image_list = []
        for batch_number, image in tqdm(enumerate(images), desc="生成SUAPP图片"):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")
            src_width, src_height = img.size

            # 将PIL.Image 转为 base64 编码
            buffered = io.BytesIO()  # 创建一个字节流对象
            img.save(buffered, format="JPEG")  # 将图像保存到字节流中
            img_str = base64.b64encode(
                buffered.getvalue()
            )  # 编码字节流为 Base64 字符串
            image_base64 = img_str.decode(
                "utf-8"
            )  # 将 Base64 字节解码为字符串（如果需要在文本中使用）


            # ================== 3. 设置图片尺寸 ==================
            if src_width > src_height:  # 如果宽大于高
                # 如果宽度大于高度，高度设置为 512，宽度按比例缩放
                suapp_width = round(src_width / (src_height / 512), 0)  
                suapp_height = 512
            elif src_height == src_width:
                suapp_width = 512
                suapp_height = 512
            else:
                # 如果高度大于宽度，宽度设置为 512，高度按比例缩放
                suapp_width = 512
                suapp_height = round(src_height / (src_width / 512), 0)

            # ================== 4. 发送请求 ==================
            nn_image = "data:image/jpeg;base64,{}".format(image_base64)  # 图片的 base64 编码
            max_seed = int(1024 * 1024 * 1024)
            # Set the payload as provided in the curl request
            split_str = ","
            p_prompt = p_prompt.replace("，", ",")
            prompt = []
            if p_prompt != "":
                for word in p_prompt.split(split_str):
                    prompt.append({"show": False, "value": word, "weight": 1})
            
            payload = json.dumps(
                {
                    "prompt": prompt,
                    "neg_prompt": "",
                    "renderStyle": render_style,  # 场景-风格-代码
                    "ss_scale": 5,
                    "width": suapp_width,
                    "height": suapp_height,
                    "nn_image": "{}".format(nn_image),
                    "nn_weight": 1,
                    "outputImageNum": 1,
                    # 高清渲染，高清渲染不能使用多张
                    "hdRendering": True,
                    "hdRenderingType": "01",  # 风格类型
                    "nn_image_scribbled": False,
                    "scribble_accuracy": 3,
                    "seed": random.randint(1, max_seed),
                    "pageID": None,
                    "camera": None,
                    "cropInfo": None,
                    "taskType": "t2i",
                }
            )
            start_time = time.time()
            # print("开始时间:{}".format(start_time))
            # Send the request to the URL provided in the curl request
            task_response = requests.post(
                "https://ai.sketchupbar.com/ai/addTask",
                headers=headers,
                data=payload,
                verify=False,
            )
            print(f"suapp生成图片，第一次请求任务结果:{task_response.json()}")

            # Check if the request was successful
            if task_response.status_code == 200:
                # "{\"code\":200,\"si\":8,\"taskId\":\"1718368_1705930855007\",\"queue\":2,\"inputImagePaths\":{}}"
                # print(response.json())
                taskId = task_response.json()["taskId"]
                print(f"suapp发送任务成功，任务ID：{taskId}")
                # print('Request successful!')
                # 请求的 URL
                url = "https://ai.sketchupbar.com/ai/getTaskResult/{}?skipIMI=true&upscale=false&channel=false".format(
                    taskId
                )


                # ================== 5. 等待请求结果 ==================
                repeat = 0
                while True:
                    # 发送请求并获取响应
                    result_response = requests.get(url, headers=headers, verify=False)
                    print(f"请求结果:{result_response.json()}")
                    print(f"请求结果:{result_response.status_code}")
                    print(f"等待次数:{repeat}")
                    print("-"*20)
                    # print("响应内容：", result_response.json())
                    if result_response.json()["msg"] == "处理成功":
                        image = result_response.json()["image"]
                        end_time = time.time()
                        # print("开始时间:{}".format(end_time))
                        # {"code":200,"msg":"处理成功","image":"air_user_images/1718368/2024/01/23/1718368_1706010776542_out_1.jpg","moreImages":null}
                        suapp_ai_img_url = "https://ai.sketchupbar.com/{}".format(image)
                        # img_name = image.split('/')[-1]
                        print(
                            "https://ai.sketchupbar.com/{}, 出图时间：{}".format(
                                image, (end_time - start_time)
                            )
                        )
                        # 发送HTTP GET请求获取图片数据
                        response = requests.get(suapp_ai_img_url, verify=False)
                        # 检查请求是否成功
                        if response.status_code == 200:
                            # 得到PIL.Image对象
                            img = Image.open(io.BytesIO(response.content))
                            suapp_image_list.append(img)

                            break
                        
                        else:
                            print(f"下载suapp图片失败，HTTP状态码：{response.status_code}")
                            raise Exception(
                                "下载suapp图片失败，HTTP状态码：", response.status_code
                            )
                    
                    # 等待 1 秒后再次请求
                    time.sleep(3)
                    repeat += 1
                    if repeat > 50:
                        print("suapp生成图片超时，请求次数超过50次")
                        break

            else:
                print("Request failed with status code:", task_response.status_code)
                raise Exception("下载suapp图片失败，Request failed with status code:", task_response.status_code)
        
        # ================== 6. 返回图片 ==================
        # 将PIL.Image对象列表返回,转换为 comfyui 的 IMAGE 类型
        output_image, output_mask = get_comfyui_images(suapp_image_list)
        return (output_image, f"{场景}-{znzmo_2_suapp[style]}",)
    



class Maliooo_Get_ShengJingKeJi:
    """生境科技图片生成"""

    # 来源：\custom_nodes\comfy_mtb\nodes\image_processing.py

    def __init__(self):
        yaml_path = r"/data/ai_draw_data/suapp_config.yaml"
        config = omegaconf.OmegaConf.load(yaml_path)
        print("读取配置文件成功")
        print("shengjingkeji.Authorization: ", config.shengjingkeji.Authorization)
        print("-------------------")
        self.authentication = config.shengjingkeji.Authorization


        # 风格
        self.style = {
            "data": [
                {
                    "id": "33f41a1c-d154-4a69-bfcf-57c76b21965c",
                    "name": "欧式",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%8C%97%E6%AC%A7%E9%A3%8E.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%8C%97%E6%AC%A7%E9%A3%8E.png"
                },
                {
                    "id": "49e9f95a-a6a6-48d8-b239-036f0018dcc0",
                    "name": "现代",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E7%8E%B0%E4%BB%A3.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E7%8E%B0%E4%BB%A3.png"
                },
                {
                    "id": "d92f0c26-a803-43e7-9d21-8d1c09efe2ef",
                    "name": "新中式",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%96%B0%E4%B8%AD%E5%BC%8F.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%96%B0%E4%B8%AD%E5%BC%8F.png"
                },
                {
                    "id": "057cb9b9-d5ed-4e6c-85fd-6a234e8636a8",
                    "name": "极简",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%9E%81%E7%AE%80.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%9E%81%E7%AE%80.png"
                },
                {
                    "id": "b7d1d086-6300-4c37-b404-8ac5cba34f91",
                    "name": "日式",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%97%A5%E5%BC%8F.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E6%97%A5%E5%BC%8F.png"
                },
                {
                    "id": "e4fb787f-a582-467b-8c59-de22809dc5ec",
                    "name": "奶油风",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%A5%B6%E6%B2%B9%E9%A3%8E.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%A5%B6%E6%B2%B9%E9%A3%8E.png"
                },
                {
                    "id": "80843d18-6539-4d6b-804a-f83ac4d830a1",
                    "name": "侘寂",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%B7%A5%E4%B8%9A%E9%A3%8E.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%B7%A5%E4%B8%9A%E9%A3%8E.png"
                },
                {
                    "id": "adbc091d-cc27-4e2a-888e-06c8fb069e10",
                    "name": "北欧风",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%8C%97%E6%AC%A7%E9%A3%8E.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E5%8C%97%E6%AC%A7%E9%A3%8E.png"
                },
                {
                    "id": "ef90875d-79ae-4598-be49-c87a88d4195f",
                    "name": "轻奢",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E8%BD%BB%E5%A5%A2.png",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/prod/resources/styles/styles_20231211/%E8%BD%BB%E5%A5%A2.png"
                }
            ],
            "message": "success"
        }

        # 分类
        self.classify = {
            "data": [
                {
                    "id": "9e04bea0-9319-4903-8683-49901384b3ff",
                    "name": "客厅",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg"
                },
                {
                    "id": "0b67545e-4e07-4dd2-bf23-95db00b4e561",
                    "name": "卧室",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg"
                },
                {
                    "id": "093da416-7856-404a-88c3-eaad2c4ced67",
                    "name": "餐厅",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg"
                },
                {
                    "id": "58ed0849-3299-42af-a1e2-1278b9d572df",
                    "name": "厨房",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg"
                },
                {
                    "id": "49dc9601-28bd-4360-8f20-fcab667e22d0",
                    "name": "卫生间",
                    "image_small": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg",
                    "image_large": "https://sengine-cos-1259101928.cos.ap-guangzhou.myqcloud.com/sengine_web/dev/user_resources/test/38c75a4e-f720-4588-8d81-05bc5c448a9c.jpg"
                }
            ],
            "message": "success"
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "需要生境科技处理的底图"}),
                "classify": Field.combo(["客厅", "卧室", "餐厅", "厨房", "卫生间"]),
                "style": Field.combo(["现代", "欧式", "新中式", "极简", "日式", "奶油风", "侘寂", "北欧风", "轻奢"]),
                "positive_prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = (
        "IMAGE",
        "分类-风格",
    )
    FUNCTION = "get_shengjingkeji_image"
    CATEGORY = "🐼malio/image"

    def get_image_base64_by_path(self, image_file):
        with open(image_file, 'rb') as reference_file:
            image_content = reference_file.read()
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            return image_base64
    
    def get_image_base64_by_pil(self, image: Image):
        output_buffer = io.BytesIO()
        image = image.convert("RGB")
        image.save(output_buffer, format='JPEG')
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        return base64_str


    def upload_image(self, p_image_file, headers):
        """上传图片到生境科技"""

        conn = http.client.HTTPSConnection("api.s-engine.com.cn")
        if isinstance(p_image_file, Image.Image):
            image_base64 = self.get_image_base64_by_pil(p_image_file)
        elif isinstance(p_image_file, str):
            image_base64 = self.get_image_base64_by_path(p_image_file)
        else:
            raise ValueError("生境科技上传图片出错，p_image_file 参数类型错误, 必须是Image.Image对象或者图片路径")
        
        payload = {"business_type": "generation",
                "image_data_uri": "data:image/png;base64,{}".format(image_base64)}

        conn.request("POST", "/images/", json.dumps(payload), headers)

        res = conn.getresponse()
        data = res.read()

        result_json = json.loads(data.decode("utf-8"))
        # {'data': {'id': 'e4fa93a8-99d5-404c-aaf7-91468d6f65ab', 'url': 'https://sengine-cos-1259101928.cos.accelerate.myqcloud.com/sengine_web/prod/user_resources/usr_fN7V2N7ajC4UbMDvDCyg/e4fa93a8-99d5-404c-aaf7-91468d6f65ab.png', 'width': 1536, 'height': 1056}, 'message': 'success'}
        return result_json


    def tasks(self, p_task_body, headers):
        """提交生境科技生成图片任务"""
        conn = http.client.HTTPSConnection("api.s-engine.com.cn")
        conn.request("POST", "/tasks/", p_task_body, headers)
        res = conn.getresponse()
        data = res.read()

        result_json = json.loads(data.decode("utf-8"))
        return result_json


    def tasks_result(self, param, headers):
        """根据生境科技生成图片任务ID获取结果"""
        conn = http.client.HTTPSConnection("api.s-engine.com.cn")
        conn.request("GET", "/tasks/{}".format(param), "", headers)
        res = conn.getresponse()
        data = res.read()
        result_json = json.loads(data.decode("utf-8"))
        return result_json


    def get_ai_gen_image(self, pil_image:Image.Image, style_name, classify_name, batch_size=1, time_sleep=3, ):
        """获取生境科技生成的图片, 线稿生成图片，返回生成图片url列表
        pil_image: PIL.Image对象，原图
        style_name: str, 风格名字
        classify_name: str, 分类名字
        batch_size: int, 生成图片的数量, 1-4张图
        time_sleep: int, 间隔时间
        return: list, 生成的ai图片url列表
        """
                
        # 构造生境科技请求头
        sjkj_headers = {
        'Accept': "application/json, text/plain, */*",
        'Accept-Encoding': "gzip, deflate, br, zstd",
        'Accept-Language': "zh-CN,zh;q=0.9",
        'Access-Control-Allow-Origin': "*",
        'Authorization': f"Bearer {self.authentication}",
        'Connection': "keep-alive",
        'Content-Type': "application/json",
        'Host': "api.s-engine.com.cn",
        'Origin': "https://www.s-engine.com.cn",
        'Referer': "https://www.s-engine.com.cn/sengineplatform/platform2Dgenerate/generaterecord?function_name=redesign_unfurnishing",
        'Sec-Fetch-Dest': "empty",
        'Sec-Fetch-Mode': "cors",
        'Sec-Fetch-Site': "same-site",
        'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        'sec-ch-ua-mobile': "?0",
        'sec-ch-ua-platform': "Windows"
        }


        image_info = self.upload_image(p_image_file=pil_image, headers=sjkj_headers)  # 上传底图到生境科技

        image_id = image_info["data"]["id"]
        image_url = image_info["data"]["url"]
        width, height = pil_image.size  # 获取图片的宽高, 生成图片的分辨率

        style_name_list = [style_item["name"] for style_item in self.style["data"]]
        classify_name_list = [classify_item["name"] for classify_item in self.classify["data"]]

        # 获取风格id
        if style_name not in style_name_list:
            raise ValueError(f"风格名字错误，当前支持的风格有：{style_name_list}")
        else:
            style_id = self.style["data"][style_name_list.index(style_name)]["id"]

        # 获取分类id
        if classify_name not in classify_name_list:
            raise ValueError(f"分类名字错误，当前支持的分类有：{classify_name_list}")
        else:
            classify_id = self.classify["data"][classify_name_list.index(classify_name)]["id"]
        
        print
        print(f"风格id：{style_id}, 分类id：{classify_id}")

        task_body = {
            "type": "redesign_unfurnishing",
            "payload": {
                "image_id": image_id,
                "style_id": style_id,  # 风格id
                "room_id": classify_id,  # 分类id
                "user_mask_id": "",
                "decoration_id": style_name,  # 对应的风格名字
                "raw_image_id": image_id,
                "object_image_id": "",
                "object_width": 0,
                "object_height": 0,
                "left": 0,
                "top": 0,
                "reverse_x": False,
                "batch_number": 1,  # 生成图片的数量
                "whole_mask": False,
                "scene_description": "深色木制地板",  # 提示词
                "resolution": [   # 生成图片的分辨率，
                    width, # width, 生成的宽度
                    height, # height, 生成的高度
                ],
                "style_description": "",
                "need_main_cnts": True,
                "furniture_name": ""
            }
        }
        task_info = self.tasks(json.dumps(task_body), headers=sjkj_headers)  # 提交任务
        # {'data': {'id': '99ca074cd3474850adce725c8532f671', 'status': 'pending', 'type': 'redesign_unfurnishing'}}

        repeat_times = 0
        while True:
            image_result = self.tasks_result(task_info["data"]["id"], headers=sjkj_headers)  # 根据任务id获取结果
            if repeat_times % 10 == 0:
                print(image_result)
            if image_result["data"]["status"] == "success":
                print("任务完成")
                image_url_refer = image_result["data"]["result"]["images"][0]
                image_url_ai_list = image_result["data"]["result"]["images"][1:]
                print("-"*30)
                print(f"生境科技, 输入底图：{image_url}")
                print(f"生境科技，AI生成图：{image_url_ai_list}")
                return image_url_ai_list
            
            time.sleep(time_sleep)
            repeat_times += 1
            if repeat_times > 100:
                print(image_result)
                print(f"连接生境科技生成图片超时，连接时间为: {repeat_times*time_sleep}, 当前任务id为：{task_info['data']['id']}")
                return []



    def get_shengjingkeji_image(
        self,
        images,
        classify,
        style,
        positive_prompt,
    ):
        """返回生境科技生成的图片"""

        # print(f"开始进行suapp生成图片， 传入图片数量：{len(images)}， 场景：{场景}， Lora名称：{Lora名称}， 正向提示词：{正向提示词}")
        print("-"*20)
        print("-"*20)
        print(f"开始进行生境科技生成图片， 传入图片数量：{len(images)}， 场景：{classify}， 风格：{style}， 正向提示词：{positive_prompt}")

        image_list = tensor2pil(images)
        url_list = []
        for pil_image in image_list:
            ai_image_url_list = self.get_ai_gen_image(pil_image, style, classify)
            print(f"生境科技生成图片url列表：{ai_image_url_list}")
            url_list.extend(ai_image_url_list)
        

        if len(url_list) == 0:
            return (torch.zeros_like(images), f"{classify}-{style}")
        else:
            # 将url_list转为PIL.Image对象列表
            pil_image_list = [Image.open(io.BytesIO(requests.get(image_url).content)) for image_url in url_list]

            # ================== 6. 返回图片 ==================
            # 将PIL.Image对象列表返回,转换为 comfyui 的 IMAGE 类型
            output_image, output_mask = get_comfyui_images(pil_image_list)
            print(f"output_image:{output_image.shape}, output_mask:{output_mask.shape}")
            return (output_image, f"{classify}-{style}")
    

class Maliooo_Get_JianZhuXueZhang:
    """建筑学长图片生成"""

    def __init__(self):        
        yaml_path = r"/data/ai_draw_data/suapp_config.yaml"
        config = omegaconf.OmegaConf.load(yaml_path)
        print("------------------- 读取建筑学长配置文件：")
        print(config.jianzhuxuezhang.sign)
        print("-------------------")
        print(config.jianzhuxuezhang.token)
        self.sign = config.jianzhuxuezhang.sign
        self.token = config.jianzhuxuezhang.token
        
        # 建筑学长_室内设计.json 保存分类的lora、controlnet、默认负向提示词信息
        self.jzxz_indoor_design = None
        if os.path.exists(os.path.join(os.path.dirname(__file__), "json", "建筑学长_室内设计.json")):
            self.jzxz_indoor_design = json.load(open(os.path.join(os.path.dirname(__file__), "json", "建筑学长_室内设计.json"), "r"))
        else:
            print("建筑学长_室内设计.json 文件不存在")
        
        # znzmo : ['新中式' '现代' '极简' '奶油风' '轻奢' '法式' '侘寂风' '美式' '中式' '简欧' '北欧' '日式' '原木风']
        # jzxz: ['日式室内', '现代轻奢', '北欧奶油风', '侘寂风格', '豪华风格', '不限定', '线稿风格', '新中式风格', '梦幻风格', '现代风格', '现代时尚', '极简高级灰', '原木风格']
        self.znzmo_2_jzxz = {
            "新中式": "新中式风格",
            "现代": "现代风格",
            "极简": "极简高级灰",
            "奶油风": "北欧奶油风",
            "轻奢": "现代轻奢",
            "法式": "不限定",
            "侘寂风": "侘寂风格",
            "美式": "豪华风格",
            "中式": "新中式风格",
            "简欧": "不限定",
            "北欧": "不限定",
            "日式": "日式室内",
            "原木风": "原木风格"
        }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "需要建筑学长处理的底图"}),
                "classify": Field.combo(["客厅", "卧室", "餐厅", "厨房", "卫生间"]),
                "style": Field.combo(['新中式','现代', '极简', '奶油风', '轻奢', '法式', '侘寂风', '美式', '中式', '简欧', '北欧', '日式', '原木风']),
                "positive_prompt": ("STRING", {"default": ""}),

            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "IMAGE",
        "SEED",
        "jzxz_classify",
        "jzxz_style",
    )
    FUNCTION = "get_jianzhuxuezhang_image"
    CATEGORY = "🐼malio/image"

    def upload_image(self, image_file):
        """建筑学长图片上传接口"""
        conn = http.client.HTTPSConnection("proxy.jianzhuxuezhang.com")

        if isinstance(image_file, str):
            # Read the file content
            with open(image_file, 'rb') as file:
                file_content = file.read()
        elif isinstance(image_file, Image.Image):
            # Save the image to a byte stream
            byte_stream = io.BytesIO()
            image_file = image_file.convert('RGB')
            # 长边
            image_file.save(byte_stream, format='JPEG')
            file_content = byte_stream.getvalue()
            file_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        else:
            raise ValueError("建筑学长图片上传接口: image_file 参数必须是文件路径或 PIL.Image.Image 对象")

        conn = http.client.HTTPSConnection("proxy.jianzhuxuezhang.com")


        boundary = "---011000010111000001101001"
        # Path(image_file).name
        payload = (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="uploadPlatformType"\r\n\r\n'
                    "HUABAN\r\n"
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="extraParams"\r\n\r\n'
                    '{"cookieValue":"sid=s:rDglId34bMJgjNYwrkMU7u7QwPctZxla.36/xQ2gAGldG7d1qENOI+8Mgi6b8SxNnzfSMEumBAD4",'
                    '"bucket":"wander-image","key":"upload_1726344999991.1726344999991","url":"https://image.soutushenqi.com"}\r\n'
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="image"; filename="{file_name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode('utf-8') + file_content + f"\r\n--{boundary}--\r\n".encode('utf-8')
        content_length = len(payload)
        headers = {
            'accept': "application/json, text/plain, */*",
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua-mobile': "0",
            'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            'content-type': f"multipart/form-data; boundary={boundary}",
            'content-length': str(content_length),
            'origin': "https://www.jianzhuxuezhang.com",
            'sec-fetch-site': "same-site",
            'sec-fetch-mode': "cors",
            'sec-fetch-dest': "empty",
            'accept-encoding': "gzip, deflate, br, zstd",
            'accept-language': "zh-CN,zh;q=0.9",
            'priority': "u=1, i"
        }

        conn.request("POST", "/upload_image_file", payload, headers)
        res = conn.getresponse()
        data = res.read()

        return json.loads(data.decode("utf-8"))

    def post_task(
            self, 
            image:Image.Image, 
            image_url:str, 
            positive_prompt:str = "", 
            batch_size:int = 1,
            classify:str = "",
            style:str = ""
        ):
        """提交建筑学长的图片生成任务"""

        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NTc1NjI1OTYsInByaW1hcnlLZXkiOiJqenh6X3Rva2VuX01OSzduTHllIn0.Ix9DGDVx_LvuTH1Hm973UiiVDAwz111nFUufS7JeCdY"

        width, height = image.size
        print(f"原始的宽高为:{width}_{height}")
        # image 长边缩放到1000
        # image.thumbnail((1000,1000))
        width, height = image.size
        print(f"长边缩放到1000后的宽高为:{width}_{height}")
        
        post_url = "https://community-backend.soutushenqi.com/cykj_community/tools/image_generate"

        zjxz_headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
            'Sign': self.sign,
            'Token': self.token,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        }

        zjxz_payload = {
            "generateImageType": "TEXT2IMG",
            "generateImagePlatformType": "SELF_DEVELOPED",
            "ignoreTranslate": False,
            "productType": "JZXZ",
            "ignoreNsfw": True,
            "generateImageParams": {
                "prompt": positive_prompt,
                "modelName": "jianzhuw_jzxz",
                "negativePrompt": "NSFW, (worst quality:1.5), (low quality:1.5), (normal quality:1.5), lowres, blur, signature, drawing, sketch, text, word, logo, cropped, out of frame, wormline,(low quality),drawing,painting,crayon,sketch,graphite,soft,deformed,ugly,(soft line)",
                "schedulerName": "DPM++ 2M Karras",
                "guidanceScale": 7.5,
                "imageGeneratedNum": batch_size,
                "numInferenceSteps": 30,
                "loraParamses": [
                    {
                        "loraName": "XSArchi_124",
                        "weight": 0.3
                    }
                ],
                "controlNetInfoList": [
                    {
                        "model": "control_v11p_sd15_lineart [43d4be0d]",
                        "module": "lineart_standard",
                        "mode": 1,
                        "url": image_url, 
                        "controlGuidanceStart": 0,
                        "controlGuidanceEnd": 1,
                        "controlnetConditioningScale": 1.6
                    }
                ],
                "width": width,
                "height": height
            }
        }

        # 根据style 获取建 筑学长_室内设计.json 中的lora、controlnet、默认负向提示词信息
        if style in self.jzxz_indoor_design:
            zjxz_payload["generateImageParams"]["loraParamses"] = self.jzxz_indoor_design[style]["loraParamses"]
            zjxz_payload["generateImageParams"]["controlNetInfoList"] = self.jzxz_indoor_design[style]["controlNetInfoList"]
            zjxz_payload["generateImageParams"]["negativePrompt"] = self.jzxz_indoor_design[style]["negativePrompt"]
            
            # 赋值 image_url
            for controlNetInfo in zjxz_payload["generateImageParams"]["controlNetInfoList"]:
                controlNetInfo["url"] = image_url

        else:
            raise ValueError(f"建筑学长风格名字错误，当前支持的风格有：{list(self.jzxz_indoor_design.keys())}")
        
        print("-"*30)
        print(f"建筑学长提交任务参数：\n{zjxz_payload}")
        print("-"*30)

        response = requests.post(post_url, headers=zjxz_headers, json=zjxz_payload)
        print(response.text)
        return response

    def tasks_result(self, request_id):
        import requests

        base_url  = "https://community-backend.soutushenqi.com/cykj_community/tools/processing"
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NTc1NjI1OTYsInByaW1hcnlLZXkiOiJqenh6X3Rva2VuX01OSzduTHllIn0.Ix9DGDVx_LvuTH1Hm973UiiVDAwz111nFUufS7JeCdY"

        params = {
            'product_id': '51',
            'version_code': '2945',
            'requestId': request_id,
            'processingPlatformType': 'SELF_DEVELOPED',
            'processingType': 'TEXT2IMG',
            'ref': 'JZXZ_SELF_AI_DRAW',
            'productType': 'JZXZ',
            'sign': self.sign
        }

        headers = {
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Token': self.token,
            'User-Agent': 'PostmanRuntime-ApipostRuntime/1.1.0'
        }

        response = requests.get(base_url, headers=headers, params=params)
        # print(response.status_code)
        return response

    def get_ai_gen_image(self, image, classify:str = "", style:str = "", positive_prompt:str = ""):

        # response = upload_image(image_file=image)
        response = self.upload_image(image_file=image)
        print(f'建筑学长上传图片返回url：{response["imageUrl"]}')
        response = self.post_task(
            image=image, 
            image_url=response["imageUrl"], 
            positive_prompt=positive_prompt,
            classify=classify,
            style=style
        )

        if response.status_code == 200:
            request_id = response.json()["requestId"]
            repeat = 0
            error_count = 0
            while True and error_count < 3:
                res = self.tasks_result(request_id)
                if res.status_code == 200:
                    if res.json()["generateStatus"] == "SUCCEEDED":
                        image_download_url_list = res.json()["generatedImages"]
                        print(f"建筑学长, 生成ai图片数量为: {len(image_download_url_list)}")
                        return image_download_url_list
                else:
                    print("-"*30)
                    print("建筑学长返回： ", res.json())
                    print("-"*30)
                    error_count += 1
                    print(f"建筑学长，等待生成图片返回请求出错, 错误次数：{error_count}")
                    time.sleep(5)

                time.sleep(3)
                repeat += 1
                if repeat > 200:
                    print("建筑学长, 重试超过200次")
                    break
        else:
            print("建筑学长，上传图片失败")
        
        return []

    def get_jianzhuxuezhang_image(
        self,
        images,
        classify,
        style,
        positive_prompt,
    ):
        """返回建筑学长生成的图片"""

        print("-"*20)
        print("-"*20)
        print(f"开始进行建筑学长生成图片， 传入图片数量：{len(images)}， 场景：{classify}， 风格：{style}， 正向提示词：{positive_prompt}")
        
        
        if style not in self.znzmo_2_jzxz:
            zjxz_style = "不限定"
        else:
            zjxz_style = self.znzmo_2_jzxz[style]
        

        if zjxz_style not in self.jzxz_indoor_design:
            zjxz_style = "不限定"  # 默认风格

        image_list = tensor2pil(images)
        url_list = []
        for pil_image in image_list:
            ai_image_url_list = self.get_ai_gen_image(
                image=pil_image, 
                positive_prompt=positive_prompt, 
                style=zjxz_style,
                classify=classify
            )
            print(f"建筑学长生成图片url列表：{ai_image_url_list}")
            url_list.extend(ai_image_url_list)

        seed = 0
        if len(url_list) == 0:
            return (torch.zeros_like(images), seed, classify, zjxz_style)
        else:
            # 将url_list转为PIL.Image对象列表
            pil_image_list = [Image.open(io.BytesIO(requests.get(image_url).content)) for image_url in url_list]

            for pil_image in pil_image_list:
                try:
                    if pil_image.info:
                        print(f"建筑学长生成图片信息：{pil_image.info}")
                        seed = re.search(r"Seed: (\d+)", pil_image.info['parameters']).group(1)
                        seed = int(seed)
                except Exception as e:
                    seed = 0
                    print(f"建筑学长生成图片信息解析错误：{e}")
                    pass

            # ================== 6. 返回图片 ==================
            # 将PIL.Image对象列表返回,转换为 comfyui 的 IMAGE 类型
            output_image, output_mask = get_comfyui_images(pil_image_list)
            print(f"output_image:{output_image.shape}, output_mask:{output_mask.shape}")

            return (output_image, seed, classify, zjxz_style,)