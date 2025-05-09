from rich import print
import torch

class Malio_BBOXES:
    """使用python的eval函数生成bboxes"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input":("STRING",),
            }
        }

    RETURN_TYPES = ("BBOXES",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "forward"

    def forward(self, input):
        """使用python的eval函数生成bboxes"""
        print("你好")
        print(f"使用eval函数，获得输入：{input}")
        output = None
        try:
            output = eval(input)
            output = torch.tensor(output)
            while output.dim() < 3:
                # 如果output的维度小于3，则将其转换为3维
                output = output.unsqueeze(0)
            print(f"使用eval函数，输入：{input} 生成数据: {output}, type-output:{type(output)}")
        except Exception as e:
            print(f"出错了，eval_function error: {e}")
            output = None
        return (output,)
    

if __name__ == "__main__":
    eval_node = Malio_BBOXES()
    s = "[[1,2,3,4],[5,6,7,8]]"
    res = eval_node.forward(s)
    print(res)
    # print(list(res))
    # print(res)
    # print(type(res))

    # res = eval_node.eval_function("[[0,10, 1100, 200]]")
    # print(res[0])
    # print(type(res))
