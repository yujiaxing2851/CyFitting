import json
import numpy as np
# 自定义编码器类
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.ndarray):
            return list(obj)  # 将 ndarray 转换为 Python 列表
        else:
            return super().default(obj)

json_cylinders=[{"test":1},{"test":2}]
json_elbows=[{"test":1},{"test":2}]

json_final=[]
json_final.append({"cylinders":json_cylinders})
json_final.append({"elbows":json_elbows})
json_final.append({"tees":{}})
json_final.append({"crosses":{}})

# 打印转换后的JSON字符串
json_string = json.dumps(json_final, cls=CustomEncoder)

print(json_string)
with open("test.json", 'a') as outfile:
    outfile.write(json_string)