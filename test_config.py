import json

# 打开 JSON 文件
with open('parameters.json', 'r') as f:
    # 使用 json.load() 函数加载 JSON 文件内容
    data = json.load(f)

# 现在，'data' 变量包含了 JSON 文件中的数据
# 你可以使用这些数据进行后续的处理
xx=data['elbows']

r1=[item['theta'] for item in xx]
r2=[item['phi'] for item in xx]
r2=[item if item<3.14 else item-3.14*2 for item in r2]
r1=sorted(r1)

r2=sorted(r2)
print(min(r1))
print(max(r1))

print(min(r2))
print(max(r2))
print(r2)
