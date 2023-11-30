import torch

# 指定.pth文件的路径
pth_file_path = "attention_RPNReweight/latest.pth"

# 使用torch.load()函数加载.pth文件
model = torch.load(pth_file_path)
model_state_dict = model['state_dict']
backbone_state_dict = {}
# 查看state_dict中的键
print("Keys in the state_dict:")
for key in model_state_dict.keys():
    print(key)
for key, value in model_state_dict.items():
    if key.startswith("backbone."):
        backbone_state_dict[key] = value
for key in backbone_state_dict.keys():
    print(key)
state_dict = {}
state_dict['state_dict'] = backbone_state_dict
torch.save(state_dict, './backbone.pth')

# # 查看state_dict中的值
# print("\nValues in the state_dict:")
# for value in model_state_dict.values():
#     print(value)