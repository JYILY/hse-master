import torch
from models.ResNetEmbed import ResNetEmbed


classes_dict = {'order': 13, 'family': 37, 'genus': 122, 'class': 200}
# model = ResNetEmbed(cdict=classes_dict)

state_dict = torch.load("E:\\CUB_200_2011\\model_cub_hse.tar")

print(state_dict.keys())