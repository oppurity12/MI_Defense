from classify import VGG16

import torch

model1 = VGG16(1000)
ckp_T = torch.load('target_model/target_ckp/VGG16_87.00_allclass.tar')
# ckp_T = torch.load('/home/m2017314029/Knowledge-Enriched-DMI/target_model/target_ckp/VGG16_86.87_allclass_pruned.tar')

dic = ckp_T['state_dict']

tmp = {}
for key, val in dic.items():
  tmp[key[7:]] = val

model1.load_state_dict(tmp, strict=True)

# print(ckp_T[])

# for m in model1.modules():
#   print(m)