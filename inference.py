from model import network
import parser

import torch

from pathlib import Path

args = parser.parse_arguments()

args.backbone = "cct384"
args.aggregation = "netvlad"
args.resize = [384, 384]
args.backbone_pretrained = False
pretrained_weights = "../msls_cct384tr8fz1_netvlad.pth"



args.backbone = "resnet101conv4"
args.aggregation = "netvlad"
args.resize = [384, 384]
args.backbone_pretrained = False
pretrained_weights = "../msls_r101l3_netvlad_partial.pth"


args.backbone = "vit"
args.aggregation = "cls"
args.resize = [224, 224]
args.backbone_pretrained = False
pretrained_weights = "../msls_vit_tr10_cls.pth"


checkpoint = torch.load(pretrained_weights, weights_only=False)
if 'model_state_dict' in checkpoint:
    model_state_dict = checkpoint['model_state_dict']
else:
    model_state_dict = checkpoint

with Path("saved_weights.txt").open("w") as f:
    names = model_state_dict.keys()
    for name in names:
        f.write(name + "\n")

model = network.GeoLocalizationNet(args)



with Path("saved_weights.txt").open("a") as f:
    names = model.state_dict().keys()
    for name in names:
        f.write(name + "\n")

# remove the 'module.' prefix from the keys
model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
# check which keys are missing
missing_keys_in_model = [k for k in model.state_dict().keys() if k not in model_state_dict]
missing_keys_in_saved = [k for k in model_state_dict.keys() if k not in model.state_dict()]

list(zip(missing_keys_in_model, missing_keys_in_saved))

model.load_state_dict(model_state_dict, strict=True)


