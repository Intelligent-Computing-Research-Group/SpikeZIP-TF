import torch
import os

if __name__ == '__main__':
    a = torch.load("Roberta-sst2-ANN/best.pth.tar")["model"]
    new_dict = {}
    print(a.keys())
    for k, v in a.items():
        if "bert." in k:
            new_dict[k.replace("bert.", "")] = v
    torch.save(new_dict, os.path.join("Roberta-sst2-ANN", "pytorch_model.bin"))