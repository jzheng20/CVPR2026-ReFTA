 

import os
import time
import torch
import numpy as np 


 
import torch.nn as nn
from .TensorTools.TensorSVD import *
from .utilized import * 
 

class replacing_with_refta(nn.Module):
    def __init__(self,args,   multi_heads, FoDs_size,qkv_name, ResidualLoRALinear,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(replacing_with_refta, self).__init__()

 
        self.device = device
        self.ResidualLoRALinear=ResidualLoRALinear
        self.layer=multi_heads
        self.FoDs_size=FoDs_size

        qkv=qkv_name
        qkv_layer = self.get_nested_attr_iter(self.layer, qkv)
        if qkv_layer is not None:
            self.replace_qkv(qkv_layer,qkv,ResidualLoRALinear)     
        else:
            print(f"Layers does NOT have attribute {qkv}")

    

    def replace_qkv(self, module,name, ResidualLoRALinear):
        child=getattr(module, name)
        setattr(module, name, ResidualLoRALinear) 
         


    def get_nested_attr_iter(self,model, objname): 
        for name, module in model.named_children(): 
            if name==objname:
                if self.FoDs_size[1]==module.weight.detach().cpu().to(torch.float32).numpy().shape[0]:
                   if self.FoDs_size[2]==module.weight.detach().cpu().to(torch.float32).numpy().shape[1]:
                     return model
                   else: 
                       return None 
                else: 
                    return None 
            else:
                result=self.get_nested_attr_iter(module,objname)
                if result is not None:
                    return result 
        return None



     

      
             