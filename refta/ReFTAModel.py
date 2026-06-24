import torch.nn as nn
from .TensorTools.TensorSVD import *
from .utilized import *  
 
from .Replacing  import *
from typing import Optional
from tqdm import tqdm
 
def Loading(args, base_model, device):
 
        if args.model_name=="vit":
 
           multi_heads=base_model.vit.encoder.layer
        elif args.model_name=="roberta":
            multi_heads=base_model.roberta.encoder.layer 
 
        total = sum(args.FoDs_size[0] for name in  args.target_layers) 
        with tqdm(total=total, desc=f"Initializing target layers {args.target_layers} of {args.model_name}") as pbar:
          for qkv  in  args.target_layers:
            layernum=args.FoDs_size[0]
            qkvs=[]
            qkvs.append(qkv)
            refta_tensors=Obtain_Tensor(args, multi_heads, qkvs)
            ReFTA_layers=ReFTA_Combined(refta_tensors.AAs, refta_tensors.BBs,refta_tensors.U1, args.FoDs_size[0], lenU=args.lenU)
 
            for ii in range(layernum):
                layer=multi_heads[ii] 
                if refta_tensors.bias_tensor == None:
                    qkv_layer = modified_Linearlayer(refta_tensors.resW[ii,:,:], None, ii, qkv,  args.target_layers)
                else:
                    qkv_layer = modified_Linearlayer(refta_tensors.resW[ii,:,:], refta_tensors.bias_tensor[ii,:], ii, qkv,  args.target_layers)
                ResidualReFTALinear= ReFTALinear(ii, qkv_layer, ReFTA_layers) 
                HH=replacing_with_refta(args,multi_heads[ii], args.FoDs_size,qkv,ResidualReFTALinear,device)
                pbar.update(1)
        
        print("=========Finish the initialization====")

        # Clear the shared-projection cache at the start of every forward pass, so
        # the concatenation is rebuilt once per forward (not once per layer) while
        # staying correct under gradient accumulation.
        def _reset_refta_cache(module, args, kwargs=None):
            for m in module.modules():
                if isinstance(m, ReFTA_B1):
                    m.reset_cache()
        base_model.register_forward_pre_hook(_reset_refta_cache, with_kwargs=True)

        return  base_model
         



class Obtain_Tensor(nn.Module):
    def __init__(self,refta_args,    multi_heads, qkv_name,  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Obtain_Tensor, self).__init__()  
        self.device=device 
        self.qkv_name=qkv_name 
 
        self.FoDs_size=refta_args.FoDs_size
        if True:
            weightTensor=self.get_tensor_bias(multi_heads, refta_args.FoDs_size)
            self.weightTensor=torch.tensor(weightTensor).to(self.device) 
            self.bias_tensor=None
        else:
            weightTensor,bias_tensor=self.get_tensor(multi_heads, refta_args.FoDs_size)
            self.weightTensor=torch.tensor(weightTensor).to(self.device) 
            self.bias_tensor=torch.tensor(bias_tensor).to(self.device)
            
      
        transform_type = getattr(refta_args, 'transform_type', 'DCT')
        AAs, BBs, U1, U2, U3,   resW=get_topkAB_3D(self.weightTensor,component_number=refta_args.r,TransformType=transform_type)
        self.U1, self.U2, self.U3  =  U1.detach(), U2.detach(),  U3.detach()
        self.resW=resW.detach()
        self.AAs, self.BBs=AAs, BBs
    
    def get_tensor(self, multi_heads, FoDs_size):
        weights_tensor = np.zeros((FoDs_size[0], FoDs_size[1], FoDs_size[2]))
        bias_tensor = np.zeros((FoDs_size[0], FoDs_size[2])) 
        for i, layer in enumerate(multi_heads):
            for qkv in self.qkv_name:
                qkv_layer =  self.get_nested_attr_iter(layer, qkv)  
                if qkv_layer is not None:
                    weights_tensor[i, :, :] =getattr(qkv_layer, qkv).weight.detach().cpu().numpy() 
                    if getattr(qkv_layer, qkv).bias is not None:
                        bias_tensor[i, :] =getattr(qkv_layer, qkv).bias.detach().cpu().numpy()
                    else:
                        print('bias is None')
                else:
                    print(f"Layer {i} and its descendant does NOT have attribute {qkv}")
        return weights_tensor,bias_tensor


    def get_tensor_bias(self, multi_heads, FoDs_size):
        weights_tensor = np.zeros((FoDs_size[0], FoDs_size[1], FoDs_size[2]+1)) 
        for i, layer in enumerate(multi_heads):
            for qkv in self.qkv_name:
                qkv_layer =  self.get_nested_attr_iter(layer, qkv)  
                if qkv_layer is not None:
                    weights =getattr(qkv_layer, qkv).weight.detach().cpu().numpy() 
                    if getattr(qkv_layer, qkv).bias is not None:
                        bias =getattr(qkv_layer, qkv).bias.detach().cpu().numpy()
                    else:
                        print('bias is None')
                        bias = np.zeros((weights.shape[0],), dtype=weights.dtype)
                    weights_with_bias = np.concatenate([weights, bias[:, np.newaxis]], axis=1)
                    weights_tensor[i,:,:] =weights_with_bias
                    
                else:
                    print(f"Layer {i} and its descendant does NOT have attribute {qkv}")
        return weights_tensor 
        
    def get_nested_attr_iter(self,model, objname): 
        for name, module in model.named_children(): 
            if name==objname: 
                if self.FoDs_size[1]==module.weight.detach().cpu().numpy().shape[0]:
                   if self.FoDs_size[2]==module.weight.detach().cpu().numpy().shape[1]:
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

class ReFTA_B1(nn.Module):
    def __init__(self,AAs, BBs, layern,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ReFTA_B1, self).__init__()  
        self.device=device
        self.indList=[]
        self.layern=layern
        #self.layernorm_before=[]  
        for i in range(layern):
        #     self.layernorm_before.append(base_model.vit.encoder.layer[i].layernorm_before)
            
            if AAs[i] is not None:
                setattr(self,f'ReFTA_A{i}',nn.Parameter(AAs[i]))
                self.output_dim = AAs[i].shape[0]
                setattr(self,f'ReFTA_B{i}',nn.Parameter(BBs[i]))
                self.indList.append(i)
        self.register_buffer("ind_tensor", torch.tensor(self.indList, dtype=torch.long))
        # Per-slice ranks R_j (= columns of A_j) and their max, used to stack the
        # ragged per-layer adapters into a single padded batch for a fused einsum.
        self.rankList = [getattr(self, f'ReFTA_A{j}').shape[1] for j in self.indList]
        self.R_max = max(self.rankList) if self.rankList else 0
        # Column-to-slice map: slice_ids[c] = global slice index owning component c,
        # so that the per-layer mode-3 (U) mixing becomes a diagonal reweighting of
        # the R_tot shared components (no explicit O(K^2) slice mixing).
        slice_ids = []
        for j, r in zip(self.indList, self.rankList):
            slice_ids += [j] * r
        self.register_buffer("slice_ids", torch.tensor(slice_ids, dtype=torch.long))
        self._cat_cache = None   # (A_cat, B_cat) built once per forward; see reset_cache()

    def reset_cache(self):
        self._cat_cache = None

    def cat_factors(self, dtype=None, device=None):
        """Concatenate per-slice (A_j, B_j) into shared low-rank projections.

        Returns A_cat: (output_dim, R_tot)   B_cat: (R_tot, in_dim),
        where R_tot = sum_j R_j. These are shared across all K layers; the
        per-layer U-mixing is applied later as a diagonal weight (see Combined).
        The same adapter module is invoked once per target layer, so we build the
        concatenation only once per forward pass and reuse it across the K calls
        (the cache is cleared by a forward pre-hook registered in Loading, which
        keeps it correct under gradient accumulation). Built from the live
        nn.Parameters, so autograd / param identity is intact.
        """
        if self._cat_cache is None:
            A_cat = torch.cat([getattr(self, f'ReFTA_A{j}') for j in self.indList], dim=1)
            B_cat = torch.cat([getattr(self, f'ReFTA_B{j}') for j in self.indList], dim=0)
            if dtype is not None or device is not None:
                A_cat = A_cat.to(dtype=dtype, device=device)
                B_cat = B_cat.to(dtype=dtype, device=device)
            # split the bias-augmented column: B_weight (R_tot, d) + b_bias (R_tot,)
            B_weight = B_cat[:, :-1].contiguous()
            b_bias = B_cat[:, -1].contiguous()
            self._cat_cache = (A_cat, B_weight, b_bias)
        return self._cat_cache

    def stacked(self, dtype=None, device=None):
        """Stack the ragged per-slice (A_j, B_j) into zero-padded batched tensors.

        Returns
          A_stack: (G, output_dim, R_max)   B_stack: (G, R_max, in_dim)
        where G = len(indList). Padding columns are zero, so they contribute
        nothing to the downstream contraction (numerically exact vs. the loop).
        Stacking from the live nn.Parameters keeps autograd / param identity intact.
        """
        A_list, B_list = [], []
        for j in self.indList:
            A = getattr(self, f'ReFTA_A{j}')   # (output_dim, R_j)
            B = getattr(self, f'ReFTA_B{j}')   # (R_j, in_dim)
            pad = self.R_max - A.shape[1]
            if pad:
                A = nn.functional.pad(A, (0, pad))            # pad last dim (rank)
                B = nn.functional.pad(B, (0, 0, 0, pad))      # pad rank dim (rows)
            A_list.append(A)
            B_list.append(B)
        A_stack = torch.stack(A_list, dim=0)
        B_stack = torch.stack(B_list, dim=0)
        if dtype is not None or device is not None:
            A_stack = A_stack.to(dtype=dtype, device=device)
            B_stack = B_stack.to(dtype=dtype, device=device)
        return A_stack, B_stack

    def forward(self, x):
        # Kept for the rare full-U path: per-slice outputs (B, T, n, G) from the raw
        # (non-augmented) input; the bias column of B is split off and added.
        A_stack, B_stack = self.stacked(dtype=x.dtype, device=x.device)
        B_weight, b_bias = B_stack[..., :-1], B_stack[..., -1]          # (G,R_max,d), (G,R_max)
        P = torch.einsum('btf,grf->btgr', x, B_weight) + b_bias         # (B, T, G, R_max)
        y = torch.einsum('btgr,gnr->btng', P, A_stack)
        return y

class ReFTA_B2(nn.Module):
    def __init__(self,U,lenU,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ReFTA_B2, self).__init__()  
        self.device=device
        if lenU:
            self.ReFTA_U = nn.Parameter(U.to(device))  
        else:
            self.ReFTA_U = nn.Parameter(U.to(device), requires_grad=False) 
 
    def forward(self,x,ind_tensor, column_idx=None): 
        U=self.ReFTA_U#.to(x.device)
        if column_idx is not None:
            u_col = U[ind_tensor, column_idx].unsqueeze(-1)  # shape: (d_in, 1)
            y = torch.matmul(x, u_col)  # shape: (B, T, 1)
        else:
            y = torch.matmul(x, U)  # shape: (B, T, d_out)
        return y
  


class ReFTA_Combined(nn.Module):
    def __init__(self, AAs, BBs, U,  layern, lenU=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ReFTA_Combined, self).__init__()
        self.device = device
        self.refta_b1 =  ReFTA_B1(AAs, BBs,  layern) 
        self.refta_b2 = ReFTA_B2(U,lenU)

    def forward(self, newx, column_idx=None):
        # newx is the raw (non-augmented) layer input; the adapter's bias term is
        # folded in via b_bias, so no ones-column concatenation is needed.
        ind_tensor = self.refta_b1.ind_tensor
        if column_idx is None:
            # Full-U path (not used during training): keep the explicit form.
            b1_output = self.refta_b1(newx)
            return self.refta_b2(b1_output, ind_tensor, column_idx)

        # Fused hot path (shared-projection form): two rank-R_tot GEMMs per layer
        # with the mode-3 (U) mixing folded into a diagonal reweight w_k. Same
        # FLOP structure as a LoRA of rank R_tot, but A_cat/B_cat are shared across
        # all K layers. The O(K^2) slice mixing of the naive form disappears.
        #   out[b,t,n] = ( (x @ B_weight^T + b_bias) * w_k ) @ A_cat^T
        A_cat, B_weight, b_bias = self.refta_b1.cat_factors(dtype=newx.dtype, device=newx.device)
        w = self.refta_b2.ReFTA_U[self.refta_b1.slice_ids, column_idx].to(newx.dtype)  # (R_tot,)
        P = torch.matmul(newx, B_weight.t()) + b_bias   # (B, T, R_tot)
        out = torch.matmul(P * w, A_cat.t())            # (B, T, n)
        return out.unsqueeze(-1)                        # (B, T, n, 1)

class modified_Linearlayer(nn.Module):
    def __init__(self, W, bias, indx, layer_name,  target_layer,TrainB=True):
        super(modified_Linearlayer, self).__init__()
        W = W.float()
        self.indx=indx
        self.layer_name=layer_name
        self.target_layer=target_layer
        if bias is None:
            # W is the bias-augmented residual (n, d+1): split off the bias column
            # so the input no longer has to be concatenated with a ones column
            # (avoids materialising/saving the augmented (m, d+1) activation).
            self.weight = nn.Parameter(W[:, :-1].contiguous(), requires_grad=False)
            self.bias   = nn.Parameter(W[:, -1].contiguous(),  requires_grad=False)
        else:
            self.weight = nn.Parameter(W, requires_grad=False)
            if TrainB is None:
                self.bias = torch.Tensor(bias).float()
            else:
                self.bias = nn.Parameter(torch.Tensor(bias).float())

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)



class ReFTALinear(nn.Module):
    def __init__(self, indx, base_linear: nn.Linear, refta_layers: Optional[nn.Module] = None):
        super().__init__()
        self.base_linear = base_linear   
        self.refta_layers = refta_layers
        self.indx=indx
    def forward(self, hidden_states: torch.Tensor):
        # No bias augmentation: the residual bias is held separately in
        # base_linear, and the adapter folds its bias term in internally.
        output = self.base_linear(hidden_states)
        if self.refta_layers is not None:
           output += self.refta_layers(hidden_states, self.indx)[..., 0]
        return output
  
        
    
    
 
