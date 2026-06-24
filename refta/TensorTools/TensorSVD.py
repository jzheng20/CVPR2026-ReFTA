import torch
import numpy as np
from scipy.fftpack import dct

 
def matrixSVD(tensor): 
    tensor=torch.tensor(tensor, dtype=torch.float32)
    U, S, V = torch.svd(tensor)  
    return U, S, V.T

def apply_dft_transform(tensor, axis,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.fft.fft(tensor, dim=axis)

def apply_dct_transform(tensor, axis, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.tensor(dct(tensor.cpu().numpy(), axis=axis, norm='ortho'), dtype=torch.float32).to(device)
 


def apply_U_transform_3d(TT, U1, U2,U3,  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    TTsize=TT.shape
    
    unfolded_tensor_3 = U1.T@(TT.permute(1, 2, 0).reshape(-1, TT.shape[0])).T 
    TT3=unfolded_tensor_3.T.reshape(TTsize[1], TTsize[2],TTsize[0])
    TT3=TT3.permute(2, 0, 1)
  
    GG=TT3

    return GG

     

 

def HOSVD_3D(TT):
    TTsize=TT.shape
    # Mode 4 Unfolding 
    unfolded_tensor_1 = TT.reshape(-1, TT.shape[-1]) 
    U3, S1, V1T=matrixSVD(unfolded_tensor_1.T)
    unfolded_tensor_1 = torch.diag(S1)@V1T 
    TT1=unfolded_tensor_1.T.reshape(TTsize[0], TTsize[1], TTsize[2]) 

    # Mode 3 Unfolding 
    unfolded_tensor_2 = TT1.permute(2, 0, 1).reshape(-1, TT1.shape[1]) 
    U2, S2, V2T=matrixSVD(unfolded_tensor_2.T)
    unfolded_tensor_2 = torch.diag(S2)@V2T 
    TT2=unfolded_tensor_2.T.reshape(TTsize[2], TTsize[0], TTsize[1])
    TT2=TT2.permute(1, 2, 0)

    # Mode 2 Unfolding 
    unfolded_tensor_3 = TT2.permute(1, 2, 0).reshape(-1, TT2.shape[0]) 
    U1, S3, V3T=matrixSVD(unfolded_tensor_3.T)
    unfolded_tensor_3 = torch.diag(S3)@V3T 
    TT3=unfolded_tensor_3.T.reshape(TTsize[1], TTsize[2],TTsize[0])
    TT3=TT3.permute(2, 0, 1)
 
    
    GG=TT3

    return GG, U1, U2, U3 


def TSVD_3D(TT):
    TTsize=TT.shape
    # Mode 4 Unfolding 
    unfolded_tensor_1 = TT.reshape(-1, TT.shape[0]) 
    U1, S1, V1T=matrixSVD(unfolded_tensor_1.T)
    unfolded_tensor_1 = torch.diag(S1)@V1T 
    TT1=unfolded_tensor_1.T.reshape(TTsize[0], TTsize[1], TTsize[2]) 

 
    GG=TT1

    return GG, U1 


def tmprod(A,Q,k): 
    Tensor=A.clone()
    reshaped_tensor,Tsize = mFolding(Tensor,k)
    newTensor=mUnfolding(torch.matmul(Q,reshaped_tensor),k,Tsize)
    return newTensor

def mFolding(A,k):
    Asize=A.size()
    h=A.ndimension()
    dims=tuple([x + k for x in range(h-k)])+tuple(range(k))
    Tensor=A.permute(dims)
    reshaped_tensor = Tensor.reshape(Asize[k], -1)
    return torch.tensor(reshaped_tensor,dtype=torch.float32), Tensor.size()

def mUnfolding(matrixA,k,nAsize): 
    h=len(nAsize)
    invdims=tuple([x + h-k for x in range(k)])+tuple(range(h-k))
    TensorA=matrixA.reshape(nAsize)
    return TensorA.permute(invdims)

 


def TensorDecomposition_3D(TT,TransformType,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    TTsize=TT.shape
    if TransformType=='DFT':
        U1=apply_dft_transform(torch.eye(TTsize[0]), 1).to(device)
        U2=torch.eye(TTsize[1]).to(device)
        U3=torch.eye(TTsize[2]).to(device)
        TT1=apply_dft_transform(TT, 0).to(device)
 
        GG=TT1        
    elif TransformType=='DCT':
        #print(TTsize[0])
        U1=apply_dct_transform(torch.eye(TTsize[0]), 1) 
        U2=torch.eye(TTsize[1]).to(device)  
        U3=torch.eye(TTsize[2]).to(device) 
        GG=apply_U_transform_3d(TT, U1, U2,U3)         
    elif TransformType=='HOSVD':
        GG, U1, U2, U3 =HOSVD_3D(TT)
    elif TransformType=='LSM-3':
        GG, U1 =TSVD_3D(TT)
        U2, U3 = torch.eye(TTsize[1],device=device),torch.eye(TTsize[2],device=device)
    else:
        GG=TT
        U1, U2, U3= torch.eye(TTsize[0],device=device),torch.eye(TTsize[1],device=device),torch.eye(TTsize[2],device=device) 
    return GG, U1, U2, U3  

 


def TensorSVD_3D(TT,TransformType='None'):
    TTsize=TT.shape
    GG, U1, U2, U3=TensorDecomposition_3D(TT,TransformType)
    TTsize12_min=torch.min(torch.tensor(TTsize[1]),torch.tensor(TTsize[2])) 
    SS= torch.zeros((TTsize[0],TTsize12_min))
    UU= torch.zeros((TTsize[0],TTsize[1],TTsize12_min))
    VVT= torch.zeros((TTsize[0],TTsize12_min,TTsize[2]))
    t=0
    for ii in range(TTsize[0]):
        UU[ii,:,:], SS[ii,:], VVT[ii,:,:]=matrixSVD(GG[ii,:,:])
        t=t+1
    return UU, SS, VVT, GG, U1, U2, U3 

 

  

 

def get_topkAB_3D(TT,component_number=1,TransformType='None',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    TT=torch.tensor(TT,dtype=torch.float32)
    TTsize=TT.shape
 
    UU, SS, VVT, GG, U1, U2, U3 = TensorSVD_3D(TT,TransformType=TransformType)
    flattened_SS = SS.flatten()
    topk_values, topk_indices = torch.topk(flattened_SS, component_number) 
    topk_values_last=topk_values[-1]
 
    AAs = []
    BBs = []
    GGs=torch.zeros(TTsize, device=device) 
    for ii in range(TTsize[0]):
        #num_ii = ((topk_indices_2d[:, 0] == ii))).sum().item() 
        num_ii = torch.sum(SS[ii,:] >= topk_values_last).item()
        if num_ii==0:
            AAs.append(None) 
            BBs.append(None)
        else:
            AA=UU[ii,:,:num_ii]
            AAs.append(AA)
            BB=torch.diag(SS[ii,:num_ii])@VVT[ii,:num_ii,:]
            BBs.append(BB)
            GGs[ii,:,:]=AA@BB  
    W_main= tmprod(GGs,U1,0) 
    W_main= tmprod(W_main,U2,1) 
    W_main= tmprod(W_main,U3,2)    
    resW=TT-W_main 
    return AAs, BBs, U1, U2, U3,  resW
 

