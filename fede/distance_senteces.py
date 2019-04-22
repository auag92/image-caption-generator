import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Distance_Sentences(nn.Module):
    def __init__(self, M,N,O):
        '''
        Arguments:
            M: the size of the embbedded vector for the captions
            N: Size of distance kernel
            O: Number of distance kernels to use
        '''
        super(Distance_Sentences, self).__init__() 
        
        self.M=M;
        self.N=N;
        self.O=O;
        self.T_senteces=nn.Linear(M,N*O, bias=False);


    def forward(self, x):
    
        nsets,set_size,M=x.shape;        
        M=(self.T_senteces(x)).view(nsets,set_size,self.N,self.O);
        o=torch.zeros(nsets, self.O*set_size);
        
        for j in range(0,nsets):
            temp=torch.zeros(set_size, self.N);
            for i in range(0,set_size):            
                temp[i,]=torch.sum(torch.exp(- torch.sum(torch.abs(M[j,i,:,:]-M[j,:,:,:]),dim=1 ) ),dim=0 ) 
                
            
            
            o[j,:]=temp.view(1,-1);
        return o

