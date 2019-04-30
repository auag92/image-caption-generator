from builtins import range
from builtins import object
import numpy as np

import torch
import torch.nn as nn

from cs231n.layers import *
from cs231n.rnn_layers import *
from torch.nn.utils.rnn import pack_padded_sequence

from my_rnn_Fernando import MY_CaptioningRNN
from distance_senteces import Distance_Sentences
from distance_image import Distance_Image


class Discriminator(nn.Module):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_Dim=1, wordvec_Dim=128,
                 hidden_Dim=128, N=128, O=128, image_input=512,set_size=10  ):
        
        super(Discriminator, self).__init__()
        
        self.sentence_embedding=MY_CaptioningRNN(word_to_idx, input_dim=input_Dim, wordvec_dim=wordvec_Dim,
                 hidden_dim=hidden_Dim, cell_type='rnn', dtype=np.float32);
                 
        vocab_size=len(word_to_idx);
        self.distance_layer_sentences=Distance_Sentences(vocab_size,N,O);
        self.distance_layer_images=Distance_Image(vocab_size,N,O,image_input);
        
        
        self.set_size=set_size
        self.projection=nn.Linear((self.set_size+1)*O,2);
        return 
        
    def forward(self, captions, features):
    
        
        nsamples,trash=captions.shape
        print(captions.shape)
        S=self.sentence_embedding.forward( torch.zeros(nsamples,1), captions);
        print(S.shape)
        S=S[:,-1,:];
        print(S.shape)
        S=S.view(S.shape[0]//self.set_size,self.set_size,-1)
        print(S.shape)
        o_sentence=self.distance_layer_sentences.forward(S);
        print(o_sentence.shape)
        o_image=self.distance_layer_images.forward(S,features);
        print(o_image.shape)
        o=torch.cat((o_image, o_sentence), 1);
        print(o.shape)
        D=nn.functional.log_softmax(self.projection(o),dim=1);
        print(D.shape)
        return D
        
