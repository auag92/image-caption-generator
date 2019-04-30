from builtins import range
from builtins import object
import numpy as np

import torch
import torch.nn as nn

from cs231n.layers import *
from cs231n.rnn_layers import *
from torch.nn.utils.rnn import pack_padded_sequence



class MY_CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        super(MY_CaptioningRNN, self).__init__()


        self.use_cuda=False;
        if torch.cuda.is_available():
            self.use_cuda=False;
        

        vocab_size = len(word_to_idx)



        # Initialize word vectors      
        self.W_embed=nn.Embedding(vocab_size, wordvec_dim);
        
        self.vocab_size=vocab_size
        self.wordvec_dim=wordvec_dim
  
        self.W_proj=nn.Linear(input_dim,hidden_dim);

        self.num_layers=10
        self.rnn=nn.LSTM(input_size  = wordvec_dim, hidden_size =hidden_dim,
        num_layers =self.num_layers,batch_first=True,dropout=0.5 );

        # Initialize output to vocab weights
        self.W_vocab=nn.Linear(hidden_dim,vocab_size);
        
        
        self.criterion=nn.CrossEntropyLoss();


        # Cast parameters to correct dtype
        #for k, v in self.params.items():
        #    self.params[k] = v.astype(self.dtype)
        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)
        
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        
        
        
        
        return

    def forward(self, features, captions):
    
        
        hidden_image=self.W_proj(features);
        
        captions=captions.long()
        word_embedding=self.W_embed(captions);        
        
        
        hidden_image=hidden_image.unsqueeze(0)
        hidden_image=hidden_image.repeat(self.num_layers,1,1)
        
        c0=torch.zeros(hidden_image.shape)
        if(self.use_cuda):
            c0=c0.cuda()
        hiddens, _ = self.rnn(word_embedding,(hidden_image,c0))
        
        
        pred=self.W_vocab(hiddens)
        
        return pred
        
    def masked_cross_entropy(self, pred, captions_out, mask ):
        
        
        N,T,V=pred.shape
        
        x_flat = pred.view(N*T,V)
        y_flat = (captions_out.long()).view(N*T)
        mask_flat = (mask.float() ).view(N*T)

        value,_=torch.max(x_flat, dim=1, keepdim=True)
        probs = torch.exp(x_flat - value)
        probs = probs/torch.sum(probs, dim=1, keepdim=True)

        loss = -torch.sum(mask_flat *torch.log(probs[np.arange(N * T), y_flat])) / N
        return loss   
        
    def loss_2(self, features, captions, max_length=30):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        loss, grads = 0.0, {}
        
        N = features.shape[0]
        
        hidden_image=self.W_proj(features);
        
      
   

        _,max_length=captions_in.shape
        
        captions_gen = self._null * torch.ones((N), dtype=torch.long)
    
        next_h=hidden_image.unsqueeze(0)
        next_h=next_h.repeat(self.num_layers,1,1)
      
        c0=torch.zeros(next_h.shape)
        
        gen=self._null * torch.ones((N, max_length,self.vocab_size))
        if(self.use_cuda):
            c0=c0.cuda()
            captions_gen=captions_gen.cuda();
            gen=gen.cuda();
        
        
        
        for t in range(max_length-1):
            #word = torch.zeros((N,1), dtype=torch.long)
            #word[ captions[:, t] ]=1;  
            word = (self.W_embed(captions_gen))
            word=word.unsqueeze(1)
            
            
            _, (next_h,c0) = self.rnn(word,(next_h,c0))
            
            #next_h,c0=out
            #next_h=(next_h[:,0,:]).unsqueeze(0)
            
            pred=self.W_vocab(next_h[-1,:,:])
            gen[:,t,:]=pred
            #print(next_h.shape)        
            sol=torch.max( pred,dim=1 )
            
            _, captions_gen = sol
        
        pass
        
        
        
        loss=self.masked_cross_entropy( gen, captions_out, mask )
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
   
    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = (captions_out != self._null)

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        ############################################################################
        

        hidden_image=self.W_proj(features);
        
        captions=captions_in.long()
        word_embedding=self.W_embed(captions);        
   
        hidden_image=hidden_image.unsqueeze(0)
        hidden_image=hidden_image.repeat(self.num_layers,1,1)
        
        c0=torch.zeros(hidden_image.shape)
        if(self.use_cuda):
            c0=c0.cuda()
        
        hiddens, out= self.rnn(word_embedding,(hidden_image,c0))
        
        next_h,c0=out
       
        pred=self.W_vocab(hiddens)
            
        loss=self.masked_cross_entropy( pred, captions_out, mask )
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     to the appropriate slot in the captions variable                    #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        ###########################################################################
        features  = torch.from_numpy(features)

        V=self.wordvec_dim
        
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)  
        captions[:, 0] = self._start
        
        if(self.use_cuda):
            captions=captions.cuda();
            features=features.cuda()
            
        hidden_image=self.W_proj(features);     
         
        next_h=hidden_image.unsqueeze(0)
        next_h=next_h.repeat(self.num_layers,1,1)
        
        
        c0=torch.zeros(next_h.shape)
        
        if(self.use_cuda):
            c0=c0.cuda()
            
        for t in range(max_length-1): 
            word = (self.W_embed(captions[:, t]))
            word=word.unsqueeze(1)
            
            
            pred, out = self.rnn(word,(next_h,c0))
            
            next_h,c0=out
            pred=self.W_vocab(next_h[-1,:,:])     
            sol=torch.max( pred,dim=1 )
            
            _, captions[:,t + 1] = sol
        captions=captions.cpu()
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions.data.numpy()
