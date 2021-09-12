# Parallel seq2seq in TensorFlow.
# For further details on parallel seq2seq see O. Ludwig, "End-to-end adversarial learning for generative conversational agents" in arXiv:1711.10122, Jan. 2018
# In case of publication using this code, please cite the paper above.                                                                                                                                         
# Author: Oswaldo Ludwig                                                                                                                                                              

from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras.api.keras import layers
import numpy as np                                   
import sys                                           
import time                                          
import pickle

# Here how to import the Par_seq2seq components:
from Par_seq2seq_beam_search import initialize
from Par_seq2seq_beam_search import operations                                      

np.random.seed(1234)


# Some parameters for this simple demo:
Num_epochs = 500                                                                  
Batch_size = 36                                                                    
Seq_length = 55
learning_rate = 1e-3
Dim_input = 46
learning_rate_decay = 0.997

# Here is how to instantiate a par_seq2seq model.
# The parameters are:
# number of LSTM layers to encode the input sequence: n_layers_EI
# number of LSTM layers to encode the incomplete output sequence: n_layers_EO
# dimension of the LSTM layers that encodes the input sequence: dim_EI
# dimension of the LSTM layers that encodes the incomplete output sequence: dim_EO
# dimension of the input vectors: dim_input
# dimension of the output vectors: dim_output
# max length of the input sequence: input_seq_len
ps2s = initialize(n_layers_EI=1, n_layers_EO=1, dim_EI=100, dim_EO=200, dim_input=Dim_input, dim_output=Dim_input, input_seq_len=Seq_length)

# This builds and saves a TF graph of the par_seq2seq:
ps2s.setup_ps2s()


chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.!?:'")
space_token = '-'                                         
pad_token = ' '                                           
BOS = '<'
EOS = '>'                                            
chars += [space_token,pad_token,BOS,EOS]                          
chars = list(sorted(set(chars)))
                          
# Some tools for this toy example on neural machine translation:

def Convert(lst): 
    res_dct = {lst[i]: i for i in range(len(lst))} 
    return res_dct 

Chars = Convert(chars)
print(Chars)

def indx2char(transcrip):
    dense_t = transcrip
    string = ''              
    for i in dense_t:        
       string += chars[int(i)]
    return string             
   
def char2indx(text, source):
    seq = []
    for i in text:
        seq.append(Chars[i])
    l = len(seq)
    if source == 'input':
        for j in range(Seq_length - l):
            seq = [Chars[' ']] + seq
    return seq

def indx2onehot(idxs):
    l = len(idxs)
    oh = np.zeros((l,Dim_input))
    for i, idx in enumerate(idxs):
        oh[i, idx] = 1
    return oh

def onehot2indx(onehot):
    indxs = []
    s = onehot.shape[0]
    for i in range(s):
        indxs.append(np.argmax(onehot[i,:]))
    return indxs

def toy_data(idx):
   return (indx2onehot(char2indx(text_eng[idx], 'input')), indx2onehot(char2indx(text_pt[idx], 'output')))

def create_batch():
    X = np.zeros((Batch_size,Seq_length,Dim_input))
    Y = np.zeros((Batch_size,Seq_length,Dim_input))
    for i in range(Batch_size):
        x, y =  toy_data(i)
        X[i,:,:] = x
        Y[i,0:y.shape[0],:] = y
    return (X,Y)    

# A small dataset:

text_eng = ['TELL-ME-NOT,-IN-MOURNFUL-NUMBERS',
'LIFE-IS-BUT-AN-EMPTY-DREAM!',
'FOR-THE-SOUL-IS-DEAD-THAT-SLUMBERS',
'AND-THINGS-ARE-NOT-WHAT-THEY-SEEM',
'LIFE-IS-REAL!-LIFE-IS-EARNEST!',
'AND-THE-GRAVE-IS-NOT-ITS-GOAL',
'DUST-THOU-ART,-TO-DUST-RETURNEST',
'WAS-NOT-SPOKEN-OF-THE-SOUL',
'NOT-ENJOYMENT,-AND-NOT-SORROW',
'IS-OUR-DESTINED-END-OR-WAY',
'BUT-TO-ACT,-THAT-EACH-TOMORROW',
'FIND-US-FARTHER-THAN-TODAY',
'ART-IS-LONG,-AND-TIME-IS-FLEETING',
'AND-OUR-HEARTS,-THOUGH-STOUT-AND-BRAVE',
'STILL,-LIKE-MUFFLED-DRUMS,-ARE-BEATING',
'FUNERAL-MARCHES-TO-THE-GRAVE',
'IN-THE-WORLD-IS-BROAD-FIELD-OF-BATTLE',
'IN-THE-BIVOUAC-OF-LIFE',
'BE-NOT-LIKE-DUMB,-DRIVEN-CATTLE!',
'BE-A-HERO-IN-THE-STRIFE!-A-PSALM-OF-LIFE',
'TRUST-NO-FUTURE,-HOWE-HER-PLEASANT!',
'LET-THE-DEAD-PAST-BURY-ITS-DEAD!',
'ACT,-ACT-IN-THE-LIVING-PRESENT!',
'HEART-WITHIN,-AND-GOD-O-ERHEAD!',
'LIVES-OF-GREAT-MEN-ALL-REMIND-US',
'WE-CAN-MAKE-OUR-LIVES-SUBLIME',
'AND,-DEPARTING,-LEAVE-BEHIND-US',
'FOOTPRINTS-ON-THE-SANDS-OF-TIME',
'FOOTPRINTS,-THAT-PERHAPS-ANOTHER',
'SAILING-LIFE-IS-SOLEMN-MAIN',
'A-FORLORN-AND-SHIPWRECKED-BROTHER',
'SEEING,-SHALL-TAKE-HEART-AGAIN',
'LET-US,-THEN,-BE-UP-AND-DOING',
'WITH-A-HEART-FOR-ANY-FATE',
'STILL-ACHIEVING,-STILL-PURSUING',
'LEARN-TO-LABOR-AND-TO-WAIT']

text_pt = ['<NAO-ME-DIGA,-EM-NUMEROS-TRISTES>',
'<A-VIDA-E-APENAS-UM-SONHO-VAZIO!>',
'<POIS-A-ALMA-ESTA-MORTA-QUE-DORME>',
'<E-AS-COISAS-NAO-SAO-O-QUE-PARECEM>',
'<A-VIDA-E-REAL!-A-VIDA-E-SINCERA!>',
'<E-O-TUMULO-NAO-E-SEU-OBJETIVO>',
'<TU-ES-PO,-AO-PO-RETORNAS>',
'<NAO-SE-FALOU-DA-ALMA>',
'<NAO-E-GOZO,-NEM-TRISTEZA>',
'<E-O-NOSSO-FIM-OU-CAMINHO-DESTINADO>',
'<MAS-PARA-AGIR,-QUE-CADA-AMANHA>',
'<ENCONTRE-NOS-MAIS-LONGE-DO-QUE-HOJE>',
'<A-ARTE-E-LONGA-E-O-TEMPO-E-PASSAGEIRO>',
'<E-NOSSOS-CORACOES,-EMBORA-ROBUSTOS-E-CORAJOSOS>',
'<AINDA-ASSIM,-COMO-TAMBORES-ABAFADOS,-ESTAO-BATENDO>',
'<O-FUNERAL-MARCHA-PARA-O-TUMULO>',
'<NO-AMPLO-CAMPO-DE-BATALHA-DO-MUNDO>',
'<NO-ACAMPAMENTO-DA-VIDA>',
'<NAO-SEJA-COMO-GADO-IDIOTA-E-DIRIGIDO!>',
'<SEJA-UM-HEROI-NA-LUTA!-A-PSALM-OF-LIFE>',
'<NAO-CONFIE-NO-FUTURO,-COMO-E-AGRADAVEL!>',
'<DEIXE-O-PASSADO-MORTO-ENTERRAR-SEUS-MORTOS!>',
'<AGIR,-AGIR-NO-PRESENTE-VIVO!>',
'<CORACAO-INTERIOR-E-DEUS!>',
'<VIDAS-DE-GRANDES-HOMENS-TODOS-NOS-LEMBRAM>',
'<PODEMOS-TORNAR-NOSSAS-VIDAS-SUBLIMES>',
'<E,-PARTINDO,-DEIXE-PARA-TRAS>',
'<PEGADAS-NAS-AREIAS-DO-TEMPO>',
'<PEGADAS,-QUE-TALVEZ-OUTRA>',
'<NAVEGANDO-PELA-PRINCIPAL-SOLENE-DA-VIDA>',
'<UM-IRMAO-ABANDONADO-E-NAUFRAGADO>',
'<VENDO,-TOMARA-CORACAO-NOVAMENTE>',
'<VAMOS,-ENTAO,-ESTAR-EM-ATIVIDADE>',
'<COM-UM-CORACAO-PARA-QUALQUER-DESTINO>',
'<AINDA-ALCANCANDO,-AINDA-PERSEGUINDO>',
'<APRENDA-A-TRABALHAR-E-ESPERAR>']

# This creates the data:
# The whole dataset composes a minibatch of training examples (X, Y), tensors of shape batch size x sequence length x input or output dimension (can be different)
X, Y = create_batch()

BOS_vector = np.zeros(Dim_input)
EOS_vector = np.zeros(Dim_input)
BOS_vector[17] = 1  #  The index of the begin of sentence (BOS) symbol is 17
EOS_vector[18] = 1  #  The index of the end of sentence (EOS) symbol is 18

# This starts a TF section to train and evaluate the model Par_seq2seq previously generated with ps2s.setup_ps2s():  
with tf.Session() as session:
        
        # This object has all the operation we need:
        op = operations(session)
        # This method imports the TF graph previously created by using ps2s.setup_ps2s(): 
        op.get_graph()
        
        tot_loss = 0
        mean_loss = 1e5                                                                 
        Count = 0
        
        # This loop trains the model:
        for curr_epoch in range(Num_epochs):  #  loop over training epochs                                                       
            
            learning_rate *= learning_rate_decay
                
            # This is how to perform a training step (update the model weights) using a batch of training examples
            # we only need to provide a batch of input and output sequences, i.e. X and Y and the learning rate
            batch_cost = op.train_ps2s(X, Y, learning_rate)
                
            tot_loss += np.mean(batch_cost)
            Count += 1
            mean_loss = tot_loss/Count
            print('Training epoch ' + str(curr_epoch) + ' LR = ' + str(learning_rate))
            print('mean loss = ' + str(mean_loss))

        # This method saves the trained model:
        op.save_ps2s()    

        # This loop evaluates the trained model:
        for i in range(Batch_size):  #  loop over batches of training data
                
            # getting training examples from the batch of examples:
            x = X[i:i+1,:,:]
            y = Y[i:i+1,:,:]
            decoded, probs = op.beam_decoder(x, BOS_vector, EOS_vector)
            P = probs.eval()[0]
            for k, seq in enumerate(decoded):
                txt = indx2char(seq)
                print('decoded: ' + txt.partition('>')[0] + '> , Prob: ' + str(P[k]))
