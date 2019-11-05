# Parallel-Seq2Seq (PS2S)
PS2S is a new seq2seq model for fast training.
I proposed this model in July 2017 in the context of the [Seq2seq-Chatbot-for-Keras](https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras) project, see [[1]](https://arxiv.org/abs/1711.10122). In case of publication with this code, please cite this paper. As this model was successfully applied to generative chatbots, I’m now providing a general purpose TensorFlow-based PS2S toolbox, and to give you a taste of its performance, I’ve also provided a didactic toy example in which the user can train a PS2S-based neural translator in just 15 minutes of CPU processing.
The figure below illustrates the PS2S graph.

<p align="center">
  <img width="460" height="550" src="https://github.com/oswaldoludwig/Parallel-Seq2Seq/blob/master/graph.png">
</p>

When I proposed this model, I had two things in mind: avoiding the vanishing gradient and avoiding forgetting encoder information during decoding.
As seen in the figure above, in the PS2S the two LSTM blocks are arranged in parallel, while the canonical seq2seq (as well as the LAS model) has the recurrent layers of encoder and decoder arranged in series. Recurrent layers are unfolded during backpropagation through time, resulting in a large number of nested functions and thus a greater risk of vanishing gradient, which is worsened by the cascade of recurrent layers of the canonical seq2seq model, even for gated architectures, like the LSTM, as can be seen in the derivative of the LSTM cell state *s<sub>t</sub>* with respect to its immediate predecessor *s<sub>t-1</sub>* [2]:

![alt text](https://github.com/oswaldoludwig/Parallel-Seq2Seq/blob/master/eq1.png)

and thus, with *t*’>*t*:

![alt text](https://github.com/oswaldoludwig/Parallel-Seq2Seq/blob/master/eq2.png)

where *v<sub>t</sub>* is the input to the forget gate and σ is the sigmoid function ranging from 0 to 1. Therefore, due to σ, the above product can decay to zero exponentially.
Regarding forgetting encoder output during decoder steps, this occurs in the canonical seq2seq model because the encoder output is only used to initialize the decoder state. The LSTM state, *h*, is adjusted for each decoding step as follows [1]: 

![alt text](https://github.com/oswaldoludwig/Parallel-Seq2Seq/blob/master/eq3.png)

where **x** is the decoder input (the encoder output) *y<sub>i</sub>* is the decoder output at step *i*, *f<sub>α</sub>* represents the set of operations that the input and forget gates apply on the state variables, and *α* is the parameter set of these gates.
Note that the nested application of operations on **x**, such as the operation applied by the forget gate, can erase the contextual information along the decoder iterations. My architecture models p(**y**|**x**) like this:

![alt text](https://github.com/oswaldoludwig/Parallel-Seq2Seq/blob/master/eq4.png)

where *f<sub>β</sub>*(·) represents the LSTM that encodes the incomplete output sequence (*y*<sub>0</sub> . . . *y*<sub>i−1</sub>). Since the encoder output *g*(*x*) is provided to the decoder at each decoding iteration *i*, it is not subject to nested functions. This also can be solved using the attention mechanism [3], such as in the LAS architecture, but my model solves both problems at the same time for a very fast training.


**Use**

The example toy_example_neural_translation.py is fully commented. This generates the model, trains the model using a set of English-Portuguese sentences (within the code), and decodes using the same training material. Obviously, the small training dataset cannot provide generalization capacity to the model, the idea is to show how the data can be quickly fitted with WER=0.   

The Par_seq2seq.py toolbox is made up of two classes: initialize and operations. The first class receives the configuration of the model to instantiate an object. This class has the setup_ps2s method that creates and saves a TF graph of the PS2S model. The model parameters are:

1.	number of LSTM layers to encode the input sequence: n_layers_EI
2.	number of LSTM layers to encode the incomplete output sequence: n_layers_EO
3.	dimension of the LSTM layers that encodes the input sequence: dim_EI
4.	dimension of the LSTM layers that encodes the incomplete output sequence: dim_EO
5.	dimension of the input vectors: dim_input
6.	dimension of the output vectors: dim_output
7.	maximum length of the input sequence: input_seq_len

The second class, operations, must be instantiated in the TF section and has the methods:

1.	get_graph, which imports the TF graph previously created using the setup_ps2s method of class initialize.
2.	train_ps2s, which performs a training step (update the model weights) using a batch of training examples. This method takes as input a batch of input and output sequences and the learning rate.
3.	save_ps2s, which saves the model checkpoint.
4.	greedy_decoder, which takes the encoded input sequence, the vectors representing the beginning and end of the sequence and decodes it. No beam search decoder so far.


**References**


[1] Ludwig, O. (2017). [End-to-end adversarial learning for generative conversational agents](https://www.researchgate.net/publication/321347271_End-to-end_Adversarial_Learning_for_Generative_Conversational_Agents). arXiv preprint arXiv:1711.10122.

[2] Bayer, Justin Simon (2015). Learning Sequence Representations. Diss. München, Technische Universität München, Diss.

[3] Bahdanau, D., Cho, K., and Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.
