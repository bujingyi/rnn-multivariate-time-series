# RNN Multivariate Time Series Processing
Processing multivariate time series by RNN with Tensorflow

Three frameworks including:
1. Acceptor
2. Generalized Transducer (GT)
3. Encoder-Decoder

solving below problems:
* forecasting (Acceptor, GT)
* embeding (GT, Encoder-Decoder)

The concept of the frameworks were borrowed from Natural Language Processing [A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726). 

RNN is unrolled as figure shown below. To train an RNN network, we firstly create the unrolled computation graph for a given input sequence, then define and add a loss node to the unrolled graph, and finally use BPTT to compute the gradients with respect to that loss. Different ways in which the supervision signal is applied or the loss is defined lead to different RNN architectures.

### Acceptor
An Acceptor maps a sequence into a scalar (which may be a probability, for example). 


### Transducer
A transducer maps a pair of sequences into a scalar (which may be interpreted as a conditional probability of one sequence given another one). 

### Generalized Transducer
Classic transducer and acceptor could be regarded as two special cases of GT.
