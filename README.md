# RNN Multivariate Time Series Processing
Processing multivariate time series by RNN with Tensorflow

Three frameworks including:
1. Acceptor
2. Generalized Transducer (GT)
3. Encoder-Decoder

solving below problems:
* forecasting (Acceptor, GT)
* embeding (GT, Encoder-Decoder)

The concept of the frameworks were borrowed from Natural Language Processing. 

RNN is unrolled as figure shown below. To train an RNN network, we firstly create the unrolled computation graph for a given input sequence, then define and add a loss node to the unrolled graph, and finally use BPTT to compute the gradients with respect to that loss. Different ways in which the supervision sequence is applied or the loss is defined lead to different RNN architectures.

<div align="center">
<img src="https://raw.githubusercontent.com/bujingyi/rnn-multivariate-time-series/master/image_markdown/rnn_unrolled.png" height="70%" width="70%" >
<p>RNN unrolled</p>
 </div>

### Acceptor
An Acceptor bases the supervision sequence only on the final output vector on which an outcome is decided. It maps a sequence into a scalar or a vector which may be interpreted as a conditional probability of one vector given one sequence. For example, an RNN is trained to read 40 points from a multivariate series and then use the final state to predict the vector of 41 points. The loss in such cases is defined as the divergence between the predict vector and the target vector. The loss can take various forms such as squared error, or even cosin similarity. 

<div align="center">
<img src="https://raw.githubusercontent.com/bujingyi/rnn-multivariate-time-series/master/image_markdown/acceptor.png" height="70%" width="70%" >
<p>Acceptor</p>
 </div>

### Transducer
A transducer products an output for each input point it reads in. So the loss is defined as the sum of local losses that is calculated from the divergence between the predict and the target of each point. A Transducer maps one sequence into another which may be interpreted as a conditional probability of one sequence given another one. For example, consider training an RNN to predect the next value of each point in a whole sequence.

<div align="center">
<img src="https://raw.githubusercontent.com/bujingyi/rnn-multivariate-time-series/master/image_markdown/transducer.png" height="70%" width="70%" >
<p>Transducer</p>
 </div>


### Generalized Transducer
Generalized Transducer is slightly different from Transducer. By adding two hyperparameter that controls which local loss is taken into the sum and the length of the input sequence respectively, the transducer is generalized. For Transducers, the first a few points are always difficult to predict because of the lack of prior information, therefore puting the first a few local losses into the loss may have a nagative influence on the predictions afterwards. If we tune the hyperparameter to only take the last local loss, Generalized Transducer then becomes an Acceptor. From this point of view, Acceptor and Transducer are two special cases of Generalized Transducer.

### Encoder
Encoder is very similar to Acceptor. However, unlike the Acceptor, where a prediction is made solely on the basis of the final state, here the final vector is treated as an encoding of the information in the sequence, and is usually used as additional information together with other signals. The loss is the same as that in Acceptor.

### Encoder-Decoder
Encoder-Decoder is composed of two RNNs: one acts as the role of the above Encoder which encodes the input sequence into a vector representation; another RNN uses the vector representation as auxiliary input to recover the original sequence. The first RNN is called Encoder and the second one is called Decoder. The loss is defined as the reconstruction loss which is the divergence between the reconstructed sequence and the original sequence. The supervision happens only for the Decoder, but the gradients are propagated all the way back to the Encoder. Encoder-Decoder is to vectorize an unstructured variable length multivariate time series while trying to keep the information as much as possible.

<div align="center">
<img src="https://raw.githubusercontent.com/bujingyi/rnn-multivariate-time-series/master/image_markdown/encoder_decoder.png" height="75%" width="75%" >
<p>Encoder-Decoder</p>
 </div>


#### Reference
[A Primer on Neural Network Models for Natural Language Processing](https://arxiv.org/abs/1510.00726)
