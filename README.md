# RNN Multivariate Time Series Processing
Processing multivariate time series by RNN with Tensorflow

Three frameworks including:
1. Acceptor
2. Generalized Transducer (GT)
3. Encoder-Decoder

solving below problems:
1. forecasting (Acceptor, GT)
2. embeding (GT, Encoder-Decoder)

The concept of the frameworks were borrowed from Natural Language Processing. An Acceptor maps a sequence into a scalar (which may be a probability, for example). A transducer maps a pair of sequences into a scalar (which may be interpreted as a conditional probability of one sequence given another one). Classic transducer and acceptor could be regarded as two special cases of GT.
