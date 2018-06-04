from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from time import time as timer

from acceptor_model import Acceptor
from utils import DataGenerator

# global variables
X_DIM = 0
Y_DIM = 0
INPUT_LENGTH = 40  # input length to predict the future
EPOCH_MAX = 10000

# main function
if __name__ == '__main__':

	# reset TensorFlow computation graph
	tf.reset_default_graph()

	# TensorFlow configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	data_generator = DataGenerator(
		trainFileName=None, 
		featureColumnNames=None, 
		max_batch_size=1024, 
		splitBy='key',
		random=True, 
		totalData=None
	)

	# TODO: define acceptor structure
	rnn_structure = None

	# create acceptor
	acceptor = Acceptor(
        x_dim=X_DIM,
        y_dim=Y_DIM,
        rnn_structure=rnn_structure,
        scope='Acceptor',
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
        summaries_dir=None, 
	)

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		training_losses, validation_losses = acceptor_train(
		    sess=sess,
		    acceptor=acceptor, 
		    data_generator=data_generator,
		    num_epochs=EPOCH_MAX, 
		    rnn_structure=rnn_structure,
		    input_length=INPUT_LENGTH, 
		    y_dim=Y_DIM, 
		    x_dim=X_DIM,
		    valid_data_generator=None,
		    verbose=True,  
		    time_limit=float("Inf"),
		    stateful=True,
		)
