import tensorflow as tf

from encoder_decoder.ed_experiment_fn import run_experiment


# main function
if __name__ == '__main__':
	tf.reset_default_graph()
	params = tf.contrib.training.HParams(
		train_data_file=None,  # specify training data file
		test_data_file=None,  # specify test data file
		model_dir=None,
		is_shuffle=True,
		epoch=100,
		batch_size=256,
		train_steps=1,
		is_padding=True,
		pad_shape=(tf.TensorShape([None, None]), tf.TensorShape([None, None])),
		buffer_size=10000,
		pad_value=0,
		min_eval_frequency=100,
		rnn_structure=None,  # TODO: a list of tuples to define rnn structure
		learning_rate=1e-4,
		num_feature=0,
		task_type=None  # TODO: alter the model according to multiple task types
	)

	# run experiment
	run_experiment(params)