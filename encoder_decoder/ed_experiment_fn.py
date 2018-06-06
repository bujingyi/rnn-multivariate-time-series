import tensorflow as tf

from encoder_decoder.ed_input_fn import input_fn
from encoder_decoder.ed_model_fn import model_fn


def experiment_fn(run_config, params):
    """
    Create an experiment to train and evaluate the model.
    """
    run_config = run_config.replace(save_checkpoints_steps=params.min_eval_frequency)
    estimator = get_estimator(run_config, params)
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=lambda: input_fn(
            data_dir_list=params.train_data_file_list, 
            num_x_feature=params.num_x_feature,
            num_y_feature=params.num_y_feature, 
            is_training=True, 
            is_predicting=False,
            is_shuffle=True, 
            is_padding=params.is_padding,
            buffer_size=params.buffer_size, 
            pad_shape=params.pad_shape,
            batch_size=params.batch_size, 
            epoch=params.epoch
        ),
        eval_input_fn=lambda: input_fn(
            data_dir_list=params.test_data_file_list, 
            num_x_feature=params.num_x_feature,
            num_y_feature=params.num_y_feature, 
            is_training=False, 
            is_predicting=False,
            is_shuffle=False, 
            is_padding=params.is_padding,
            buffer_size=params.buffer_size, 
            pad_shape=params.pad_shape,
            batch_size=params.batch_size
        ),
        eval_steps=None,
        min_eval_frequency=params.min_eval_frequency
    )
    return experiment


def get_estimator(run_config, params):
    """
    Create an estimator for experiment
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params=params
    )


def get_hooks():
    """
    Create Hooks to monitor training
    """
    tensor_to_log = {'loss': loss}
    # tensor_to_log = {}
    logging_tensor_hook = tf.train.LoggingTensorHook(
        tensors=tensor_to_log,
        every_n_iter=2
    )
    return [logging_tensor_hook]


def run_experiment(params):
    """
    Run the training experiment.
    """
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=params.model_dir)
    experiment = experiment_fn(run_config, params)
    # train_hooks = get_hooks()
    # experiment.extend_train_hooks(train_hooks)
    experiment.train_and_evaluate()