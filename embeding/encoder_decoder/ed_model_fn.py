import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys

from encoder_decoder.ed_build_blocks import unidirect_lstm_layer, output_layer


def model_fn(features, labels, mode, params):
    """
    Model_fn for TensorFlow estimator
    :param features: tensor passed in by input function
    :param labels: tensor passed in by input function
    :param mode: ModeKeys indicate whether training, evaluating or predicting
    :param params: tf.contrib.training.HParams, named tuple.
    :return: tf.estimator.EstimatorSpec
    """
    # check training states
    is_training = mode == ModeKeys.TRAIN
    is_predicting = mode == ModeKeys.PREDICT

    # choose GPU
    with tf.device('/device:GPU:0'):
        with tf.variable_scope('input'):
            features.set_shape([None, None, params.num_feature])
            if not is_predicting:
                labels.set_shape([None, None, params.num_feature])
                seqlen = get_seq_length(features, params.pad_value)
                current_batch_size, max_time = tf.shape(features)[0], tf.shape(features)[1]

        with tf.name_scope('Encoder'):
            encoder_outputs, final_encoder_state = unidirect_lstm_layer(params.rnn_structure, features, seqlen)

        with tf.variable_scope('Output'):
            # due to padding, need to extract the last effective output according seqlen
            last_effective_output = get_last_effective_output(encoder_outputs, seqlen) 

        # if predicting, return
        if is_predicting:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=last_effective_output
            )

        with tf.variable_scope('Decoder'):
            # exp_lastOutput shape: [batch size, 1, last layer hidden size]
            exp_last_output = tf.expand_dims(last_effective_output, 1)
            # encoder's last hidden state is feeded as input for every time points of decoder
            decoder_input = tf.tile(exp_last_output, [1, max_time, 1])
            last_hidden_size = params.rnn_structure[-1][0]
            decoder_input.set_shape([None, None, last_hidden_size])
            decode_outputs, decoder_final_state = unidirect_lstm_layer(
                params.rnn_structure, decoder_input, seqlen
            )

        # reconstruct the encoder input sequence
        with tf.name_scope('Reconstruction'):
            recon_output = output_layer(
                num_feature=params.num_feature, 
                rnn_structure=params.rnn_structure, 
                inputs=decode_outputs, 
                current_batch_size=current_batch_size,
                max_time=max_time, 
                dense_drop_1=params.dense_drop_1, 
                dense_drop_2=params.dense_drop_2, 
                mode=mode
            )

        # calculate reconstruction loss
        loss = get_loss(labels, recon_output, seqlen)
        eval_metric_ops = get_eval_metric_ops(
            targets=labels, 
            preds=recon_output, 
            seqlen=seqlen,
            task_type=params.task_type  # TODO: not coded yet
        )

        # estimator mode
        if mode == tf.estimator.ModeKeys.TRAIN:  # ModeKeys.TRAIN
            train_op, _, _ = get_train_op_fn(loss, params)
        else:
            train_op = None

        # hooks for monitoring
        logging_hook = tf.train.LoggingTensorHook(
            tensors={'loss': loss},
            every_n_iter=10
        )

        # estimator model_fn return an EstimatorSpec
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=lastOutput,
            loss=loss,
            train_op=train_op,
            training_hooks=[logging_hook],
            eval_metric_ops=eval_metric_ops
        )


# below are functions to build up model_fn
def get_last_effective_output(output, seq_length):
    """
    extract last effective output dynamically unrolled from LSTM layer.
    :param output: output dynamically unrolled from LSTM layer. shape [batch, max time, last layer hidden size]
    :param seq_length: effective length of input sequence data tensor. 
    :return: last effective output. shape [batch size, last layer hidden size]
    """
    current_batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    last_layer_state_size = tf.shape(output)[2]

    # flat nested outputs from dynamic unrolling
    flat = tf.reshape(output, [-1, last_layer_state_size])

    # build index of last non-pad position
    # first build index as beginning of each batch in the flattened array
    # next add effective length minus 1 (because 0 index)
    index = tf.range(0, current_batch_size) * max_length + (seq_length - 1)

    # get last effective values
    last_effective = tf.gather(flat, index)
    return last_effective


def get_loss(targets, preds, seqlen, task_type):
    """
    Calculate loss
    :param targets: true values
    :param preds: predicted values
    :param seqlen: number of time steps in targets
    :param task_type: indicating task type
    :return: loss
    """
    # TODO: calculate different loss types according to different task types...
    # example: seq2seq, regression
    mask = build_mask(preds, seqlen)
    loss_args = {'weights': mask}
    loss_func = tf.losses.mean_squared_error
    loss = loss_func(targets, preds, **loss_args)
    return loss


def get_train_op_fn(loss, params):
    """ 
    TensorFlow train operation 
    """
    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    gvs = optimizer.compute_gradients(loss)
    # using gradient clipping to eliminate gradient explosion that may happen during rnn training
    capped_gvs = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
    return train_op, gvs, capped_gvs


def get_eval_metric_ops(targets, preds, seqlen, task_type):
    """
    Assemble eval_metric_ops for model_fn
    :param targets: true values
    :param preds: predicted values
    :param seqlen: number of time steps in targets
    :param task_type: indicating task type
    :return: eval_metric_ops, a dictionary
    """
    # TODO: return different eval_metric_ops according to task type - classification/regression...
    # example: seq2seq, regression
    mask = build_mask(preds, seqlen)
    return {
        'Eval_Error': tf.metrics.mean_squared_error(
            labels=targets, 
            predictions=preds, 
            weights=mask
        )
    }


def get_seq_length(sequence, pad_value, eps=1e-6):
    """
    Due to padding, sequences have effective length. 
    Calculate the effective length of sequence input based on provided padding value.
    :param sequence: input sequence tensor with padding
    :param pad_value: a scalar number as padding value.
    :param eps: the tolerance used for comparing input value and padding value to determine if the value
              is effective value or padding value.
    :return: effective length of shape [batch size]
    """
    not_zero = tf.greater(tf.reduce_max(tf.abs(sequence - pad_value), 2), eps)
    length = tf.reduce_sum(tf.cast(not_zero, tf.int32), 1)
    length = tf.cast(length, tf.int32)
    return length


def build_mask(preds, seqlen):
    """
    Mask indicating effective digits of a batch of sequence data.
    :param preds: batch of sequence data
    :param seqlen: effective length of sequence data
    :return: a tensor of float (0.0 or 1.0). 1.0 for effective digits
    """
    num_feature = tf.shape(preds)[2]
    seqlen_transposed = tf.expand_dims(seqlen, 1)
    # expand from [batch size] to [batch size, 1]
    # [[5],
    #  [4],
    #  [3]]
    range_row = tf.expand_dims(tf.range(0, tf.shape(preds)[1], 1), 0)
    # [[1,2,3,4,5,6,7,8]]
    mask = tf.cast(tf.less_equal(range_row, seqlen_transposed), tf.float32)
    # [[T,T,T,T,T,F,F,F],
    #  [T,T,T,T,F,F,F,F],
    #  [T,T,T,F,F,F,F,F]]

    # after casting:
    # [[1,1,1,1,1,0,0,0],
    #  [1,1,1,1,0,0,0,0],
    #  [1,1,1,0,0,0,0,0]]
    # mask shape: [batch size, max time]
    mask = tf.expand_dims(mask, 2)
    # mask shape: [batch size, max time, 1]
    mask = tf.tile(mask, [1, 1, num_feature])
    return mask



    