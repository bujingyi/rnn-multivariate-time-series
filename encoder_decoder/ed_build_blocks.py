import tensorflow as tf


def unidirect_lstm_layer(rnn_structure, inputs, seqlen):
    """
    Construct an LSTM processing input sequence data with dynamic unrolling
    :param rnn_structure: define the rnn structure, list of tuples
    :param inputs: input data
    :param seqlen: effective length of input sequence data
    :return: outputs, final_state from tf.nn.dynamic_rnn
    """
    # extract rnn layer number from rnn_structure
    num_layers = len(rnn_structure)

    # define LSTM cell
    def lstm_cell(state_size, drop_keep_rate):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=drop_keep_rate)
        return cell

    # stack LSTM by num_layers
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(rnn_structure[idx][0], rnn_structure[idx][1]) for idx in range(num_layers)],
        state_is_tuple=True
    )

    outputs, final_state = tf.nn.dynamic_rnn(
        cell=cell, 
        dtype=tf.float32, 
        inputs=inputs,
        initial_state=None, 
        sequence_length=seqlen
    )

    return outputs, final_state


def output_layer(
    num_feature, 
    rnn_structure, 
    inputs, 
    current_batch_size, 
    max_time, 
    dense_drop_1, 
    dense_drop_2, 
    mode
):
    """
    Output layer
    :param num_feature: number of output features after this layer.
    :param rnn_structure: define the rnn structure, list of tuples 
    :param outputs: output dynamically unrolled from LSTM layer. shape [batch, max time, last layer hidden size]
    :param current_batch_size: number of samples in this current batch
    :param max_time: maximum number of time steps in current batch
    :param dense_drop_1: first full connected layer dropout rate
    :param dense_drop_2: second full connected layer dropout rate
    :param mode: training/prediction mode
    :return: output after fully connected layers
    """
    # define fully connected layers
    def dense_layers(last_output, last_layer_state_size, num_feature, dense_drop_1, dense_drop_2, mode):
        # first fully connected layer
        fc1 = tf.layers.dense(inputs=last_output, units=last_layer_state_size, activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=fc1, rate=dense_drop_1, training=mode == tf.estimator.ModeKeys.TRAIN)
        # second fully connected layer
        fc2 = tf.layers.dense(inputs=dropout1, units=last_layer_state_size, activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=fc2, rate=dense_drop_2, training=mode == tf.estimator.ModeKeys.TRAIN)
        outputs = tf.layers.dense(inputs=dropout2, units=num_feature)
        return outputs

    last_layer_state_size = rnn_structure[-1][0]
    output_flat = tf.reshape(outputs, [current_batch_size * max_time, last_layer_state_size])
    preds_flat = dense_layers(output_flat, last_layer_state_size, num_feature, dense_drop_1, dense_drop_2, mode)
    outputs = tf.reshape(preds_flat, [current_batch_size, max_time, num_feature])

    return outputs


