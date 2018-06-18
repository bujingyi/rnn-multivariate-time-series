import tensorflow as tf
import os


class GeneralizedTransducer:
    """
    Generalized RNN transducer architecture for multivariante time series forcasting.
    Classic transducer and transducer could be regarded as two special cases of GT.
    """
    def __init__(
        self, 
        x_dim,
        y_dim,
        rnn_structure,
        drop_x_head, 
        drop_y_head,
        scope='Generalized_Transducer',
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
        summaries_dir=None,
    ):
    """
    Transducer initializer
    :param x_dim: input feature x dimension
    :param y_dim: input feature y dimension
    :param rnn_structure: define the structure of the transducer. list of state tuples. 
        Each tuple is a combinator of state size and drop out keep rate.[(9, .9), (20, .7)] 
        means two stacked layers of 9 hidden units in first layer with 0.9 dropout KEEP rate 
        and 20 units in second layer of 0.7 dropout KEEP rate.
    :param drop_x_head: drop the first several x data points during loss calculation
    :param drop_y_head: drop the first several y data points during loss calculation
    :param scope: TensorFlow variable scope
    :param initial_learning_rate: initial gradient descent learning rate
    :param decay_steps: GD learning rate decay steps
    :param decay_rate: GD learning rate decay rate
    :param summaries_dir: TensorBoard summaries
    """
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.rnn_structure = rnn_structure
    self.drop_x_head = drop_x_head
    self.drop_y_head = drop_y_head
    self.initial_learning_rate = initial_learning_rate
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate

    # build TensorFlow computation graph
    with tf.variable_scope(self.scope):
        self._build_model()
        if summaries_dir:
            summaries_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)

    def _build_model(self):
        """
        Build computation graph
        """
        # extract the number of RNN layers from rnn_structure 
        num_layers = len(self.rnn_structure)

        # placeholders
        # x_ph shape [batch size, max time (padded), num feature]
        # y_ph shape [batch size, max time (padded), num feature]
        self.x_ph = tf.placeholder(tf.float32, [None, None, self.x_dim], name='inputs')
        self.y_ph = tf.placeholder(tf.float32, [None, None, self.y_dim], name='targets')
        self.seqlen_ph = tf.placeholder(tf.int32, [None, 1])
        # LSTM init_state of shape [num layers, 2:(cell, hidden), batch size, max of state size]
        self.init_state_ph = tf.placeholder(tf.float32, [num_layers, 2, None, max(i[0] for i in rnn_structure)])

        with tf.name_scope('Transducer_LSTM'):
            # state_per_layer_list is a list of state tuples. Each state tuple is a tuple (cell, hidden)
            # cell/hidden is a 2-D array [max batch size, max state size]
            # thus slice out the top k values if state size of this layer is k
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_state_tuple = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(
                    state_per_layer_list[idx][0][:, :rnn_structure[idx][0]],
                    state_per_layer_list[idx][1][:, :rnn_structure[idx][0]]
                ) for idx in range(num_layers)]
            )
            # define single LSTM cell
            def lstm_cell(state_size, dropKeepRate):
                cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size, state_is_tuple=True)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropKeepRate)
                return cell

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [lstm_cell(rnn_structure[idx][0], rnn_structure[idx][1]) for idx in range(num_layers)],
                state_is_tuple=True
            )

            # outputs is [batch size, max time, last layer state size]
            # position over provided sequence length (due to padding) will be filled with value of zero
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=cell, dtype=tf.float32, inputs=x,
                initial_state=rnn_state_tuple, sequence_length=seqlen
            )

        # predictions, loss, train
        last_layer_state_size = rnn_structure[-1][0]
        output_shape = tf.shape(outputs)
        current_batch_size, max_time = op_shape[0], op_shape[1]
        # fully connected layer before output
        with tf.variable_scope('Output'):
            Wh = tf.get_variable('Wh', [last_layer_state_size, last_layer_state_size])
            bh = tf.get_variable('bh', [last_layer_state_size], initializer=tf.constant_initializer(0.0))
            W = tf.get_variable('W', [last_layer_state_size, self.y_dim])
            b = tf.get_variable('b', [self.y_dim], initializer=tf.constant_initializer(0.0))

        # predictions
        with tf.name_scope('Prediction'):
            output_reshape = tf.reshape(
                outputs[:, self.drop_x_head:, :],
                [current_batch_size * (max_time - self.drop_x_head), last_layer_state_size]
            )
            # preds_reshape: [current_batch_size * (max_time - drop_head), y_dim]
            preds_reshape = tf.matmul(tf.matmul(output_reshape, Wh) + bh, W) + b
            self.preds = tf.reshape(preds_reshape, [current_batch_size, max_time - self.drop_x_head, self.y_dim])

        # loss is defined as avg across all features for all samples in batch excluding those positions padded with 0
        # Note: input y values must be padded with and only with 0, tail-end padded
        with tf.name_scope('Loss'):
            # Get the tail of Y along time, to match with X value with head dropped.
            y = y[:, -(max_time - self.drop_y_head):, :]
            self.loss = tf.losses.mean_squared_error(y, preds)

        # training with gradient descent, global variables
        with tf.name_scope('Global'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.initial_learning_rate,
                global_step=global_step,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate
            )

        # define train operation
        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

        # summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
        ])

    def update(self, sess, x, y, seqlen, init_state):
        """
        Updates the transducer towards the given targets
        :param sess: TensorFlow session
        :param x: input x of shape [batch_size, known_length]
        :param y: targets to predict of shape[batch_size, pred_length]
        :param seqlen: sequence length of shape [batch_size, 1]
        :param init_state: is to set LSTM state zero before each batch begins
        :return: loss, final_state, learning_rate
        """
        feed_dict = {
            self.x_ph: x, 
            self.y_ph: y, 
            self.seqlen_ph: seqlen, 
            self.init_state_ph: init_state
        }
        # update
        preds, loss, final_state, _, learning_rate, global_step, summaries = sess.run(
            [
                self.preds,
                self.loss, 
                self.final_state, 
                self.train_op, 
                self.learning_rate, 
                tf.train.get_global_step(), 
                self.summaries
            ],
            feed_dict
        )
        if self.summary_writer and save:
            self.summary_writer.add_summary(summaries, global_step)
        return preds, loss, final_state, learning_rate

    def predict(self, sess, x, y, seqlen, init_state):
        """
        Updates the transducer towards the given targets
        :param sess: TensorFlow session
        :param x: input x of shape [batch_size, known_length]
        :param y: targets to predict of shape[batch_size, pred_length]
        :param seqlen: sequence length of shape [batch_size, 1]
        :param init_state: is to set LSTM state zero before each batch begins
        :return: predictions, final_state
        """
        feed_dict = {
            self.x_ph: x, 
            self.y_ph: y, 
            self.seqlen_ph: seqlen, 
            self.init_state_ph: init_state
        }
        return sess.run([self.preds, self.final_state], feed_dict)


