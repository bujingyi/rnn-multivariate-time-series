import tensorflow as tf
import numpy as np
from time import time as timer

# from utils import DataSlicer


def transducer_step(
    sess,
    transducer, 
    x,
    y,
    seqlen,
    init_state,
    training=True
):
    """
    One step for either update or predict
    :param sess: TensorFlow session
    :param transducer: transducer instance
    :param x: input data
    :param y: targets
    :param training: boolean. true is training, false is validation or prediction
    :return: preds, loss, final_state
    """
    # if it is now training the transducer, update the network, or predict with the network
    if training:
        # update the transducer with one gradient decent step
        preds, loss, final_state, learning_rate = transducer.update(
            sess=sess, 
            x=x, 
            y=y, 
            seqlen=seqlen, 
            init_state=init_state
        )
    else:
        # predict with the transducer
        preds, final_state = transducer.predict(
            sess=sess,
            x=x,
            y=y,
            seqlen=seqlen,
            init_state=init_state
        )
        # diff: shape [batch size, x_dim]
        diff = ((preds - y) ** 2)
        # update total_loss by average of current time step across all samples in current batch
        # excluding those padded too much with 0
        loss = diff[seqlen > 0, :].mean(axis=1).sum()
    return preds, loss, final_state


def transducer_epoch(
    sess,
    transducer,
    data_generator,
    rnn_structure,
    t_timer,
    training=True
):
    """
    One epoch for either training or validation
    :param sess: TensorFlow session
    :param transducer: transducer instance
    :param data_generator: iterator for generating batches
    :param t_timer: in case running time is too long
    :param training: boolean. true is training, false is validation or prediction
    :return: averaged_loss for the epoch
    """
    # one epoch for either training or prediction
    num_layers = len(rnn_structure)
    training_loss = 0
    validation_loss = 0

    # Batch Generator returns a tuple of (data, sequence length, keys, current size)
    # data: np 3-D array [record, time, feature]
    # sequence length: np 1-D array [record]
    # keys: np 1-D array [record]
    # current size: scalar number of samples in current batch
    for batch, seqlen, keys, current_size in data_generator:
        if timer() - t_timer > timeLimit:
            timeout = True
            print("Training timeout during batching at epoch:", epoch)
            break

        # transducer state initialization, "stateful" only make sense for acceptor
        init_state = np.zeros((num_layers, 2, current_size, max(i[0] for i in rnn_structure)))

        # one step
        preds, loss, final_state = transducer_step(
            sess=sess, 
            transducer=transducer, 
            x=x, 
            y=y, 
            seqlen=seqlen, 
            init_state=init_state, 
            training=training
        )

        # accumulate losses
        total_loss += loss

    # if training is stopped due to timeout
    if timeout:
        # TODO: raise timeout exception
        raise Exception("Timeout!")
    return total_loss


def transducer_train(
    sess,
    transducer, 
    data_generator,
    num_epochs, 
    rnn_structure,
    valid_data_generator=None,
    verbose=True,  
    time_limit=float("Inf"),
):
    """
    Train the transducer
    :param sess: TensorFlow session
    :param transducer: transducer instance
    :param data_generator: iterator for generating batches
    :param data_slicer: iterator for slicing each batch to fit the transducer network
    :param num_epochs: number of epochs to run for training the network
    :param rnn_structure: define the structure of the transducer. 
        list of state tuples. Each tuple is a combinator of state size and drop out 
        keep rate.[(9, .9), (20, .7)] means two stacked layers of 9 hidden units in 
        first layer with 0.9 dropout KEEP rate and 20 units in second layer of 0.7 dropout KEEP rate.
    :param valid_data_generator: iterator for generating validation batches
    :param verbose: for debug and monitor purpose
    :param time_limit: in case running time is too long
    :return: training_losses, validation_losses
    """
    # training statistics 
    training_losses = []
    validation_losses = []

    # initilize the timer, in case the training time is too long
    t_timer = timer()
    timeout = False

    # start training
    epoch = 0

    # extract the number of RNN layers from rnn_structure
    num_layers = len(self.rnn_structure)

    while epoch <= num_epochs:
        if timer() - t > timeLimit:
            timeout = True
            print("Training timeout at beginning of epoch:", epoch)
            break

        # take one epoch
        averaged_loss = transducer_epoch(
            sess=sess,
            transducer=transducer,
            data_generator=data_generator,
            rnn_structure=rnn_structure,
            t_timer=t_timer,
            training=True
        )

        # training statistics
        training_losses.append(averaged_loss)

        # when one epoch is done, print training loss, test with validation
        if verbose:
            print("Average training loss for Epoch{}, loss:{}".format(epoch, averaged_loss))

        # if there are data for validation
        if valid_data_generator:
            averaged_valid_loss = transducer_epoch(
                sess=sess,
                transducer=transducer,
                data_generator=valid_data_generator,
                rnn_structure=rnn_structure,
                t_timer=t_timer,
                training=False
            )
            # training statistics
            validation_losses.append(averaged_valid_loss)
            print("Average validation loss for Epoch{}, loss:{}".format(epoch, averaged_valid_loss))

        # next epoch
        epoch += 1
    return training_losses, validation_losses