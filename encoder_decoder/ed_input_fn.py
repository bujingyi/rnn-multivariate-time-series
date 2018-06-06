import tensorflow as tf


def input_fn_data(
    features, 
    labels, 
    feature_dim, 
    label_dim, 
    is_training, 
    is_predicting, 
    is_shuffle,
    is_padding, 
    buffer_size, 
    pad_shape, 
    batch_size,  
    epoch=1,
    feature_type=tf.float32, 
    label_type=tf.float32, 
    label_shape=tf.TensorShape([]),
    label_onehot=False
):
    """
    Input function which provides batches for train or eval.
    :param features: feature for input_fn
    :param labels: label for input_fn
    :param feature_dim: feature dimension
    :param label_dim: label dimension
    :param is_training: bool, whether training
    :param is_predicting: bool, whether predicting
    :param is_shuffle: bool, whether to perform shuffle during batching
    :param is_padding: bool, whether to perform padding
    :param buffer_size: buffer size for pre-fetching data before batching
    :param pad_shape: padding shape corresponding to input data shape.
    :param batch_size: size of mini-batch in each step of gradient descent
    :param epoch: repeat number of times on training data
    :param feature_type: feature data type
    :param label_type: label data type
    :param label_shape: label shape
    :param label_onehot: bool, whether one-hot encoding for classification
    :return: batch_features, batch_labels, two tensors used by model function.
    """
    # create tf.data.Dataset to build data pipeline
    dataset_features = tf.data.Dataset.from_generator(
        lambda: features, output_types=feature_type, output_shapes=tf.TensorShape([None, feature_dim])
    )

    # assemble tf.data.Dataset
    if is_predicting:
        dataset = dataset_features
    else:
        datasetataset_labels = tf.data.Dataset.from_generator(
            lambda: labels, output_types=label_type, output_shapes=label_shape
        )
        dataset = tf.data.Dataset.zip((dataset_features, dataset_labels))

    # dataset repeat times
    if is_training:
        # shuffle with a dataset batch or not
        if is_shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.repeat(count=epoch)
    else:
        dataset = dataset.repeat(count=1)

    # padding 
    if is_padding:
        dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=pad_shape,
                                   padding_values=None)
    else:
        dataset = dataset.batch(batch_size=batch_size)

    # create iterator
    iterator = dataset.make_one_shot_iterator()

    if is_predicting:
        batch_features = iterator.get_next()
        return batch_features, None
    else:
        batch_features, batch_labels = iterator.get_next()
        if label_onehot:  # classification problem
            onehot_labels = tf.one_hot(batch_labels, label_dim)
            return batch_features, onehot_labels
        else:  # regression problem
            return batch_features, batch_labels


def input_fn(
    data_file, 
    feature_dim, 
    label_dim, 
    is_training,
    is_predicting, 
    is_shuffle, 
    is_padding, 
    buffer_size,
    pad_shape, 
    batch_size, 
    label_shape=tf.TensorShape([]), 
    epoch=1,
    feature_type=tf.float32, 
    label_type=tf.float32, 
    label_onehot=False
):
    """
    TensorFlow estimator input function which provides batches for train or eval
    :param data_file: data files
    :param feature_dim: feature dimension
    :param label_dim: label dimension
    :param is_training: bool, whether training
    :param is_predicting: bool, whether predicting
    :param is_shuffle: bool, whether to perform shuffle during batching
    :param is_padding: bool, whether to perform padding
    :param buffer_size: buffer size for pre-fetching data before batching
    :param pad_shape: padding shape corresponding to input data shape.
                    Example: if data is of shape [batch, max time, num feat]
                    then each record is of shape [max time, num feat]
                    so the pad shape is [None, None]
                    if data is of shape ([batch, max time, num feat], [batch, max time, num feat])
                    then each record is of shape ([max time, num feat], [max time, num feat])
                    so the pad shape is ([None, None], [None, None])
    :param batch_size: size of mini-batch in each step of gradient descent
    :param epoch: repeat number of times on training data
    :param feature_type: feature data type
    :param label_type: label data type
    :param label_shape: label shape
    :param label_onehot: bool, whether one-hot encoding for classification
    :return: batch_features, batch_labels, two tensors used by model function.
    :return: Estimator input_fn
    """
    features = None
    labels = None
    # TODO: create features and labels from data files...
    return input_fn_data(
        features=features, 
        labels=labels, 
        feature_dim=feature_dim, 
        label_dim=label_dim, 
        is_training=is_training, 
        is_predicting=is_predicting, 
        is_shuffle=is_shuffle,
        is_padding=is_padding, 
        buffer_size=buffer_size, 
        pad_shape=pad_shape, 
        batch_size=batch_size, 
        label_shape=tf.TensorShape([]), 
        epoch=1,
        feature_type=tf.float32, 
        label_type=tf.float32, 
        label_onehot=False
    )