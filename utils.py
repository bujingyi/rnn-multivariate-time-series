import numpy as np
import pandas as pd
import random


class DataGenerator:
    '''
    Batch data generator, a class of Iterator.
    Read in total data and yield a mini-batch data during each iteration
    Instantiate with total data containing "Offset" column.
    '''

    def __init__(self, dataFileName, featureColumns, batch_size, splitBy="key", random=False, totalData=None):
        '''
        Initializer. Read in total data. data must contain a column named "Offset" to indicate time sequence.
        :param dataFileName: path and filename to the data
        :param featureColumns: X feature columns to extract to build data. Excluding "Offset" which will be added implicitly.
        :param batch_size: number of samples to generate in one batch
        :param splitBy: default as "key" to define one sample.
        :param random: whether to shuffle samples in one batch
        '''

        self.batch_size = batch_size
        self.cursor = 0
        self.epochs = 0
        self.isRandom = random
        if dataFileName is not None:
            totalData = pd.read_csv(dataFileName)

        if dataFileName is None and totalData is None:
            raise Exception("Either dataFileName or a pandas DataFrame as totalData parameter must be provided!")

        # self.dat as list of tuple: (length, data)
        # length: seq length
        # data: data array: 2-d np array (time, feature)
        self.dat = []

        for key, subdata in totalData.groupby(splitBy):
            subdata = subdata.loc[:, featureColumns + ['Offset']]
            subdata.sort_values('Offset', inplace=True)
            subdata = subdata.drop('Offset', 1)
            # subdata: shape: (time, feature)
            self.dat.append((subdata.shape[0], subdata.values, key))
        self.dat = sorted(self.dat, key=lambda x: x[0])
        self.keys = [x[-1] for x in self.dat]
        self.dat = [x[:2] for x in self.dat]
        self.size = len(self.dat)

    # make it an iterator
    def __iter__(self):
        return self

    # python 3.x uses __next__ instead of next
    def __next__(self):
        '''
        when called in iteration method, generate a batch of data
        :return: a tuple of (padded, lengths, keys, current_size)
                padded: data of current batch, padded with 0 if effective length is less than allocated array size
                        shape as [batch size, max time step of current batch, num feature]
                lengths: effective lengths of all records in current batch (non-zero, non-padding).
                        shape: [batch size]
                keys: keys of records in current batch. "keys" are used to uniquely identify one record.
                current_size: an integer number as number of records in current batch
        '''
        # if overflowed, reset cursor and stop current iteration
        if self.cursor >= self.size:
            self.cursor = 0
            raise StopIteration()
        else:
            # calculate end index of sub-selection index: must be smaller than total length bound
            end = min(self.size, self.cursor + self.batch_size)
            current_size = end - self.cursor

            # slice out current batch data
            subdata = self.dat[self.cursor:end]
            keys = np.array(self.keys[self.cursor:end])

            # update cursor. OK if greater than bound, which will stop iteration next round
            self.cursor += current_size

            # prepare data
            lengths = np.array([x[0] for x in subdata])

            # res as list of 3-d np array (entry, time, feature)
            res = [x[1] for x in subdata]

            # Pad sequences with 0s so they are all the same length
            maxlen = lengths.max()

            # output x: shape: (batch_size:N, time, feature)
            padded = np.zeros([current_size, maxlen, res[0].shape[-1]], dtype=np.float32)

            # fill data into padded
            for i in range(current_size):
                padded[i, :(res[i].shape[0]), :] = res[i]

            # shuffle the order of data
            if self.isRandom:
                ind = np.random.choice(np.arange(current_size), current_size, replace=False)
                padded = padded[ind]
                lengths = lengths[ind]
                keys = keys[ind]

            return padded, lengths, keys, current_size


class DataSlicer:
    '''
    An iterator object to slice along time axis in one batch of data to generate data (X, y) of fixed number of time steps.
    '''

    def __init__(self, nparray, seqlen, num_step, y_dim=None, x_dim=None, minLookback=10, random=False):
        '''
        Initializer. Accept one batch of data from Batch Generator.
        :param nparray: one batch of data, shape [batch size, time (padded), num feature]
        :param seqlen: effective sequence length, length of non-zero (non-padded) values for each record. shape: [batch size]
        :param num_step: Integer value, max number of time steps to look back in one iteration.
        :param y_dim: array of indices of features to predict by the model
        :param x_dim: array of indices of features which the model uses to learn to predict
        :param minLookback: Integer value, the min number of effective time steps in one iteration such that the model can use to learn to predict
        :param random: Boolean. Indicate whether to slice the data every time step or by random time step.
                    Default False: will slice the data every time step to feed into model
                    else True: after each iteration, a random step k will be chosen between 1 and num_step (inclusive).
                                in the next iteration, data along time axis [k:k+num_step] will be yield out.
        '''
        self.dat = np.array(nparray)
        self.seqlen = np.array(seqlen)
        self.num_step = num_step
        self.maxSigLen = self.seqlen.max() - 1
        self.minLookback = minLookback
        self.counter = 0
        self.alive = self.maxSigLen > minLookback
        self.isRandom = random
        if yind is None:
            self.yind = np.arange(nparray.shape[-1])
        else:
            self.yind = yind
        if xind is None:
            self.xind = np.arange(nparray.shape[-1])
        else:
            self.xind = xind

    # make it an iterator
    def __iter__(self):
        return self

    def update(self):
        '''
        Update the object after each iteration.
        Slice out the head of data for next iteration to begin from 0 index, as convenient.
        Update effective signal length.
        Update the determinator: self.alive
        '''
        if self.isRandom:
            stride = int(self.num_step * random.random())
        else:
            stride = 1
        self.seqlen -= stride
        self.dat = self.dat[:, stride:]
        self.maxSigLen = self.seqlen.max() - 1
        self.alive = self.maxSigLen >= self.num_step

    # python 3.x uses __next__ instead of next
    def __next__(self):
        '''
        called during iteration.
        :return: a tuple of (x, y, signal_len)
                x: array of data, shape [batch size, time (num_step), num feature]
                y: array of target values in next time step. shape [batch size, num feature]
                signal_len: array of effective X lengths, shape [batch size]
        '''
        if self.alive:
            x = self.dat[:, :self.num_step, self.xind]
            y = self.dat[:, self.num_step, self.yind]
            signal_len = self.seqlen - 1
            signal_len[signal_len < self.minLookback] = 0
            signal_len[signal_len > self.num_step] = self.num_step
            self.update()
            # x: [batch, time(num_step), feature]
            return x, y, signal_len
        else:
            raise StopIteration()
