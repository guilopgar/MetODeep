import keras.layers.core as core
import keras.layers.convolutional as conv
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation, Input
from keras.regularizers import l1
from keras.optimizers import Adam
from attention import Attention,myFlatten

from abc import ABCMeta, abstractmethod


class MultiCNN:
    """
    Abstract class that initializes and trains multiple convolutional neural network models.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_model(self, input_row, input_col):
        """
        Abstract method to create a certain CNN model.
        """
        pass

    def cnn_train(self, trainX, trainY,valX=None, valY=None, batch_size=1200, nb_epoch=500, earlystop=None,
                  transferlayer=1, weights=None, forkinas=False, compiletimes=0, compilemodels=None,
                  loss='binary_crossentropy', optimizer=Adam(), metrics='accuracy'):
        """
        It trains a model, either an existing one or a new one created in the function.

        :return: A fitted model on the training data.
        """
        input_row = trainX.shape[2]
        input_col = trainX.shape[3]

        trainX_t = trainX
        valX_t = valX
        trainX_t.shape = (trainX_t.shape[0], input_row, input_col)
        if (valX is not None):
            valX_t.shape = (valX_t.shape[0], input_row, input_col)

        if compiletimes == 0:
            cnn = self.create_model(input_row, input_col)
            cnn.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

        else:
            cnn = compilemodels

        if (weights is not None and compiletimes == 0):  # for the first time
            w = isinstance(weights, basestring)  # if this was a big if-else, lot of code would be repeated
            if w:
                print "load weights:" + weights
            else:
                print "load weights from model"
            if not forkinas:
                if w:
                    cnn.load_weights(weights)
                else:
                    cnn.set_weights(weights.get_weights())
            else:
                if w:
                    cnn2 = self.create_model(input_row, input_col)
                    cnn2.load_weights(weights)
                else:
                    cnn2 = weights
                for l in range((len(cnn2.layers) - transferlayer)):  # the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                    # cnn.layers[l].trainable= False  # for frozen layer

        if (valX is not None):
            if (earlystop is None):
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch,
                                     validation_data=(valX_t, valY))
            else:
                # nb_epoch set to a very big value since earlystop used
                fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=10000,
                                     validation_data=(valX_t, valY),
                                     callbacks=[EarlyStopping(monitor='val_loss', patience=earlystop)])
        else:
            fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)

        return cnn


class DeepBind(MultiCNN):
    """
    Implements the abstract method to create a DeepBind CNN model
    """
    def __init__(self):
        """
        All the attributes are initialized with their default values, though this could be done adding arguments
        to the constructor.
        """
        self.n_filter = 1000
        self.kernel_size = 21
        self.stride = 1
        # 'same' padding is added in order to apply a max-pooling of size 33
        self.padding = 'same'
        self.kernel_init = 'he_normal'
        self.conv_act = 'relu'
        self.drop_1 = 0.4
        self.drop_2 = 0
        self.pool = 33
        self.n_class = 2
        self.out_act = 'softmax'

    def create_model(self, input_row, input_col):
        input = Input(shape=(input_row, input_col))
        x = conv.Conv1D(filters=self.n_filter, kernel_size=self.kernel_size, strides=self.stride, padding=self.padding,
                        kernel_initializer=self.kernel_init)(input)
        x = Dropout(self.drop_1)(x)
        x = Activation(self.conv_act)(x)
        x = conv.MaxPooling1D(pool_size=self.pool)(x)
        x = core.Flatten()(x)
        output = Dense(units=self.n_class, activation=self.out_act, kernel_initializer=self.kernel_init)(x)

        return Model(input, output)


class MusiteDeep(MultiCNN):
    """
    Implements the abstract method to create a MusiteDeep CNN model
    """
    def __init__(self):
        """
        All the attributes are initialized with their default values, though this could be done adding arguments
        to the constructor.
        """
        self.filtersize1 = 1
        self.filtersize2 = 9
        self.filtersize3 = 10
        self.filter1 = 200
        self.filter2 = 150
        self.filter3 = 200
        self.dropout1 = 0.75
        self.dropout2 = 0.75
        self.dropout4 = 0.75
        self.dropout5 = 0.75
        self.dropout6 = 0
        self.L1CNN = 0
        self.nb_classes = 2
        self.actfun = 'relu'
        self.kernel_init = 'he_normal'
        self.attentionhidden_x = 10
        self.attentionhidden_xr = 8
        self.attention_reg_x = 0.151948
        self.attention_reg_xr = 2
        self.dense_size1 = 149
        self.dense_size2 = 8
        self.dropout_dense1 = 0.298224
        self.dropout_dense2 = 0

    def create_model(self, input_row, input_col):
        input = Input(shape=(input_row, input_col))
        x = conv.Convolution1D(self.filter1, self.filtersize1, kernel_initializer=self.kernel_init,
                               kernel_regularizer=l1(self.L1CNN), padding="same")(input)
        x = Dropout(self.dropout1)(x)
        x = Activation(self.actfun)(x)
        x = conv.Convolution1D(self.filter2, self.filtersize2, kernel_initializer=self.kernel_init,
                               kernel_regularizer=l1(self.L1CNN), padding="same")(x)
        x = Dropout(self.dropout2)(x)
        x = Activation(self.actfun)(x)
        x = conv.Convolution1D(self.filter3, self.filtersize3, kernel_initializer=self.kernel_init,
                               kernel_regularizer=l1(self.L1CNN), padding="same")(x)
        x = Activation(self.actfun)(x)
        x_reshape = core.Reshape((x._keras_shape[2], x._keras_shape[1]))(x)
        x = Dropout(self.dropout4)(x)
        x_reshape = Dropout(self.dropout5)(x_reshape)
        decoder_x = Attention(hidden=self.attentionhidden_x, activation='linear', init=self.kernel_init,
                              W_regularizer=l1(self.attention_reg_x))  # success
        decoded_x = decoder_x(x)
        output_x = myFlatten(x._keras_shape[2])(decoded_x)
        decoder_xr = Attention(hidden=self.attentionhidden_xr, activation='linear', init=self.kernel_init,
                               W_regularizer=l1(self.attention_reg_xr))
        decoded_xr = decoder_xr(x_reshape)
        output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)

        output = concatenate([output_x, output_xr])
        output = Dropout(self.dropout6)(output)
        output = Dense(self.dense_size1, kernel_initializer='he_normal', activation='relu')(output)
        output = Dropout(self.dropout_dense1)(output)
        output = Dense(self.dense_size2, activation='relu', kernel_initializer=self.kernel_init)(output)
        output = Dropout(self.dropout_dense2)(output)
        output = Dense(self.nb_classes, kernel_initializer=self.kernel_init, activation='softmax')(output)

        return Model(input, output)


if __name__ == "__main__":
    a = MusiteDeep().create_model(33, 21)
    print a.summary()
