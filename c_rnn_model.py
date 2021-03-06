# from keras import backend as K
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Input, Dense, Activation
# from keras.layers import Reshape, Lambda, BatchNormalization
# from keras.layers.merge import add, concatenate
# from keras.models import Model
# from keras.layers.recurrent import LSTM

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.layers import Reshape, Lambda, BatchNormalization
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM


# 放缩后图片长宽分别是256，64，总共需要识别6个字符(字母+数字)，加上blank总共分类类别数为34+1
img_w = 256
img_h = 64
char_num = 7
num_class = 54 + 1


# Loss and train functions, network architecture
def ctc_lambda_func(args):
    y_pre, labels, input_length, label_length = args

    # the 2 is critical here， since the first couple outputs of the RNN tend to be garbage:

    y_pre = y_pre[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pre, input_length, label_length)


def get_model(loss_model=True):

    input_shape = (img_w, img_h, 1)    # (256, 64, 1)

    inputs = Input(name='the_input', shape=input_shape, dtype='float32')    # (None, 256, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1',
                   kernel_initializer='he_normal')(inputs)                  # (None, 256, 64, 64)
    inner = BatchNormalization()(inner)                                     # (None, 256, 64, 64)
    inner = Activation('relu')(inner)                                       # (None, 256, 64, 64)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)              # (None, 128, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2',
                   kernel_initializer='he_normal')(inner)                   # (None, 128, 32, 128)
    inner = BatchNormalization()(inner)                                     # (None, 128, 32, 128)
    inner = Activation('relu')(inner)                                       # (None, 128, 32, 128)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)              # (None, 64, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3',
                   kernel_initializer='he_normal')(inner)                   # (None, 64, 16, 256)
    inner = BatchNormalization()(inner)                                     # (None, 64, 16, 256)
    inner = Activation('relu')(inner)                                       # (None, 64, 16, 256)

    inner = Conv2D(256, (3, 3), padding='same', name='conv4',
                   kernel_initializer='he_normal')(inner)                   # (None, 64, 16, 256)
    inner = BatchNormalization()(inner)                                     # (None, 64, 16, 256)
    inner = Activation('relu')(inner)                                       # (None, 64, 16, 256)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)              # (None, 64, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5',
                   kernel_initializer='he_normal')(inner)                   # (None, 64, 8, 512)
    inner = BatchNormalization()(inner)                                     # (None, 64, 8, 512)
    inner = Activation('relu')(inner)                                       # (None, 64, 8, 512)

    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)        # (None, 64, 8, 512)
    inner = BatchNormalization()(inner)                                     # (None, 64, 8, 512)
    inner = Activation('relu')(inner)                                       # (None, 64, 8, 512)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)              # (None, 64, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', name='con7',
                   kernel_initializer='he_normal')(inner)                   # (None, 64, 4, 512)
    inner = BatchNormalization()(inner)                                     # (None, 64, 4, 512)
    inner = Activation('relu')(inner)                                       # (None, 64, 4, 512)

    # CNN to RNN, Map to Sequence
    inner = Reshape(target_shape=(32, -1), name='reshape')(inner)           # (None, 32, 4096)
    inner = Dense(64, activation='relu', name='dense1',
                  kernel_initializer='he_normal')(inner)                    # (None, 32, 64)

    # RNN layer
    lstm_1 = LSTM(256, return_sequences=True, name='lstm1',
                  kernel_initializer='he_normal')(inner)                    # (None, 32, 256)
    lstm_1b = LSTM(256, return_sequences=True,
                   go_backwards=True, name='lstm1_b',
                   kernel_initializer='he_normal')(inner)                   # (None, 32, 256)
    reversed_lstm_1b = Lambda(lambda lstm_tensor:
                              K.reverse(lstm_tensor, axes=1))(lstm_1b)      # (None, 32, 256)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])                          # (None, 32, 256)
    lstm1_merged = BatchNormalization()(lstm1_merged)                       # (None, 32, 256)

    lstm_2 = LSTM(256, return_sequences=True, name='lstm2',
                  kernel_initializer='he_normal')(lstm1_merged)             # (None, 32, 256)
    lstm_2b = LSTM(256, return_sequences=True,
                   go_backwards=True, name='lstm2_b',
                   kernel_initializer='he_normal')(lstm1_merged)            # (None, 32, 256)
    reversed_lstm_2b = Lambda(lambda lstm_tensor:
                              K.reverse(lstm_tensor, axes=1))(lstm_2b)      # (None, 32, 256)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])                  # (None, 32, 512)
    lstm2_merged = BatchNormalization()(lstm2_merged)                       # (None, 32, 512)

    # transforms RNN output to character recognition matrix:
    inner = Dense(num_class, name='dense2',
                  kernel_initializer='he_normal')(lstm2_merged)             # (None, 32, 66)
    y_pre = Activation('softmax', name='softmax')(inner)                    # (None, 32, 66)

    # create loss layer

    labels = Input(name='the_labels',
                   shape=[char_num], dtype='float32')                       # (None, 7)

    input_length = Input(name='input_length',
                         shape=[1], dtype='int64')                          # (None, 1)
    label_length = Input(name='label_length',
                         shape=[1], dtype='int64')                          # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([y_pre, labels,
                                   input_length, label_length])             # (None, 1)

    if loss_model:
        c_rnn_loss = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        return c_rnn_loss
    else:
        c_rnn = Model(inputs=[inputs], outputs=y_pre)
        return c_rnn


# Total params: 7,564,930
# Trainable params: 7,558,914
# Non-trainable params: 6,016


