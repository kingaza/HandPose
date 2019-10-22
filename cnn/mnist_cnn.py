'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def SimpleCNN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def KerasVGG():
    """
    模型采用类似于 VGG16 的结构：
        使用固定尺寸的小卷积核 (3x3)
        以2的幂次递增的卷积核数量 (64, 128, 256)
        两层卷积搭配一层池化
        全连接层没有采用 VGG16 庞大的三层结构，避免运算量过大，仅使用 128 个节点的单个FC
        权重初始化采用He Normal
    :return:
    """
    name = 'VGG'
    inputs = Input(shape=(28, 28, 1))
    net = inputs
    # (32, 32, 3)-->(32, 32, 64)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=128, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=128, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=256, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(8, 8, 256)
    net = Conv2D(filters=256, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=10, activation='softmax',
                kernel_initializer='he_normal')(net)
    return inputs, net, name


def KerasBN():
    """
    添加batch norm 层
    :return:
    """
    name = 'BN'
    inputs = Input(shape=(28, 28, 1))
    net = inputs

    # (32, 32, 3)-->(32, 32, 64)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=128, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=128, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=256, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=256, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=10, activation='softmax',
                kernel_initializer='he_normal')(net)

    return inputs, net, name


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs, logits, name = KerasBN()
model = Model(inputs=inputs, outputs=logits, name='model')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
