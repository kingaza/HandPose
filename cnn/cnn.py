import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# only useful for working with matplotlib on OSX, 
# which is not a framework build of Python
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import buildPosesDataset as dataset


def SimpleCNN(input_shape, n_class):
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
    model.add(Dense(n_class, activation='softmax'))

    # categorical ce since we have multiple classes (10)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=0.01),
                metrics=['accuracy'])    

    return model


def KerasVGG(input_shape, n_class):
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
    inputs = Input(shape=input_shape)
    net = inputs
    # (32, 32, 3)-->(32, 32, 64)
    net = Conv2D(filters=16, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Conv2D(filters=16, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=32, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=32, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(8, 8, 256)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128 or 64??
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=n_class, activation='softmax',
                kernel_initializer='he_normal')(net)
    return inputs, net, name


def KerasBN(input_shape, n_class):
    """
    添加batch norm 层
    :return:
    """
    name = 'BN'
    inputs = Input(shape=input_shape)
    net = inputs

    # (32, 32, 3)-->(32, 32, 64)
    net = Conv2D(filters=16, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(32, 32, 64)
    net = Conv2D(filters=16, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (32, 32, 64)-->(16, 16, 64)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=32, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 64)-->(16, 16, 128)
    net = Conv2D(filters=32, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (16, 16, 128)-->(8, 8, 128)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 128)-->(8, 8, 256)
    net = Conv2D(filters=64, kernel_size=3, strides=1,
                 padding='same', activation='relu',
                 kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    # (8, 8, 256)-->(4, 4, 256)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # (4, 4, 256) --> 4*4*256=4096
    net = Flatten()(net)
    # 4096 --> 128 or 64 ??
    net = Dense(units=128, activation='relu',
                kernel_initializer='he_normal')(net)
    # Dropout
    net = Dropout(0.5)(net)
    # 128 --> 10
    net = Dense(units=n_class, activation='softmax',
                kernel_initializer='he_normal')(net)

    return inputs, net, name



def train():
    batch_size = 128
    epochs = 10
    model_name = "cnn/models/handposes_vgg64_v1.h5"

    # input image dimensions
    img_rows, img_cols = 64, 64

    # the data, shuffled and split between train and test sets
    x_train, y_train, x_test, y_test = dataset.load_data(poses=["all"], im_size=64)

    num_classes = len(np.unique(y_test))

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    inputs, logits, name = KerasVGG(input_shape, num_classes)
    model = Model(inputs=inputs, outputs=logits, name=name)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    #model = SimpleCNN(input_shape, num_classes)



    model.summary()            

    ####### TRAINING #######
    hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save(model_name)

    # plotting the metrics
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
