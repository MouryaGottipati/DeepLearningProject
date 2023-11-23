import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time


def depthwise_separable_conv_block(x, filters, kernel_size, strides):
    # Depthwise Separable Convolution Block
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x

def mobile_net(input_shape):
    # Input Layer
    input_tensor = layers.Input(shape=input_shape)

    # Convolution Block
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Convolution Blocks
    x = depthwise_separable_conv_block(x, 64, (3, 3), (1, 1))
    x = depthwise_separable_conv_block(x, 128, (3, 3), (2, 2))
    x = depthwise_separable_conv_block(x, 128, (3, 3), (1, 1))
    x = depthwise_separable_conv_block(x, 256, (3, 3), (2, 2))
    x = depthwise_separable_conv_block(x, 256, (3, 3), (1, 1))
    x = depthwise_separable_conv_block(x, 512, (3, 3), (2, 2))

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    num_classes = 10

    # Output Layer
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    # Model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def train(model,x_train,y_train,epochs,batch_size):    
    model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])  
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# mobile_net_model.evaluate(x_test.reshape((-1, 28, 28, 1)), y_test)
def test(model,x_test):
    # Get predicted values for the test set
    y_pred_one_hot = model.predict(x_test)
    # Convert predictions to integer labels
    y_pred_labels = np.argmax(y_pred_one_hot, axis=1)
    return y_pred_labels
    




