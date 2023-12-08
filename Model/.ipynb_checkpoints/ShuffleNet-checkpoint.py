import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


# Define depthwise separable convolution block
def depthwise_separable_conv_block(x, filters, kernel_size, strides):
    # Depthwise Separable Convolution Block
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Channel Shuffle Operation
def channel_shuffle(x, groups):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_group = channels // groups
    x = tf.reshape(x, [-1, height, width, groups, channels_per_group])
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    x = tf.reshape(x, [-1, height, width, channels])
    return x

    

# Define ShuffleNet-like model
def shuffle_net(input_shape):
    # Input Layer
    input_tensor = layers.Input(shape=input_shape)

    # Convolution Block
    x = layers.Conv2D(24, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Depthwise Separable Convolution Blocks
    x = depthwise_separable_conv_block(x, 24, (3, 3), (1, 1))
    x = depthwise_separable_conv_block(x, 24, (3, 3), (1, 1))
    x = depthwise_separable_conv_block(x, 24, (3, 3), (1, 1))

    print(x.shape)
    print("Before")

    x = channel_shuffle(x, groups=3)

    print(x.shape)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    num_classes=10

    # Output Layer
    output_tensor = layers.Dense(num_classes, activation='softmax')(x)

    # Model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model


def train(model,x_train,y_train,epochs,batch_size):
    model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0)
    return model
def test(model,x_test):
    # Get predicted values for the test set
    y_pred_one_hot = model.predict(x_test)
    # Convert predictions to integer labels
    y_pred_labels = np.argmax(y_pred_one_hot, axis=1)
    return y_pred_labels