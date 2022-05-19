import tensorflow as tf
import tensorflow_addons as tfa


# custom convolution
def conv_bn(filters,
            model,
            kernel=3,
            activation='relu', 
            strides=(1, 1),
            padding='same'):
    conv = tf.keras.layers.Conv2D(filters, kernel, strides, padding)(model)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)

    return conv


def get_callbacks(name):
    return [
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}.h5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    ]


# building the segnet model
def segnet():
    inputs = tf.keras.layers.Input((256, 256, 3))

    # encoder
    filters = 64
    encoder1 = conv_bn(filters, inputs)
    encoder1 = conv_bn(filters, encoder1)
    pool1, mask1 = tf.nn.max_pool_with_argmax(encoder1, 3, 2, padding="SAME")

    filters *= 2
    encoder2 = conv_bn(filters, pool1)
    encoder2 = conv_bn(filters, encoder2)
    pool2, mask2 = tf.nn.max_pool_with_argmax(encoder2, 3, 2, padding="SAME")

    filters *= 2
    encoder3 = conv_bn(filters, pool2)
    encoder3 = conv_bn(filters, encoder3)
    encoder3 = conv_bn(filters, encoder3)
    pool3, mask3 = tf.nn.max_pool_with_argmax(encoder3, 3, 2, padding="SAME")

    filters *= 2
    encoder4 = conv_bn(filters, pool3)
    encoder4 = conv_bn(filters, encoder4)
    encoder4 = conv_bn(filters, encoder4)
    pool4, mask4 = tf.nn.max_pool_with_argmax(encoder4, 3, 2, padding="SAME")

    encoder5 = conv_bn(filters, pool4)
    encoder5 = conv_bn(filters, encoder5)
    encoder5 = conv_bn(filters, encoder5)
    pool5, mask5 = tf.nn.max_pool_with_argmax(encoder5, 3, 2, padding="SAME")

    # decoder
    unpool1 = tfa.layers.MaxUnpooling2D()(pool5, mask5)
    decoder1 = conv_bn(filters, unpool1)
    decoder1 = conv_bn(filters, decoder1)
    decoder1 = conv_bn(filters, decoder1)

    unpool2 = tfa.layers.MaxUnpooling2D()(decoder1, mask4)
    decoder2 = conv_bn(filters, unpool2)
    decoder2 = conv_bn(filters, decoder2)
    decoder2 = conv_bn(filters/2, decoder2)

    filters /= 2
    unpool3 = tfa.layers.MaxUnpooling2D()(decoder2, mask3)
    decoder3 = conv_bn(filters, unpool3)
    decoder3 = conv_bn(filters, decoder3)
    decoder3 = conv_bn(filters/2, decoder3)

    filters /= 2
    unpool4 = tfa.layers.MaxUnpooling2D()(decoder3, mask2)
    decoder4 = conv_bn(filters, unpool4)
    decoder4 = conv_bn(filters/2, decoder4)

    filters /= 2
    unpool5 = tfa.layers.MaxUnpooling2D()(decoder4, mask1)
    decoder5 = conv_bn(filters, unpool5)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder5)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder5)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    # selecting custom adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss='mse',
        loss_weights=[0.3, 0.7],
        optimizer=optimizer,
        metrics='accuracy'
    )

    return model
