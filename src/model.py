import tensorflow as tf


# custom convolution
def conv_bn(filters, kernel=3, activation='relu', padding='same', model):
    conv = tf.keras.layers.Conv2D(filters, kernel, padding)(model)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)


# building the segnet model
def segnet():
    inputs = tf.keras.layers.Input((188, 188, 1))

    # encoder
    filters = 64
    encoder1 = conv_bn(filters)(inputs)
    encoder1 = conv_bn(filters)(encoder1)
    pool1, mask1 = tf.keras.layers.MaxPooling2D(strides=2, padding='same')(encoder1)

    filters *= 2
    encoder2 = conv_bn(filters)(pool1)
    encoder2 = conv_bn(filters)(encoder2)
    pool2, mask2 = tf.keras.layers.MaxPooling2D(strides=2, padding='same')(encoder2)

    filters *= 2
    encoder3 = conv_bn(filters)(pool2)
    encoder3 = conv_bn(filters)(encoder3)
    encoder3 = conv_bn(filters)(encoder3)
    pool3, mask3 = tf.keras.layers.MaxPooling2D(strides=2, padding='same')(encoder3)

    filters *= 2
    encoder4 = conv_bn(filters)(pool3)
    encoder4 = conv_bn(filters)(encoder4)
    encoder4 = conv_bn(filters)(encoder4)
    pool4, mask4 = tf.keras.layers.MaxPooling2D(strides=2, padding='same')(encoder4)

    encoder5 = conv_bn(filters)(pool4)
    encoder5 = conv_bn(filters)(encoder5)
    encoder5 = conv_bn(filters)(encoder5)
    pool5, mask5 = tf.keras.layers.MaxPooling2D(strides=2, padding='same')(encoder5)

    # decoder
    unpool1 = tf.keras.layers.MaxUnpooling2D()(pool5, mask5)
    decoder1 = conv_bn(filters)(unpool1)
    decoder1 = conv_bn(filters)(decoder1)
    decoder1 = conv_bn(filters)(decoder1)

    unpool2 = tf.keras.layers.MaxUnPooling2D()(decoder1, mask4)
    decoder2 = conv_bn(filters)(unpool2)
    decoder2 = conv_bn(filters)(decoder2)
    decoder2 = conv_bn(filters/2)(decoder2)

    filters /= 2
    unpool3 = tf.keras.layers.MaxUnPooling2D()(decoder2, mask3)
    decoder3 = conv_bn(filters)(unpool3)
    decoder3 = conv_bn(filters)(decoder3)
    decoder3 = conv_bn(filters/2)(decoder3)

    filters /= 2
    unpool4 = tf.keras.layers.MaxUnPooling2D()(decoder3, mask2)
    decoder4 = conv_bn(filters)(unpool4)
    decoder4 = conv_bn(filters/2)(decoder4)

    filters /= 2
    unpool5 = tf.keras.layers.MaxUnPooling2D()(decoder4, mask1)
    decoder5 = conv_bn(filters)(unpool5)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder5)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder5)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    return model
