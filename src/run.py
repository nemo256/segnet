import glob
import os
import tensorflow as tf

# custom imports
import model, data


def train(model_name='binary_crossentropy'):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    # loading train dataset and test datasets
    imgs, mask, edge = data.load_data(train_img_files)
    img_chips, mask_chips, edge_chips = data.test_chips(
        imgs,
        mask,
        edge=edge
    )

    # converting train and test datasets to tensorflow datasets
    train_dataset = tf.data.Dataset.from_generator(
        data.train_generator(imgs, mask, edge),
        (tf.float64, ((tf.float64), (tf.float64))),
        ((188, 188, 3), ((100, 100, 1), (100, 100, 1)))
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (img_chips, (mask_chips, edge_chips))
    )

    # initializing the segnet model
    model = model.segnet()

    # selecting custom adam optimizer
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss='binary_crossentropy',
        loss_weights=[0.1, 0.9],
        optimizer=optimizer,
        metrics='accuracy'
    )

    # fitting the model
    model.fit(
        train_dataset.batch(8),
        epochs=1,
        steps_per_epoch=125,
        validation_data=test_dataset.batch(8),
        max_queue_size=16,
        use_multiprocessing=False,
        workers=8,
        verbose=1,
        callbacks=get_callbacks(model_name)
    )


# main program
if __name__ == '__main__':
    train('binary_crossentropy')
