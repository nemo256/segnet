import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# custom imports
import data
from model import segnet, get_callbacks


def generate_train_dataset(img_files):
    img, mask, edge = data.load_data(img_files)

    def train_gen():
        return data.train_generator(img, mask,
                                    edge=edge,
                                    padding=100,
                                    input_size=224,
                                    output_size=224)

    return tf.data.Dataset.from_generator(
        train_gen,
        (tf.float64, ((tf.float64), (tf.float64))),
        ((224, 224, 3), ((224, 224, 1), (224, 224, 1)))
    )


def generate_test_dataset(img_files):
    img, mask, edge = data.load_data(img_files)

    img_chips, mask_chips, edge_chips = data.test_chips(img, mask,
                                                        edge=edge,
                                                        padding=100,
                                                        input_size=224,
                                                        output_size=224)

    return tf.data.Dataset.from_tensor_slices(
        (img_chips, (mask_chips, edge_chips))
    )


def train(model_name='binary_crossentropy'):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    # loading train dataset and test datasets
    train_dataset = generate_train_dataset(train_img_files)
    test_dataset = generate_test_dataset(test_img_files)

    # initializing the segnet model
    model = segnet()

    # fitting the model
    model.fit(
        train_dataset.batch(8),
        validation_data=test_dataset.batch(8),
        epochs=800,
        steps_per_epoch=125,
        max_queue_size=16,
        use_multiprocessing=False,
        workers=8,
        verbose=1,
        callbacks=get_callbacks(model_name)
    )


# extract number of image chips for an image
def get_sizes(img,
              offset=212,
              input=224,
              output=224):
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


# reshape numpy arrays
def reshape(img,
            size_x,
            size_y,
            type='input'):
    if type == 'input':
        return img.reshape(size_x, size_y, 224, 224, 1)
    elif type == 'output':
        return img.reshape(size_x, size_y, 224, 224, 1)
    else:
        print(f'Invalid type: {type} (input, output)')


# concatenate images
def concat(imgs):
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:,:]])


# predict (segment) image and save a sample output
def predict(img='Im037_0.jpg',
            model_name='binary_crossentropy'):
    image = glob.glob(f'data/test/{img}')

    # initialize segnet
    model = segnet()

    # Check for existing weights
    if not os.path.exists(f'models/{model_name}.h5'):
        train(model_name)

    # load best weights
    model.load_weights(f'models/{model_name}.h5')

    # load test data
    img, mask, edge = data.load_data(image, padding=200)
    img_chips, mask_chips, edge_chips = data.test_chips(
        img,
        mask,
        edge=edge,
        padding=100,
        input_size=224,
        output_size=224
    )

    # segment all image chips
    output = model.predict(img_chips)
    new_mask_chips = np.array(output[0])
    new_edge_chips = np.array(output[1])

    # reshape chips arrays to be concatenated
    new_mask_chips = reshape(new_mask_chips, get_sizes(img)[0][0], get_sizes(img)[0][1], 'output')
    new_edge_chips = reshape(new_edge_chips, get_sizes(img)[0][0], get_sizes(img)[0][1], 'output')

    # concatenate chips into a single image (mask and edge)
    new_mask = concat(new_mask_chips)
    new_edge = concat(new_edge_chips)

    # save predicted mask and edge
    plt.imsave('output/mask.png', new_mask)
    plt.imsave('output/edge.png', new_edge)
    plt.imsave('output/edge-mask.png', new_mask - new_edge)

    # organize results into one figure
    fig = plt.figure(figsize=(25, 12), dpi=80)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title('Test image')
    ax.imshow(np.array(img)[0,:,:,:])
    ax = fig.add_subplot(2, 3, 2)
    ax.set_title('Test mask')
    ax.imshow(np.array(mask)[0,:,:])
    ax = fig.add_subplot(2, 3, 3)
    ax.set_title('Test edge')
    ax.imshow(np.array(edge)[0,:,:])
    ax = fig.add_subplot(2, 3, 5)
    ax.set_title('Predicted mask')
    ax.imshow(new_mask)
    ax = fig.add_subplot(2, 3, 6)
    ax.set_title('Predicted edge')
    ax.imshow(new_edge)

    # save the figure as a sample output
    plt.savefig('sample.png')


# evaluate model accuracies (mask accuracy and edge accuracy)
def evaluate(model_name='binary_crossentropy'):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    # initialize segnet
    model = segnet()

    # check for existing weights
    if not os.path.exists(f'models/{model_name}.h5'):
        train(model_name)

    # load best weights
    model.load_weights(f'models/{model_name}.h5')

    # load test data
    imgs, mask, edge = data.load_data(test_img_files)
    img_chips, mask_chips, edge_chips = data.test_chips(
        imgs,
        mask,
        edge=edge,
        padding=100,
        input_size=224,
        output_size=224
    )

    # print the evaluated accuracies
    print(model.evaluate(img_chips, (mask_chips, edge_chips)))


# threshold image using otsu's threshold
def threshold(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    image = cv2.imread(f'output/{img}')

    # convert to grayscale and apply otsu's thresholding
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)

    # save the resulting thresholded image
    plt.imsave(f'output/threshold_{img}', image)


# count how many cells from the predicted edges
def count_circles(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    img = cv2.imread(f'output/{img}')

    # convert to grayscale and apply Circle Hough Transform (CHT)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=33, maxRadius=55, minRadius=28, param1=30, param2=20)
    output = img.copy()

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # save the output image
        plt.imsave('output/count_circles.png', np.hstack([img, output]))

    print(f'Real count: {len(data.make_polygon_lists(["data/test/Im037_0.json"])[0])}')
    print(f'Predicted count: {len(circles)}')


# main program
if __name__ == '__main__':
    train('binary_crossentropy')
    # evaluate(model_name='binary_crossentropy')
    # predict(model_name='binary_crossentropy', img='Im037_0.jpg')
    # threshold(img='mask.png')
    # threshold(img='edge.png')
    # threshold(img='edge-mask.png')
    # count_circles(img='edge.png')
