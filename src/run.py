import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# custom imports
import data
from model import segnet, get_callbacks


# global variables
input_shape     = (128, 128, 3)
output_shape    = (128, 128, 1)
padding         = [200, 100]


def generate_train_dataset(img_files):
    img, mask, edge = data.load_data(img_files)

    def train_gen():
        return data.train_generator(img, mask,
                                    edge=edge,
                                    padding=padding[0],
                                    input_size=input_shape[0],
                                    output_size=output_shape[0])

    return tf.data.Dataset.from_generator(
        train_gen,
        (tf.float64, ((tf.float64), (tf.float64))),
        (input_shape, (output_shape, output_shape))
    )


def generate_test_dataset(img_files):
    img, mask, edge = data.load_data(img_files)

    img_chips, mask_chips, edge_chips = data.test_chips(
        img,
        mask,
        edge=edge,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0]
    )

    return tf.data.Dataset.from_tensor_slices(
        (img_chips, (mask_chips, edge_chips))
    )


def train(model_name='mse'):
    train_img_files = glob.glob('data/train/*.jpg')
    test_img_files = glob.glob('data/test/*.jpg')

    # loading train dataset and test datasets
    train_dataset = generate_train_dataset(train_img_files)
    test_dataset = generate_test_dataset(test_img_files)

    # initializing the segnet model
    model = segnet()

    model.load_weights(f'models/{model_name}.h5')

    # fitting the model
    history = model.fit(
        train_dataset.batch(8),
        validation_data=test_dataset.batch(8),
        epochs=500,
        steps_per_epoch=125,
        max_queue_size=16,
        use_multiprocessing=False,
        workers=8,
        verbose=1,
        callbacks=get_callbacks(model_name)
    )

    # save the history
    np.save(f'models/{model_name}_history.npy', history.history)


def normalize(img):
    return np.array((img - np.min(img)) / (np.max(img) - np.min(img)))


# extract number of image chips for an image
def get_sizes(img,
              padding=padding[1],
              input=input_shape[0],
              output=output_shape[0]):
    offset = padding + (output / 2)
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


# reshape numpy arrays
def reshape(img,
            size_x,
            size_y):
    return img.reshape(size_x, size_y, output_shape[0], output_shape[0], 1)


# concatenate images
def concat(imgs):
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:]])


# denoise an image
def denoise(img):
    # read the image
    img = cv2.imread(img)
    # return the denoised image
    return cv2.fastNlMeansDenoising(img, 23, 23, 7, 21)


# predict (segment) image and save a sample output
def predict(img='Im037_0.jpg',
            model_name='mse'):
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
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0]
    )

    # segment all image chips
    output = model.predict(img_chips)
    new_mask_chips = np.array(output[0])
    new_edge_chips = np.array(output[1])

    # get image dimensions
    dimensions = [get_sizes(img)[0][0], get_sizes(img)[0][1]]

    # reshape chips arrays to be concatenated
    new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])
    new_edge_chips = reshape(new_edge_chips, dimensions[0], dimensions[1])

    new_mask_chips = np.squeeze(new_mask_chips)
    new_edge_chips = np.squeeze(new_edge_chips)

    # concatenate chips into a single image (mask and edge)
    new_mask = concat(new_mask_chips)
    new_edge = concat(new_edge_chips)

    # save predicted mask and edge
    plt.imsave('output/mask.png', new_mask, cmap='gray')
    plt.imsave('output/edge.png', new_edge, cmap='gray')
    plt.imsave('output/edge_mask.png', new_mask - new_edge, cmap='gray')

    # denoise all the output images
    new_mask = denoise('output/mask.png')
    new_edge = denoise('output/edge.png')
    edge_mask = denoise('output/edge_mask.png')

    # save predicted mask and edge after denoising
    plt.imsave('output/mask.png', new_mask, cmap='gray')
    plt.imsave('output/edge.png', new_edge, cmap='gray')
    plt.imsave('output/edge_mask.png', edge_mask, cmap='gray')

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
def evaluate(model_name='mse'):
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
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0]
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
    plt.imsave(f'output/threshold_{img}', image, cmap='gray')


# count how many cells from the predicted edges
def hough_transform(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image in grayscale mode
    image = cv2.imread(f'output/{img}')
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply hough circles
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
        plt.imsave('output/hough_transform.png',
                   np.hstack([img, output]))

    # show the hough_transform results
    print('Hough transform:')
    print(f'Real count: {len(data.make_polygon_lists(["data/test/Im037_0.json"])[0])}')
    print(f'Predicted count: {len(circles)}')


# count how many cells from the predicted edges
def component_labeling(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'output/{img}')
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)
    
    # map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    output = cv2.merge([label_hue, blank_ch, blank_ch])

    # converting cvt to BGR
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    # set bg label to black
    output[label_hue==0] = 0
    
    # saving image after Component Labeling
    plt.imsave('output/component_labeling.png',
               np.hstack([image, output]))

    # show number of labels detected
    print('Connected component labeling:')
    print(f'Real count: {len(data.make_polygon_lists(["data/test/Im037_0.json"])[0])}')
    print(f'Predicted count: {num_labels}')


# get a minimal of each cell to help with the counting
def distance_transform(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'output/{img}')
    # convert to numpy array
    img = np.asarray(image)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # transorm rgb channels
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=0)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=0)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=0)
    
    # merge the transformed channels back to an image
    img = cv2.merge((b, g, r))
    img = normalize(img)

    # saving image after Component Labeling
    plt.imsave('output/distance_transform.png', img)


# main program
if __name__ == '__main__':
    train('mse_unsupervised')
    # evaluate(model_name='mse')
    # predict(model_name='mse')
    # threshold(img='mask.png')
    # threshold(img='edge.png')
    # threshold(img='edge_mask.png')
    # distance_transform(img='threshold_edge_mask.png')
    # hough_transform(img='edge.png')
    # component_labeling(img='edge_mask.png')
