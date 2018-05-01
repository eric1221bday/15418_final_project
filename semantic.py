# use deep network for semantic segmentation - label each pixel
import gluoncv
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import image, ndarray
from mxnet.gluon.data.vision import transforms
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.util import img_as_float
from scipy.signal import correlate2d
# from scipy.misc import imresize, imread
from typing import List

def semantic_encoding(filename):
    print("filename: " + filename)
    # using cpu
    ctx = mx.cpu(0)

    # read in images
    # img = image.imread(filename) # gives a ndarray directly
    # img = image.resize_short(img, 255)
    # original_img = imresize(imread(filename), 24)
    original_img = rescale(imread(filename), 0.24, preserve_range=True)
    img = ndarray.array(original_img, dtype="uint8")
    print(img.shape)

    # normalize the image using dataset mean
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    img = transform_fn(img)
    img = img.expand_dims(0).as_in_context(ctx)

    # get pre-trained model
    model = gluoncv.model_zoo.get_model('fcn_resnet101_voc', pretrained=True)
    # make prediction using single scale
    output = model.evaluate(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    # numpy.set_printoptions(threshold=numpy.nan)
    print("shape of semantic encoding: " + str(predict.shape))
    # show_color_pallete(predict)  # Add color pallete for visualization
    print(original_img.shape)
    print(np.unique(predict))
    return (original_img.astype('uint8'), predict)

def show_color_pallete(predict):
    from gluoncv.utils.viz import get_color_pallete
    import matplotlib.image as mpimg
    mask = get_color_pallete(predict, 'pascal_voc')
    mask.save('output.png')

    # show the predicted mask
    mmask = mpimg.imread('output.png')
    plt.imshow(mmask)
    plt.show()
