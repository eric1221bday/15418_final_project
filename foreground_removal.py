from skimage.io import imread_collection, imshow, imsave, use_plugin
from scipy.spatial.distance import cdist
from skimage.util import img_as_float
from skimage.transform import rescale
from scipy.signal import correlate2d
from matplotlib import pyplot as plt
from typing import List
import numpy as np


def energy_min_compose(imgs: List[np.ndarray]):
    height, width, channels = imgs[0].shape
    composite = np.zeros(imgs[0].shape)

    imgs_stack = np.stack(imgs, axis=3)
    imgs_avg = np.average(imgs_stack, axis=3)

    diffs = np.sum(np.linalg.norm((imgs_stack.T - imgs_avg.T).T, axis=2), axis=2)
    consensus_threshold = 0.1

    consensus = diffs < consensus_threshold
    ii, jj = np.meshgrid(range(height), range(width), sparse=False, indexing='ij')

    consensus3 = np.stack([consensus, consensus, consensus], axis=2)
    composite[consensus3] = imgs_avg[consensus3]
    # imsave('cut_consensus.png', composite)
    num_dif = np.sum(np.sum(~consensus))

    while num_dif > 0:
        have_neighbors = correlate2d(consensus, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode='same')
        boundary = (~consensus) & have_neighbors.astype(bool)
        sel_i = ii[boundary]
        sel_j = jj[boundary]
        indices = np.arange(len(sel_i))
        np.random.shuffle(indices)

        for n in indices:
            i = sel_i[n]
            j = sel_j[n]
            neighbors = []

            if i > 0 and consensus[i - 1, j]:
                neighbors.append(imgs_avg[i - 1, j, :])
            if i < (height - 1) and consensus[i + 1, j]:
                neighbors.append(imgs_avg[i + 1, j, :])
            if j > 0 and consensus[i, j - 1]:
                neighbors.append(imgs_avg[i, j - 1, :])
            if j < (width - 1) and consensus[i, j + 1]:
                neighbors.append(imgs_avg[i, j + 1, :])
            if i > 0 and j > 0 and consensus[i - 1, j - 1]:
                neighbors.append(imgs_avg[i - 1, j - 1])
            if i > 0 and j < (width - 1) and consensus[i - 1, j + 1]:
                neighbors.append(imgs_avg[i - 1, j + 1])
            if i < (height - 1) and j > 0 and consensus[i + 1, j - 1]:
                neighbors.append(imgs_avg[i + 1, j - 1])
            if i < (height - 1) and j < (width - 1) and consensus[i + 1, j + 1]:
                neighbors.append(imgs_avg[i + 1, j + 1])

            neighbors = np.stack(neighbors)
            distances_euclid = np.sum(cdist(neighbors, imgs_stack[i, j, :, :].T, 'euclidean'), axis=0)
            distances_cosine = np.sum(cdist(neighbors, imgs_stack[i, j, :, :].T, 'cosine'), axis=0)
            distances = distances_euclid + distances_cosine

            composite[i, j, :] = imgs_stack[i, j, :, np.argmin(distances)]
            consensus[i, j] = True
            num_dif -= 1

        # consensus = consensus | boundary
        print(num_dif)
        # imsave('cut_{}.png'.format(num_dif), composite)

    return composite


def median_stack(imgs: List):
    imgs_r = np.stack([img[:, :, 0] for img in imgs], axis=0)
    imgs_g = np.stack([img[:, :, 1] for img in imgs], axis=0)
    imgs_b = np.stack([img[:, :, 2] for img in imgs], axis=0)
    composite = np.zeros(imgs[0].shape)

    composite[:, :, 0] = np.median(imgs_r, axis=0)
    composite[:, :, 1] = np.median(imgs_g, axis=0)
    composite[:, :, 2] = np.median(imgs_b, axis=0)

    return composite


def main():
    use_plugin('matplotlib')

    # collection = imread_collection('shibuya/*.png')
    # imgs = [img_as_float(img[189:616, 635:1307]) for img in collection[:5]]

    # collection = imread_collection('photos/DJI/cut/*.JPG')
    # imgs = [img_as_float(collection[i]) for i in range(4, 8)]

    # for some reason this reads duplicate images
    # collection = imread_collection('photos/DJI/cut/original/*.JPG')
    # imgs = [img_as_float(collection[i]) for i in (8, 10, 12, 14)]

    # collection = imread_collection('photos/tripod/hammerschlag/*.jpg')
    # imgs = [rescale(img_as_float(collection[i]), 0.5) for i in range(3)]

    # collection = imread_collection('photos/tripod/doherty/*.jpg')
    # imgs = [rescale(img_as_float(collection[i]), 0.5) for i in range(4)]

    # collection = imread_collection('photos/tripod/cohon/*.jpg')
    # imgs = [rescale(img_as_float(img), 0.3) for img in collection]

    collection = imread_collection('photos/tripod/hunt/*.jpg')
    imgs = [rescale(img_as_float(collection[i]), 0.3) for i in range(4)]

    composite_median = median_stack(imgs)
    imsave('hunt_median.png', composite_median)
    composite_en_min = energy_min_compose(imgs)
    imsave('hunt_en_min_euclid+cosine.png', composite_en_min)

    # imshow(composite)
    # plt.show()


if __name__ == '__main__':
    main()
