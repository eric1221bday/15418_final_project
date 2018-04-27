from skimage.io import imread_collection, imshow, imsave, use_plugin
from skimage.util import img_as_float
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
    imsave('cut_consensus.png', composite)
    num_dif = np.sum(np.sum(~consensus))

    while num_dif > 0:
        have_neighbors = correlate2d(consensus, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode='same')
        boundary = (~consensus) & have_neighbors
        boundary_im = composite
        boundary_im[np.stack([boundary, boundary, boundary], axis=2)] = 1
        imsave('cut_boundaries.png', boundary_im)
        sel_i = ii[boundary]
        sel_j = jj[boundary]

        for n in range(len(sel_i)):
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

            neighbors = np.stack(neighbors)
            distances = np.apply_along_axis(lambda x: np.sum(np.linalg.norm(x - neighbors, axis=1)), 1,
                                            imgs_stack[i, j, :, :].T)
            composite[i, j, :] = imgs_stack[i, j, :, np.argmin(distances)]
            consensus[i, j] = True
            num_dif -= 1

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

    collection = imread_collection('photos/DJI/cut/*.JPG')
    imgs = [img_as_float(collection[i]) for i in range(4, 8)]

    # for some reason this reads duplicate images
    # collection = imread_collection('photos/DJI/cut/original/*.JPG')
    # imgs = [img_as_float(collection[i]) for i in (8, 10, 12, 14)]

    # composite = median_stack(imgs)
    composite = energy_min_compose(imgs)

    # imshow(composite)
    # plt.show()


if __name__ == '__main__':
    main()
