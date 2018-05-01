from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from scipy.spatial.distance import cdist
from scipy.stats import mode
from semantic import semantic_encoding
from skimage.io import imread_collection, imshow, imsave, use_plugin, imread
from skimage.transform import rescale
from skimage.util import img_as_float
from typing import List


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
    # imsave('cut_final.png', composite)
    return composite

def energy_min_compose_semantic(imgs: List[np.ndarray], encoding: List[np.ndarray]):
    height, width, channels = imgs[0].shape
    composite = np.zeros(imgs[0].shape)
    composite_encoding = np.zeros(encoding[0].shape)

    imgs_stack = np.stack(imgs, axis=3)
    imgs_avg = np.average(imgs_stack, axis=3)

    # find majority vote (random draw if multiple modes) for encoding
    def get_majority(a):
        return np.random.choice(mode(a)[0])
    encoding_stack = np.stack(encoding, axis=2)
    encoding_majority = np.apply_along_axis(get_majority, 2, encoding_stack)

    diffs = np.sum(np.linalg.norm((imgs_stack.T - imgs_avg.T).T, axis=2), axis=2)
    consensus_threshold = 0.1

    consensus = diffs < consensus_threshold
    ii, jj = np.meshgrid(range(height), range(width), sparse=False, indexing='ij')

    consensus3 = np.stack([consensus, consensus, consensus], axis=2)
    composite[consensus3] = imgs_avg[consensus3]
    composite_encoding[consensus] = encoding_majority[consensus]
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
                neighbors.append(composite_encoding[i - 1, j])
            if i < (height - 1) and consensus[i + 1, j]:
                neighbors.append(composite_encoding[i + 1, j])
            if j > 0 and consensus[i, j - 1]:
                neighbors.append(composite_encoding[i, j - 1])
            if j < (width - 1) and consensus[i, j + 1]:
                neighbors.append(composite_encoding[i, j + 1])
            if i > 0 and j > 0 and consensus[i - 1, j - 1]:
                neighbors.append(composite_encoding[i - 1, j - 1])
            if i > 0 and j < (width - 1) and consensus[i - 1, j + 1]:
                neighbors.append(composite_encoding[i - 1, j + 1])
            if i < (height - 1) and j > 0 and consensus[i + 1, j - 1]:
                neighbors.append(composite_encoding[i + 1, j - 1])
            if i < (height - 1) and j < (width - 1) and consensus[i + 1, j + 1]:
                neighbors.append(composite_encoding[i + 1, j + 1])

            # since distances are symmetric for one-hot encoded features, minimizing any distance is equivalent to finding the majority vote
            neighbors = np.array(neighbors)
            majority_neighbor_encoding = np.random.choice(mode(neighbors)[0])
            majority_voters = np.where(encoding_stack[i, j, :] == majority_neighbor_encoding)[0]
            majority_pixel_vals = [imgs_stack[i, j, :, t] for t in majority_voters]
            mean_majority_val = np.mean(majority_pixel_vals, axis=0)

            composite[i, j, :] = mean_majority_val
            composite_encoding[i, j] = majority_neighbor_encoding
            consensus[i, j] = True
            num_dif -= 1

        # consensus = consensus | boundary
        print(num_dif)
        # imsave('cut_{}.png'.format(num_dif), composite)
    return composite.astype("uint8")


def median_stack(imgs: List):
    imgs_r = np.stack([img[:, :, 0] for img in imgs], axis=0)
    imgs_g = np.stack([img[:, :, 1] for img in imgs], axis=0)
    imgs_b = np.stack([img[:, :, 2] for img in imgs], axis=0)
    composite = np.zeros(imgs[0].shape)

    composite[:, :, 0] = np.median(imgs_r, axis=0)
    composite[:, :, 1] = np.median(imgs_g, axis=0)
    composite[:, :, 2] = np.median(imgs_b, axis=0)
    return composite

def total_error(folder: str, truth_path: str):
    use_plugin('matplotlib')
    results = imread_collection(folder + '/*.png')
    imgs = [img_as_float(img) for img in results]
    truth = rescale(imread(truth_path), 0.24)
    print(truth.shape)
    height, width, channels = truth.shape

    for img in imgs:
        print(np.sum(np.linalg.norm(np.reshape(truth, [width*height, channels]) - np.reshape(img, [width*height, channels]), axis=1)) / (width*height))

def main():
    # use_plugin('matplotlib')

    # collection = imread_collection('expt/forbes/*.jpg')
    # imgs = [rescale(img_as_float(collection[i]), 0.24) for i in range(3)]

    # sem_imgs = [semantic_encoding("expt/forbes/%d.jpg"%i) for i in range(1, 4)]
    # sem_input_images = [i[0] for i in sem_imgs]
    # sem_input_encoding = [i[1] for i in sem_imgs]

    # composite_median = median_stack(imgs)
    # imsave('forbes_median.png', composite_median)
    # # composite_en_min = energy_min_compose(imgs)
    # # imsave('hunt_en_min_euclid+cosine.png', composite_en_min)
    # composite_en_min_semantic = energy_min_compose_semantic(sem_input_images, sem_input_encoding)
    # imsave('forbes_en_min_semantic.png', composite_en_min_semantic)
    # # imshow(composite)
    # # plt.show()

    total_error('expt/ham_result', 'expt/groundtruth/ham.jpg')


if __name__ == '__main__':
    main()
