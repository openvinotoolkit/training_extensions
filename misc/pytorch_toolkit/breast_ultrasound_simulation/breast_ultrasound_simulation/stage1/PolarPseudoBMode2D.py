# ~~~ Program: Ultrasound Pseudo B-Mode 2D Stage 0 Output Creation ~~~ #
# ~~~ Author: Raj Krishan Ghosh ~~~ #

import numpy as np
import scipy as sp
from scipy.signal import hilbert
import random

random.seed(0)
np.random.seed(0)


def grayscale_normalize_2d(input_tensor):
    output_tensor = input_tensor.astype(float)
    min_val = np.min(output_tensor)
    max_val = np.max(output_tensor)
    if max_val != min_val:
        output_tensor = (output_tensor - min_val) / (max_val - min_val)

    return output_tensor


def echo_model_2d(input_tensor, low=0.01, mid=0.5, high=0.02):
    output_tensor = np.ones_like(input_tensor)
    output_tensor = output_tensor * mid
    output_tensor[input_tensor < 0.1] = low
    output_tensor[input_tensor > 0.9] = high

    return output_tensor


def imfilter_2d(input_tensor, kernel_array):
    input_height = len(input_tensor)
    input_width = len(input_tensor[0])
    kernel_height = len(kernel_array)
    kernel_width = len(kernel_array[0])
    kernel_h_mid = int(kernel_height / 2.0)
    kernel_w_mid = int(kernel_width / 2.0)

    temp_tensor = np.zeros((len(input_tensor) + kernel_height - 1, len(input_tensor[0]) + kernel_width - 1),
                           dtype=input_tensor.dtype)
    temp_tensor[kernel_h_mid:(kernel_h_mid + input_height), kernel_w_mid:(kernel_w_mid + input_width)] = input_tensor

    output_tensor = np.zeros(shape=(np.shape(input_tensor)[0], np.shape(input_tensor)[1]), dtype=input_tensor.dtype)

    for i in range(input_height):
        for j in range(input_width):
            temp_array = temp_tensor[i:(i + kernel_height), j:(j + kernel_width)]

            output_tensor[i, j] = np.sum(np.multiply(kernel_array, temp_array))

    return output_tensor


def hilbert_transform_2d(input_tensor):
    fourier = sp.fft(input_tensor)
    for i in range(len(fourier)):
        for j in range(len(fourier[0])):
            if fourier[i, j] < 0.0:
                fourier[i, j] = 0.0
            else:
                fourier[i, j] = 2 * fourier[i, j]
    inv_fourier = sp.ifft(fourier)

    output_tensor = input_tensor + (1.0j * inv_fourier)

    return output_tensor


def pseudo_b_mode_2d(input_tensor, f0=10e6, c=1540.0, sigma_x=2.0, sigma_y=1.5, speckle_variance=0.01):
    k0 = 2.0 * np.pi * f0 / c
    n_rows, n_columns = np.size(input_tensor, 0), np.size(input_tensor, 1)
    g = np.random.uniform(size=(n_rows, n_columns)).astype(np.double)
    mean_g = np.mean(g)
    g = (g - mean_g) * speckle_variance
    t = (input_tensor.astype(np.double)) * g

    step_size = 1.0
    x = np.arange(start=-10 * sigma_x, stop=(10 * sigma_x) + (step_size / 2.0), step=step_size)
    y = np.arange(start=-10 * sigma_y, stop=(10 * sigma_y) + (step_size / 2.0), step=step_size)

    hx = np.array((np.sin(k0 * x)) * (np.exp((-(x ** 2.0)) / (2 * (sigma_x ** 2.0))))).reshape((-1, 1))
    hy = np.array(np.exp((-(y ** 2.0)) / (2 * (sigma_y ** 2.0)))).reshape((1, -1))

    v = sp.ndimage.correlate(t, hx, mode='constant', cval=0.0)
    v = sp.ndimage.correlate(v, hy, mode='constant', cval=0.0)
    v_cap = np.transpose(hilbert(np.transpose(v)))
    v_a = v + (1.0j * v_cap)
    rf_envelop = np.abs(v_a)

    output_tensor = (255.0 * grayscale_normalize_2d(np.log10(rf_envelop + np.finfo(np.float64).eps))).astype(np.uint8)

    return output_tensor


def sub2ind_2d(size_of_tensor, row_ind_array, col_ind_array):
    output_linear_indices = []
    for i, r_elem in enumerate(row_ind_array):
        linear_index = ((size_of_tensor[0] * col_ind_array[i]) + r_elem)
        output_linear_indices.append(linear_index)

    return np.array(output_linear_indices, dtype=int)

def x_exp_2d(x, a):
    return np.exp(-a * x)


def attenuation_weighting_2d(input_tensor, alpha):
    dw = np.array([(i + 1) for i in range(np.size(input_tensor, 0))], dtype=np.double)
    dw = dw / np.size(input_tensor, 0)
    dw = np.repeat(dw, np.size(input_tensor, 1))
    dw = np.reshape(dw, (np.size(input_tensor, 0), np.size(input_tensor, 1)))
    dw = (dw - np.min(dw)) / (np.max(dw) - np.min(dw))
    w = 1.0 - x_exp_2d(dw, alpha)

    return w


def padarray_2d(input_tensor, pad):
    output_tensor = np.zeros(shape=(np.shape(input_tensor)[0] + (2 * pad[0]),
                                    np.shape(input_tensor)[1] + (2 * pad[1])),
                             dtype=input_tensor.dtype)
    output_tensor[pad[0]: pad[0] + np.shape(input_tensor)[0], pad[1]: pad[1] + np.shape(input_tensor)[1]] = input_tensor

    return output_tensor


def find_2d(input_tensor):
    input_tensor = np.array(input_tensor)
    if input_tensor.ndim == 2:
        input_vector = np.transpose(input_tensor).reshape(np.shape(input_tensor)[0] * np.shape(input_tensor)[1])
    elif input_tensor.ndim == 1:
        input_vector = input_tensor
    else:
        print(input_tensor.ndim)

    output_tensor = np.argwhere(input_vector != 0).reshape(-1) + 1

    return output_tensor


def confidence_laplacian_2d(p_capital, a_capital, beta, gamma):
    m, _ = np.shape(p_capital)
    p = find_2d(p_capital)

    p_capital_flattened = np.transpose(p_capital).reshape(-1)
    a_capital_flattened = np.transpose(a_capital).reshape(-1)
    p_flattened = np.transpose(p).reshape(-1)

    i = [p_capital_flattened[x - 1] for x in p]  # index vector
    j = [p_capital_flattened[x - 1] for x in p]  # index vector

    s = np.zeros_like(p, dtype=np.double)  # Entries vector, initially for diagonal

    # vertical edges
    for k in [-1, 1]:
        q_capital = np.take(p_capital_flattened, p + k - 1)
        q = find_2d(q_capital)
        ii = np.take(p_capital_flattened, np.take(p_flattened, q - 1) - 1)
        i = np.concatenate((i, ii))
        jj = np.take(q_capital, q - 1)
        j = np.concatenate((j, jj))
        w_capital = np.abs(np.take(a_capital_flattened,
                                   np.take(p_flattened, ii - 1) - 1) - \
                                    np.take(a_capital_flattened,
                                            np.take(p_flattened, jj - 1) - 1))  # Intensity derived weight
        s = np.concatenate((s, w_capital))

    vl = np.size(s)

    # diagonal edges
    for k in [m-1, m+1, -m-1, -m+1]:
        q_capital = np.take(p_capital_flattened, p + k - 1)
        q = find_2d(q_capital)
        ii = np.take(p_capital_flattened, np.take(p_flattened, q - 1) - 1)
        i = np.concatenate((i, ii))
        jj = np.take(q_capital, q - 1)
        j = np.concatenate((j, jj))
        w_capital = np.abs(np.take(a_capital_flattened,
                                   np.take(p_flattened, ii - 1) - 1) - \
                                    np.take(a_capital_flattened,
                                            np.take(p_flattened, jj - 1) - 1))  # Intensity derived weight
        s = np.concatenate((s, w_capital))

    # horizontal edges
    for k in [m, -m]:
        q_capital = np.take(p_capital_flattened, p + k - 1)
        q = find_2d(q_capital)
        ii = np.take(p_capital_flattened, np.take(p_flattened, q - 1) - 1)
        i = np.concatenate((i, ii))
        jj = np.take(q_capital, q - 1)
        j = np.concatenate((j, jj))
        w_capital = np.abs(np.take(a_capital_flattened,
                                   np.take(p_flattened, ii - 1) - 1) - \
                                    np.take(a_capital_flattened,
                                            np.take(p_flattened, jj - 1) - 1))  # Intensity derived weight
        s = np.concatenate((s, w_capital))

    # normalize weights
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + np.finfo(np.float64).eps)

    # horizontal penalty
    s[vl:] = s[vl:] + gamma

    # normalize differences
    s = (s - np.min(s)) / (np.max(s) - np.min(s) + np.finfo(np.float64).eps)

    # gaussian weighting function
    epsilon = 10e-6
    s = -((np.exp((-beta) * s)) + epsilon)

    # create laplacian, diagonal missing
    i = i - 1
    j = j - 1
    capital_l = sp.sparse.csr_matrix((s, (i, j)))

    # reset diagonal weights to zero for summing up the weighted edge degree in the next step
    capital_l.setdiag(0)

    # weighted edge degree
    capital_d = np.absolute(np.sum(capital_l, axis=1)).getH()
    capital_d = np.asarray(capital_d).reshape(-1)

    # finalize laplacian by completing the diagonal
    capital_l.setdiag(capital_d)

    return capital_l


def delete_from_csr(mat, row_indices=None, col_indices=None):
    if row_indices is None:
        row_indices = []
    if col_indices is None:
        col_indices = []
    if not isinstance(mat, sp.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if rows and cols:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:, col_mask]
    if rows:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    if cols:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    return mat


def confidence_estimation_2d(input_tensor, seeds, labels, beta, gamma):
    # index matrix with boundary padding
    g = find_2d(np.ones_like(input_tensor))
    g = np.reshape(g, (np.shape(input_tensor)[1], np.shape(input_tensor)[0])).transpose()

    pad = 1

    g = padarray_2d(g, (pad, pad))
    b = padarray_2d(input_tensor, (pad, pad))

    # laplacian
    d = confidence_laplacian_2d(g, b, beta, gamma)

    # select marked columns from laplacian to create L_M and B^T
    b = d[:, seeds - 1]

    # select marked nodes to create B^T
    n = np.sum(np.reshape(g, -1) > 0)
    i_u = np.asarray(list(range(n)))
    np.put(i_u, seeds - 1, 0)
    i_u = find_2d(i_u)  # index of unmarked nodes
    b = b[i_u - 1, :]

    # remove marked nodes from laplacian by deleting rows and cols
    d = delete_from_csr(d, seeds - 1, seeds - 1)

    # adjust labels
    label_adjust = np.min(labels)
    labels = labels - label_adjust + 1  # labels > 0

    # find number of labels (K)
    labels_present = np.unique(labels)
    number_labels = len(labels_present)

    # define M matrix
    m = np.zeros((len(seeds), number_labels))

    for k in range(number_labels):
        m[:, k] = np.reshape(labels, -1) == labels_present[k]

    # right-hand side (-B^T*M)
    rhs = sp.sparse.csr_matrix(-b * m)

    # solve system
    if number_labels == 2:
        x = []
        x.append(sp.sparse.linalg.spsolve(d, rhs[:, 0]))
        x.append(1 - x[0])
        x = np.array(x)
    else:
        x = sp.sparse.linalg.spsolve(d, rhs)

    probabilities = np.zeros((n, number_labels))

    for k in range(number_labels):
        # Probabilities for unmarked nodes
        probabilities[i_u - 1, k] = x[k, :]
        # Max probability for marked node of each label
        probabilities[seeds[labels == (k + 1)] - 1, k] = 1.0

    probabilities = np.reshape(probabilities, (input_tensor.shape[1], input_tensor.shape[0], number_labels))
    probabilities = np.transpose(probabilities, (1, 0, 2))

    return probabilities


def confidence_map_2d(input_tensor, alpha=2.0, beta=90.0, gamma=0.05, b_mode=True):
    input_data = input_tensor.astype(np.double)
    input_data = (input_data - np.min(input_data)) / \
        (np.max(input_data) - np.min(input_data) + np.finfo(np.float64).eps)

    if not b_mode:
        input_data = np.abs(np.transpose(hilbert(np.transpose(input_data))))

    seeds = np.array([], dtype=int)
    labels = np.array([], dtype=int)

    sc = np.array(list(range(np.size(input_data, 1))), dtype=int)  # all elements

    # source elements
    sr_up = np.ones(len(sc), dtype=int)
    seed = sub2ind_2d([np.size(input_tensor, 0), np.size(input_tensor, 1)], sr_up, sc)
    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # label 1
    label = np.zeros(len(seed), dtype=int)
    label = label + 1
    labels = np.concatenate((labels, label))

    # sink elements
    sr_down = np.ones(len(sc), dtype=int) * len(input_data)
    seed = sub2ind_2d([np.size(input_tensor, 0), np.size(input_tensor, 1)], sr_down, sc)
    seed = np.unique(seed)
    seeds = np.concatenate((seeds, seed))

    # label 2
    label = np.zeros(len(seed), dtype=int)
    label = label + 2
    labels = np.concatenate((labels, label))

    # Attenuation with Beer - Lambert
    w = attenuation_weighting_2d(input_data, alpha)

    # apply weighting directly to image
    # same as applying it individually during the formation of the Laplacian
    input_data = input_data * w

    # find confidence values
    conf_map = confidence_estimation_2d(input_data, seeds, labels, beta, gamma)

    return conf_map[:, :, 0]


def generate_pseudo_b_mode_2d(input_tensor,
                              low=0.01,
                              mid=0.5,
                              high=0.02,
                              f0=10e6,
                              c=1540.0,
                              sigma_x=2.0,
                              sigma_y=1.5,
                              speckle_variance=0.01,
                              alpha=2.0,
                              beta=90.0,
                              gamma=0.05,
                              b_mode=True):
    # convert mask image (x1) into grayscale in the range of 0.0 to 1.0
    grayscale_normalised_tensor = grayscale_normalize_2d(input_tensor)

    # convert the grayscale mask image (x1) into echo model...
    # (0.01*((grayscale mask image))>0.9) + 0.02*((grayscale mask image)<0.1) + ...
    # 0.5*(((grayscale mask image)<0.9).*((grayscale mask image)>0.1))
    echo_model_tensor = echo_model_2d(grayscale_normalised_tensor, low, mid, high)

    # convert the echo model (x1) to pseudo b-mode version...
    # arguments: echo model, 20e6 (center frequency, f0), 1540 (speed of sound, c), 5 (sigma_x), ...
    # 2 (sigma_y), 0.01 (speckle_variance)
    pseudo_b_mode_tensor = pseudo_b_mode_2d(echo_model_tensor, f0, c, sigma_x, sigma_y, speckle_variance)

    # create confidence map (x2) taking pseudo b-mode version (x1), alpha (=2.0), beta (= 90.0) and gamma (= 0.06))
    confidence_map_tensor = confidence_map_2d(pseudo_b_mode_tensor, alpha, beta, gamma, b_mode)

    # create output (y) as y = (x1 converted into grayscale in range 0.0 to 1.0) * x2 (element-wise multiplication)
    output_tensor = (grayscale_normalize_2d(pseudo_b_mode_tensor) * confidence_map_tensor) * 255.0

    return output_tensor
