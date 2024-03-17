import numpy as np


def make_convolution_weight_mask(
    in_size: int, kernel_width: int, stride: int
) -> np.ndarray:
    out_size = (in_size - kernel_width) / stride + 1
    assert out_size.is_integer()
    out_size = int(out_size)

    out = np.zeros((out_size * out_size, in_size * in_size))

    for i in range(out_size):
        for j in range(out_size):
            out_idx = i * out_size + j
            x_start = i * stride
            y_start = j * stride

            for k in range(kernel_width):
                in_idx_start = (x_start + k) * in_size + y_start
                out[out_idx, in_idx_start : in_idx_start + kernel_width] = 1

    return out


def get_competition_matrix(v: np.ndarray) -> np.ndarray:
    N = len(v)
    col = v.reshape(N, 1)
    row = v.reshape(1, N)

    greater = col > row
    less = col < row

    result = np.zeros((N, N))
    result[greater] = 1
    result[less] = -1

    return result


def make_proximity_weight_mask(size: int, reach: int) -> np.ndarray:
    out = np.zeros((size * size, size * size))

    for i in range(size):
        for j in range(size):
            out_idx = i * size + j
            x_start = max(0, i - reach)
            x_end = min(i + reach + 1, size)
            y_start = max(0, j - reach)
            y_end = min(j + reach + 1, size)

            for x_idx in range(x_start, x_end):
                in_idx_start = x_idx * size + y_start
                in_idx_end = x_idx * size + y_end

                out[out_idx, in_idx_start:in_idx_end] = 1

    return out
