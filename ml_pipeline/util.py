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
                out[out_idx, in_idx_start : in_idx_start + kernel_width] = 1.0

    return out
