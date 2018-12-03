from numba import jit
import numpy as np


def forward(x,in_channels, out_channels, height, width, stride=1, padding=0, init_scale=1e-2):
    paramsw = init_scale * np.random.randn(out_channels, in_channels, height, width):
    out = None
    N, C, H, W = x.shape
    F, _, HH, WW = self.params['w']['param'].shape
    # assert (W + 2 * self.padding - WW) % self.stride == 0, 'width does not work'
    # assert (H + 2 * self.padding - HH) % self.stride == 0, 'height does not work'
    x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
    H_new = 1 + (H + 2 * self.padding - HH) // self.stride
    W_new = 1 + (W + 2 * self.padding - WW) // self.stride
    s = self.stride
    out = np.zeros((N, F, H_new, W_new))

    for n in range(N):
        for f in range(F):
            for j in range(H_new):
                for k in range(W_new):
                    out[n, f, j, k] = np.sum(
                        x_padded[n, :, j * s:HH + j * s, k * s:WW + k * s] * self.params['w']['param'][f]) + \
                                      self.params['b']['param'][0, f]
    return out