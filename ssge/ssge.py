"""
"""
import numpy as np
import scipy


class SSGE:

    def rbf_kernel(self, x1, x2, kernel_width):
        return np.exp(-np.sum((x1 - x2) ** 2, axis=1) /
                      (2 * kernel_width ** 2))

    def gram_matrix(self, x1, x2, kernel_width):
        # x_row =
        # x_col =
        # return self.rbf_kernel(x_row, x_col, kernel_width)
        pass

    def gram_grad(self, x1, x2, kernel_width):
        pass

    def nystrom(self, samples):
        M = np.shape(samples)

        target = np.sqrt(M)
        pass

    def compute_gradients(self):
        pass


a = np.array([[12., -22., -20., -19., -3.],
              [-23., 21., -17., -11., -1.],
              [-4., -5., 16., -9., -14.],
              [-10., -6., -18., 15., -8.],
              [-25., -2., -13., -7., 24.]])
