import numpy as np
from numba import njit, types as nt


@njit
def running_stat(xs: np.ndarray, n: int, M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, M4: np.ndarray):
    for x in xs:
        n1 = n
        n += 1
        delta = x - M1
        delta_n = delta / n
        delta_n2 = delta_n ** 2
        term1 = delta * delta_n * n1
        M1 += delta_n
        M4 += term1 * delta_n2 * (n*n - 3*n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1
    return n


def merge_stat(
    a_n: np.ndarray, a_M1: np.ndarray, a_M2: np.ndarray, a_M3: np.ndarray, a_M4: np.ndarray,
    b_n: np.ndarray, b_M1: np.ndarray, b_M2: np.ndarray, b_M3: np.ndarray, b_M4: np.ndarray):
    c_n = a_n + b_n

    delta = b_M1 - a_M1
    delta2 = delta ** 2
    delta3 = delta*delta2
    delta4 = delta2 ** 2

    c_M1 = (a_n*a_M1 + b_n*b_M1) / c_n

    c_M2 = a_M2 + b_M2 + delta2 * a_n * b_n / c_n

    c_M3 = a_M3 + b_M3 + delta3 * a_n * b_n * (a_n - b_n)/(c_n*c_n)
    c_M3 += 3.0*delta * (a_n*b_M2 - b_n*a_M2) / c_n

    c_M4 = a_M4 + b_M4 + delta4*a_n*b_n * (a_n*a_n - a_n*b_n + b_n*b_n) / (c_n*c_n*c_n)
    c_M4 += 6.0*delta2 * (a_n*a_n*b_M2 + b_n*b_n*a_M2)/(c_n*c_n) + 4.0*delta*(a_n*b_M3 - b_n*a_M3) / c_n

    return c_n, c_M1, c_M2, c_M3, c_M4


class RunningStat:
    """
    http://www.johndcook.com/blog/standard_deviation/
    https://www.johndcook.com/blog/skewness_kurtosis/
    """
    def __init__(self, shape=()):
        self._n = 0
        self._M1 = np.zeros(shape)
        self._M2 = np.zeros(shape)
        self._M3 = np.zeros(shape)
        self._M4 = np.zeros(shape)

    def push(self, x: np.ndarray):
        x = np.asarray(x).reshape(-1, *self._M1.shape)
        self._n = running_stat(x, self._n, self._M1, self._M2, self._M3, self._M4)

    def update(self, other: "RunningStat"):
        self._n, self._M1, self._M2, self._M3, self._M4 = merge_stat(
            self._n, self._M1, self._M2, self._M3, self._M4,
            other._n, other._M1, other._M2, other._M3, other._M4
        )

    def __repr__(self):
        return "RunningStat(n={}, mean_mean={}, mean_std={}, mean_skew={}, mean_kurt={})".format(
            self.n, np.mean(self.mean), np.mean(self.std), np.mean(self.skew), np.mean(self.kurt)
        )

    def copy(self):
        other = RunningStat()
        other._n = self._n
        other._M1 = self._M1
        other._M2 = self._M2
        other._M3 = self._M3
        other._M4 = self._M4
        return other

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M1

    @property
    def var(self):
        return self._M2 / (self._n - 1)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def skew(self):
        return self._n ** 0.5 * self._M3 / self._M2 ** 1.5

    @property
    def kurt(self):
        return self.n * self._M4 / self._M2 ** 2 - 3.0

    @property
    def shape(self):
        return self._M1.shape

if __name__ == "__main__":
    import scipy.stats

    s = RunningStat()
    a = np.random.randn(100)
    s.push(a)
    print(s)
    print(np.mean(a), np.std(a, ddof=1), scipy.stats.skew(a), scipy.stats.kurtosis(a))

    s = RunningStat(100)
    a = np.random.randn(100, 100)
    s.push(a)
    print(s)
    print(np.mean(a, axis=0).mean(), np.std(a, axis=0, ddof=1).mean(), scipy.stats.skew(a, axis=0).mean(), scipy.stats.kurtosis(a, axis=0).mean())
