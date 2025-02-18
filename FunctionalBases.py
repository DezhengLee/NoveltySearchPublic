import numpy as np
from numpy.polynomial import legendre, chebyshev


class FunctionalBases:
    """
    一个示例性的类，包含常用的函数基及其评估方法：
    1. Fourier 基 (sin, cos, sin+cos)
    2. Haar 小波基 (示例实现)
    3. 勒让德正交多项式基
    4. (可选) 切比雪夫多项式基
    """

    def __init__(self, x):
        """
        初始化

        参数:
        x (array-like): 一维的网格/采样点，例如 np.linspace(0, 1, 100)。
                        假设我们要在这些点上评估基函数。
        """
        self.x = np.array(x, dtype=float)


    def fourier_basis(self, order, kind='cos'):
        """
        生成 Fourier 基函数在 x 上的值。支持 'sin' / 'cos' / 'sin+cos' 三种类型。

        参数:
        order (int): 最高阶数(从 1 到 order)。
        kind (str): 选择 'sin'、'cos' 或 'sin+cos'。

        返回:
        basis_matrix (2D numpy array): 形状为 (len(x), M)，M 取决于 kind 和 order。
                                       每一列是一阶基函数在 x 上的取值。
        """
        x = self.x
        n_points = len(x)

        # 按照 kind 不同，基函数数量不同
        if kind == 'sin':
            # 只包含 sin(k * 2pi x) (k=1..order)
            basis_matrix = np.zeros((n_points, order))
            for k in range(1, order + 1):
                basis_matrix[:, k - 1] = np.sin(2 * np.pi * k * x)
            return basis_matrix

        elif kind == 'cos':
            # 只包含 cos(k * 2pi x) (k=0..order-1) 或者 (k=1..order)，可按需选择
            # 这里示例直接从 k=1 到 order
            basis_matrix = np.zeros((n_points, order))
            for k in range(1, order + 1):
                basis_matrix[:, k - 1] = np.cos(2 * np.pi * k * x)
            return basis_matrix

        elif kind == 'sin+cos':
            # 同时包含 sin(k*x) 和 cos(k*x)，共 2*order 个基函数
            basis_matrix = np.zeros((n_points, 2 * order))
            idx = 0
            for k in range(1, order + 1):
                basis_matrix[:, idx] = np.sin(2 * np.pi * k * x)
                basis_matrix[:, idx + 1] = np.cos(2 * np.pi * k * x)
                idx += 2
            return basis_matrix

        else:
            raise ValueError("未知的 kind 类型，可选: 'sin', 'cos', 'sin+cos'")


    def haar_wavelet_basis(self, level):
        """
        生成简单的 Haar 小波基（连续定义）在 [0,1] 范围内的取值。
        假设 self.x 分布在 [0,1]。

        level (int): 小波的层数，例如 level=3 则包含 3 层缩放。

        返回:
        basis_matrix (2D numpy array): 形状 (len(x), n_basis)，其中 n_basis
                                       与 level 相关 (包含最高层的缩放函数和不同平移的母小波)。
        说明:
        - Haar 缩放函数 (父小波): phi_{j,k}(x) = 2^(j/2)*phi(2^j x - k)，k = 0..(2^j -1)
        - Haar 母小波: psi_{j,k}(x) = 2^(j/2)*psi(2^j x - k)，k = 0..(2^j -1)
        - 这里只是示例实现，未必包含所有细节。可以按照需要裁剪或拓展。
        """

        def phi(x):
            """Haar 缩放函数(父小波)：[0,1) 上为 1，其它为 0"""
            return np.where((x >= 0) & (x < 1), 1.0, 0.0)

        def psi(x):
            """Haar 母小波：在 [0, 0.5) 上为 +1, [0.5, 1) 上为 -1，其它为 0"""
            return np.where((x >= 0) & (x < 0.5), 1.0,
                            np.where((x >= 0.5) & (x < 1.0), -1.0, 0.0))

        # 收集所有基函数的值
        basis_functions = []

        # 最高层（或每一层）的“缩放函数”（父小波）
        # 通常可以只取最高层，也可以每层都加入一个。此处示例只加入最高层 level。
        j = level - 1  # 最高层
        for k in range(2 ** j):
            def scaling_function(t, j=j, k=k):
                return (2 ** (j / 2)) * phi(2 ** j * t - k)

            basis_functions.append(scaling_function)

        # 母小波函数
        # 遍历从第 0 层到 level-1 层，每层都有 2^j 个平移
        for j in range(level):
            for k in range(2 ** j):
                def wavelet_function(t, j=j, k=k):
                    return (2 ** (j / 2)) * psi(2 ** j * t - k)

                basis_functions.append(wavelet_function)

        # 在 self.x 上评估
        n_points = len(self.x)
        n_basis = len(basis_functions)
        basis_matrix = np.zeros((n_points, n_basis))

        for i, func in enumerate(basis_functions):
            basis_matrix[:, i] = func(self.x)

        return basis_matrix


    def legendre_basis(self, order):
        """
        生成前 order+1 阶(从 0 到 order)的勒让德多项式在 x 上的取值。
        若只需要 1 ~ order，可以酌情更改。

        参数:
        order (int): 多项式最高阶。

        返回:
        basis_matrix (2D numpy array): 形状 (len(x), order+1)，
                                       第 k 列表示 P_k(x) 在 self.x 上的值。
        注意:
        - 默认情况下，numpy.polynomial.legendre.legval() 的勒让德多项式定义域是 [-1,1]。
          如果你的 x 不在 [-1,1]，可能需要先做线性映射使其落在 [-1,1]。
          这里假设 x 本身已经在 [-1,1]。
        """
        # 如果 self.x 不在 [-1,1] 上，可以做一次映射
        # 例如:
        # x_mapped = 2 * (self.x - min_x)/(max_x - min_x) - 1
        x = self.x

        # 生成勒让德基函数矩阵
        # legval(x, c): c 是系数数组，长度 n+1 对应 P_0, P_1, ..., P_n
        n_points = len(x)
        basis_matrix = np.zeros((n_points, order + 1))
        for k in range(order + 1):
            c = np.zeros(k + 1)
            c[-1] = 1.0  # 表示 P_k(x)
            basis_matrix[:, k] = legendre.legval(x, c)
        return basis_matrix


    def chebyshev_basis(self, order):
        """
        生成切比雪夫多项式 T_k(x) (k = 0..order) 在 x 上的取值。
        同样需要注意 x 要在 [-1, 1] 上。

        参数:
        order (int): 最高阶。

        返回:
        basis_matrix (2D numpy array): 形状 (len(x), order+1)。
        """
        x = self.x
        n_points = len(x)
        basis_matrix = np.zeros((n_points, order + 1))
        for k in range(order + 1):
            c = np.zeros(k + 1)
            c[-1] = 1.0  # 表示 T_k(x)
            basis_matrix[:, k] = chebyshev.chebval(x, c)
        return basis_matrix


# -----------------------------------------------------------------------------
# 使用示例
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. 构造 x 点，示例取 [0,1] 区间上的 100 个点
    x = np.linspace(0, 1, 100)

    # 如果要使用勒让德/切比雪夫，需要将 x 映射到 [-1, 1]
    # 这里只是示例，实际可根据需要选择
    x_legendre = 2 * (x - 0) / (1 - 0) - 1  # 把 [0,1] -> [-1,1]

    fb_fourier = FunctionalBases(x)
    fb_poly = FunctionalBases(x_legendre)

    # 2. 生成 Fourier 基 (sin+cos)，阶数=3
    fourier_matrix = fb_fourier.fourier_basis(order=3, kind='sin+cos')
    print("Fourier (sin+cos) 基矩阵形状: ", fourier_matrix.shape)

    # 3. 生成 Haar 小波基，层数=3
    haar_matrix = fb_fourier.haar_wavelet_basis(level=3)
    print("Haar 小波基矩阵形状: ", haar_matrix.shape)

    # 4. 生成勒让德多项式基，最高阶=3 (生成 P0, P1, P2, P3)
    legendre_matrix = fb_poly.legendre_basis(order=3)
    print("勒让德多项式基矩阵形状: ", legendre_matrix.shape)

    # 5. (可选) 生成切比雪夫多项式基，最高阶=3
    cheb_matrix = fb_poly.chebyshev_basis(order=3)
    print("切比雪夫多项式基矩阵形状: ", cheb_matrix.shape)
