"""
This algorithm used quaternion to solve the rotation and transformation matrices
"""
import numpy as np
import time
from sklearn.neighbors import KDTree


def kd_tree(model, X):
    """
    此函数用来用kd-tree的方法寻找一个点集model中点X的最近点，
    输入：
    model: 3*N
    X: 3*1
    返回值：
    最近点在model中的下表ind
    """
    # Reading data points
    pointset = np.array(model)
    pointset = pointset.T
    point = np.array(X)
    point = point.T
    point = point.tolist()

    np.random.seed(0)
    tree = KDTree(pointset, leaf_size=2)
    # ind：最近的3个邻居的索引
    # dist：距离最近的3个邻居
    # [X[2]]:搜索点
    start_time = time.clock()
    print("start time", start_time)
    dist, ind = tree.query(point, k=1)

    return ind


def find_nearest_points(src, dst):
    """
    计算两个点集src和dst最小欧氏距离
    其中，点集src相当于data, dst相当于model
    输入：
    src: 3*m的矩阵
    dst: 3*n的矩阵
    一般来说，n应该远大于m
    返回：
    Y: 点集dst中对应最短欧氏距离的点组成的集合
    distance: 最短欧氏距离（两个点集之间的平均距离）
    """
    # for a point in data set x_i in X, find the corresponding point y_i in P
    i = 0
    d_min = 10e10 * np.mat(np.ones((1, src.shape[1])))
    Y = np.mat(np.zeros((3, src.shape[1])))

    while i < src.shape[1]:
        Y[:, i] = dst[:, kd_tree(dst, src[:, i])].reshape([3, 1])
        d_min[:, i] = np.linalg.norm(Y[:, i] - src[:, i])
        i += 1

    least_distance = d_min.sum(axis=1) / src.shape[1]

    return Y, least_distance


def solve_t_r(data, model):
    """
    计算data和model之间的变换T，R，使得data经过变换后可以接近model

    输入：
    data: 3xm numpy array
    model: 3xm numpy array

    返回：
    R: mxm rotation matrix
    T: mx1 translation vector
    """
    # calculate the center of mass
    data_center = np.mean(data, axis=1)
    model_center = np.mean(model, axis=1)

    # Solve transformations
    sigma_px = np.zeros([3, 3])
    i = 0
    while i < data.shape[1]:
        sigma_px = sigma_px + data[:, i] * model[:, i].reshape(1, 3)
        i += 1
    sigma_px = sigma_px / data.shape[1] - data_center * model_center.reshape(1, 3)

    A = sigma_px - np.transpose(sigma_px)
    delta_T = np.zeros(3)
    delta_T[0] = A[1, 2]
    delta_T[1] = A[2, 0]
    delta_T[2] = A[0, 1]
    delta = delta_T.reshape(3, 1)
    tr = np.trace(sigma_px)
    Q_11 = tr
    Q_12 = delta_T
    Q_21 = delta
    Q_22 = sigma_px + np.transpose(sigma_px) - tr * np.eye(3)
    Q_upper = np.hstack((Q_11, Q_12))
    Q_lower = np.hstack((Q_21, Q_22))
    Q = np.zeros([4, 4])
    Q[0, :] = Q_upper
    Q[1, :] = Q_lower[0, :]
    Q[2, :] = Q_lower[1, :]
    Q[3, :] = Q_lower[2, :]
    # 求出Q的最大特征值和对应的特征向量
    lamda, q = np.linalg.eig(Q)
    lamda_max = i = index = 0
    while i < np.size(lamda):
        if lamda_max < lamda[i]:
            lamda_max = lamda[i]
            index = i
        i += 1
    q_R = q[:, index]
    q0 = q_R[0]
    q1 = q_R[1]
    q2 = q_R[2]
    q3 = q_R[3]

    R = [[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
         [2 * (q1*q2 + q0*q3), q0**2 + q2**2 - q1**2 - q3**2, 2 * (q2*q3 - q0*q1)],
         [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), q0**2 + q3**2 - q1**2 - q2**2]]

    q_T = model_center - R * data_center
    T = q_T.tolist()
    return R, T


def iterate(data, model, init_tran, max_iterations=200, threshold=0.01):
    """
    迭代最近点算法

      输入：
           data Nxm numpy array of source points
           model: Nxm numpy array of destination point
           init_tran: 用来做粗匹配的矩阵
           max_iterations: 最大迭代次数 exit algorithm after max_iterations
           threshold: 停止迭代的阈值 convergence criteria
      返回：
            src: 经过变换后的data点集
       """
    R = np.eye(3)
    T = [[0], [0], [0]]
    src = init_tran * data
    d_previous = 0
    dst, d_new = find_nearest_points(src, model)
    i = 0
    r = np.eye(3)
    while (abs(d_previous - d_new) > threshold) & (i < max_iterations):

        R, T = solve_t_r(src, dst)
        src = R * src + T
        d_previous = d_new
        dst, d_new = find_nearest_points(src, model)
        print("Iteration No.", i + 1, "  The distance is", abs(d_previous - d_new))
        i += 1
        end_time = time.clock()
        print("end time", end_time)
        print(end_time - start_time)
    return src
