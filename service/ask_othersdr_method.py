import numpy as np
from direpack import sudire
import random
from scipy.linalg import inv,sqrtm

def ask_othersdrmodel_service(x, y, Other_model_config, DR_results):
    '''
    输入x(n*p) y，降维方法
    输出beta的列表
    返回最优维度、空间距离
    同时也返回SDR网络的距离
    '''
    # 转化数据类型，提取参数

    methods = Other_model_config['methods']
    B = Other_model_config['B']
    n_slices=Other_model_config['n_slices']

    x = np.matrix(x)
    y = np.matrix(y)

    varmatx = np.cov(x, rowvar=0)
    meanx = x.mean(axis=0)
    N2 = inv(sqrtm(varmatx))

    #手动进行标准化
    x=x-meanx
    x=np.matmul(x, N2)


    # 记录其他模型的距离
    for method in methods:
        # 调用函数计算最优子空间

        best_dim, _  = my_estimate_dim(method, x, y, B, n_slices, N2)

        #已经标准化过了，所以不使用它的标准化，它的标准化很垃圾
        sdr_obj = sudire(sudiremeth=method, n_components=best_dim, scale_data = False, center_data = False, n_slices=n_slices )
        sdr_obj.fit(x, y)

        #这个时候的x_loadings并没有反标准化，也是原包里出错的地方，必须要有这一步。
        loadingBasis = np.matmul(N2 , sdr_obj.x_loadings_ )
        Space=hat(loadingBasis)

        # 保存信息
        DR_results.append({'method':method,
                      'DimensionFound':best_dim,
                      'EstimatedBasis':loadingBasis,
                      'EstimatedSpace':Space
                      })


def hat(P):
    #回归里面的帽子矩阵， x (x'x)^-1x'
    return np.matmul(np.matmul(P,inv(np.matmul(P.T,P))),P.T)


def my_estimate_dim(method, x, y, B, n_slices,N2):
    """
    Estimates the dimension of the central subspace using
    the sudiremeth.  This approach is based on the bootstrap method of Sheng and Yin (2016)

    Parameters
    ----------

        sudiremeth : str
                    the SDR method to use in the estimation.

        X :  numpy array or dataframe
            Input X data

        Y : vector or 1d matrix
            Input Y data as

        B : int
            Number of bootstrap replications

        n_slices: number of slices for SIR/SAVE/DR

    Returns
    ----------

    h : int
        representing the dimension of the central subspace
    ----------
    """

    n, p = x.shape

    diff_b = []
    mean_diff = []
    for k in range(1, p + 1):
        print('possible dim', k)

        sdr_obj = sudire(sudiremeth=method, n_components=k, scale_data = False, center_data = False, n_slices=n_slices )
        sdr_obj.fit(x, y=y)

        loadingBasis = np.matmul(N2 ,  sdr_obj.x_loadings_ )
        Space=hat(loadingBasis)

        for b in range(B):
            idx = np.random.randint(0, n, n)
            x_b = x[idx, :].copy()
            sdr_b = sudire(sudiremeth=method, n_components=k, scale_data=False, center_data=False, n_slices=n_slices)
            sdr_b.fit(x_b, y=y)

            loadingBasis_b = np.matmul( N2 , sdr_b.x_loadings_)
            Space_b = hat(loadingBasis_b)

            uh, sh, vh = np.linalg.svd( Space - Space_b)
            diff_b.append(np.nanmax(sh))

        mean_diff.append(np.mean(diff_b))

    return (np.argmin(mean_diff) + 1, mean_diff)