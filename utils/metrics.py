import numpy as np

def RSE(pred, true):
    """
    相对平方误差
    """
    return np.sqrt(np.sum(np.square(pred - true))) / np.sqrt(np.sum(np.square(true - np.mean(true))))

def CORR(pred, true):
    """
    相关系数
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)

def MAE(pred, true):
    """
    平均绝对误差
    """
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """
    均方误差
    """
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """
    均方根误差
    """
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """
    平均绝对百分比误差
    """
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    """
    均方百分比误差
    """
    return np.mean(np.square((pred - true) / true))

def metrics(pred, true):
    """
    评价指标
    """
    return {'MAE': MAE(pred, true), 'MSE': MSE(pred, true), 'RMSE': RMSE(pred, true),
            'MAPE': MAPE(pred, true), 'MSPE': MSPE(pred, true), 'RSE': RSE(pred, true),
            'CORR': CORR(pred, true)}

