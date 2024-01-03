from typing import Any, List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    这是一个抽象基类，定义了时间特征类应有的基本结构和方法
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    返回每分钟的第几秒(将秒数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    返回每小时的第几分钟(将分钟数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    返回每天的第几小时(将小时数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    返回每周的第几天(将天数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    返回每月的第几天(将天数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    返回每年的第几天(将天数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    返回每年的第几周(将周数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    返回每年的第几月(将月数编码为介于[-0.5, 0.5]的值)
    """

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


def time_features_from_freq_str(freq_str: str) -> List[TimeFeature]:
    """
    根据给定的频率字符串返回一个适当的时间特征类列表。

    Args:
        freq_str (str): 表示时间频率的字符串，如 "12H"（每12小时）、"5min"（每5分钟）、"1D"（每天）等。

    Returns:
        List[TimeFeature]: 一个TimeFeature类实例的列表（List[TimeFeature]）。列表中的每个元素都是一个时间特征类的实例，这些类实例可以用来将日期时间索引转换为具体的时间特征编码。
    """
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    不支持的频率: {freq_str}
    支持以下频率:
        Y   - 每年
            别称: A
        M   - 每月
        W   - 每周
        D   - 每天
        B   - 每工作日
        H   - 每小时
        T   - 每分钟
            别称: min
        S   - 每秒
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """
    将日期时间索引转换为具体的时间特征编码。

    Args:
        dates (pd.DatetimeIndex): 日期时间索引
        freq (str, optional): 表示时间频率的字符串，如 "12H"（每12小时）、"5min"（每5分钟）、"1D"（每天）等. Defaults to 'h'.

    Returns:
        np.ndarray: 时间特征编码
    """
    features = time_features_from_freq_str(freq)
    return np.vstack([feature(dates) for feature in features])
