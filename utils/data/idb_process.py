import pandas as pd
from pytz import timezone

def transfer_to_ts_data(file_path):
    """
    将Idb导出的csv文件转换为标准时序预测csv文件格式
    """
    data = pd.read_csv(file_path, skiprows=2)
    # 重命名列以便于理解
    data.rename(columns={'Unnamed: 5': 'date', 'Unnamed: 6': 'value', 'Unnamed: 7': 'feature'}, inplace=True)
    
    # 只保留需要的列，并去除可能存在的缺失特征行
    data = data[['date', 'feature', 'value']].dropna()
    
    # 将'value'列转换为数值型，无法转换的变为NaN
    data['value'] = pd.to_numeric(data['value'], errors='coerce')
    
    # 转换数据格式，每个特征一列
    time_series_data = data.pivot_table(index='date', columns='feature', values='value', aggfunc='first').reset_index()
    beijing = timezone('Asia/Shanghai')
    
    # 将时间转换为北京时间
    time_series_data['date'] = pd.to_datetime(time_series_data['date'], utc=True).dt.tz_convert(beijing).apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    
    return time_series_data

transfer_to_ts_data('./data/mdl/data.csv').to_csv('./data/mdl/ts_data.csv', index=False)