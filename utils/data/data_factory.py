from utils.data.data_loader import *
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom
}


def data_provider(args, flag):
    """
    数据提供器函数，用于根据指定的参数和标志创建数据集和数据加载器。

    Args:
        args (Namespace): 包含数据加载和处理所需所有设置的参数对象
        flag (str): 指定数据集的类型（'train', 'test', 'pred'等）

    Returns:
        data_set: 创建的数据集对象
        data_loader: 创建的数据加载器对象, 用于批量、高效地加载数据。
    """
    Data = data_dict[args.data]
    timeenc = 1 if args.embedding == 'timeF' else 0

    if flag == 'test':
        # 测试集
        shuffle_flag = False
        drop_last = True  # 丢弃最后一个batch
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        # 预测集
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        # 训练集和验证集
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return data_set, data_loader
