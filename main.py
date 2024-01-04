import argparse

import numpy as np
import torch
import wandb

from experiment.exp_Transformer import Exp_Transformer

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Transformer系列模型用于时序预测')

    # 基础设置
    parser.add_argument('--is_training', type=int,
                        required=False, default=1, help='状态')
    parser.add_argument('--model_id', type=str,
                        required=False, default='test', help='模型id')
    parser.add_argument('--model', type=str, required=False,
                        default='Transformer', help='Transformer模型')

    # 数据加载设置
    parser.add_argument('--data', type=str, required=False,
                        default='custom', help='数据集类型')
    parser.add_argument('--root_path', type=str,
                        default='./data/mdl/', help='数据集根路径')
    parser.add_argument('--data_path', type=str,
                        default='ts_data.csv', help='数据集文件名')
    parser.add_argument('--features', type=str, default='S',
                        help='预测任务类型, 选项:[M, S, MS]; M:多变量预测, S:单变量预测多变量, MS:多变量预测单变量')
    parser.add_argument('--target', type=str,
                        default='p_PDC201_T', help='预测目标特征, 在 S 或 MS 任务中')
    parser.add_argument('--freq', type=str, default='t',
                        help='时间特征编码频率, 选项:[s:秒, t:分钟, h:小时, d:日, b:工作日, w:周, m:月])')
    parser.add_argument('--checkpoints', type=str,
                        default='./checkpoints/', help='模型保存的文件夹路径')

    # 预测参数设置
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48, help='预测起始序列长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度')

    # Transformers系列模型参数设置
    parser.add_argument('--embedding_type', type=int, default=0,
                        help='0: 默认 (值嵌入 + 位置编码前嵌入 + 时间特征嵌入) 1: 值嵌入 + 时间特征嵌入 3: 值嵌入 + 位置编码前嵌入 4: 值嵌入')
    # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--enc_input', type=int, default=1, help='编码器输入特征维度')
    parser.add_argument('--dec_input', type=int, default=1, help='解码器输入特征维度')
    parser.add_argument('--c_out', type=int, default=1, help='输出特征维度')
    parser.add_argument('--d_model', type=int, default=512, help='模型特征维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头的数量')
    parser.add_argument('--e_layers', type=int, default=1, help='编码层层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码层层数')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络特征嵌入维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均的窗口大小')
    parser.add_argument('--factor', type=int, default=1,
                        help='Informer稀疏注意力采样因子')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='随机失活层概率')
    parser.add_argument('--embedding', type=str, default='timeF',
                        help='时间特征编码方式, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention',
                        action='store_true', help='编码器推理时是否同时输出注意力分数矩阵')
    parser.add_argument('--do_predict', action='store_true', help='是否预测未来数据')

    # 优化器设置
    parser.add_argument('--num_workers', type=int,
                        default=10, help='data loader workers数量')
    parser.add_argument('--itr', type=int, default=2, help='实验次数')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='训练输入数据批量大小')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--lr', type=float,
                        default=0.0001, help='优化器学习率')
    parser.add_argument('--des', type=str, default='test',
                        help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')
    parser.add_argument('--lradj', type=str, default='type3', help='调整学习率方式')
    parser.add_argument('--pct_start', type=float,
                        default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true',
                        help='use automatic mixed precision training', default=False)

    # GPU 设置
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--device_id', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str,
                        default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true',
                        default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.device_id = args.device_ids[0]
    
    wandb.init(project="transformer4TSF", config=args)
    

    print('实验参数设置:')
    print(args)

    Exp = Exp_Transformer(args)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embedding,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>开始训练 : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>预测 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embedding,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>测试 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
