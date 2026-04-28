import os
import torch  # 第1步：先import torch
# 第2步：立即设置环境变量（在任何CUDA操作之前）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 第3步：现在可以安全地进行CUDA操作
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())  # 应该输出 1

import pickle
import numpy as np
from random import random
from data_loader import get_loader
from solver import Solver
import torch
# ddddd
import os
import argparse
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn
import optuna
import shap  # 确保已安装 SHAP：pip install shap
import optuna.visualization as vis
from optuna.visualization import plot_parallel_coordinate
word_emb_path = '../glove.840B.300d.txt'
assert(word_emb_path is not None)


username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = project_dir.joinpath('datasets')
data_dict = {'mosi': data_dir.joinpath('MOSI'), 'mosei': data_dir.joinpath(
    'MOSEI'), 'ur_funny': data_dir.joinpath('UR_FUNNY'), 'cmdc': data_dir.joinpath('CMDC'),
             'iemocap': os.path.join(data_dir,'IEMOCAP')}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh, "mish": nn.Mish}


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:       #此时kwargs是字典
            for key, value in kwargs.items(): #kwargs.items()生成键值对(key, value) 
                if key == 'optimizer':
                    value = optimizer_dict[value]
                if key == 'activation':
                    value = activation_dict[value]
                setattr(self, key, value)  #self是对象

        # Dataset directory: ex) ./datasets/cornell/
        self.dataset_dir = data_dict[self.data.lower()]
        self.sdk_dir = sdk_dir
        # Glove path
        self.word_emb_path = word_emb_path
        self.output_dim = 1 # 回归任务通常输出 1 维（如情感分数、数值预测等）

        # Data Split ex) 'train', 'valid', 'test'
        # self.data_dir = self.dataset_dir.joinpath(self.mode)
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()  #创建一个 ArgumentParser 对象
    parser.add_argument('--best_model_Configuration_Log', type=str, default='./src/best_Configuration_optuna.txt',
                        help='Load the best model to save features')
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=80)
    # Mode
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--runs', type=int, default=5)

    # Bert
    parser.add_argument('--use_bert', type=str2bool, default=True)
    parser.add_argument('--use_cmd_sim', type=str2bool, default=True)

    # Train
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    parser.add_argument('--name', type=str, default=f"{time_now}")
    parser.add_argument('--num_classes', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=10)

    # weight
    parser.add_argument('--cos_weight', type=float, default=0.01)
    parser.add_argument('--info_weight', type=float, default=0.3)
    parser.add_argument('--r_weight', type=float, default=0.1)
    parser.add_argument('--c_weight', type=float, default=0.3)
    parser.add_argument('--s_weight', type=float, default=0.1)
    parser.add_argument('--o_weight', type=float, default=0.05)
    # parser.add_argument('--w_silence', type=float, default=0.3)
    parser.add_argument('--temp', type=float, default=1.2)  # TriCoeffPredictor 里用于平滑Cr和Cc
    parser.add_argument('--num_heads', type=int, default=4) # ---------------------------------

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-04)

    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)

    parser.add_argument('--rnncell', type=str, default='lstm')  # rnn
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--sem_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--reverse_grad_weight', type=float, default=1.0)
    # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh", "mish"
    parser.add_argument('--activation', type=str, default='rrelu')

    # Model
    parser.add_argument('--model', type=str, default='SAMBA')

    # Data
    parser.add_argument('--data', type=str, default='mosi')  # cmdc\mosi\iemocap\mosei------------------------------------
    # cmdc的5折交叉验证
    parser.add_argument("--cross_validation", type=str,
                        choices=["cmdc_data_all_modal_1", "cmdc_data_all_modal_2", "cmdc_data_all_modal_3",
                                 "cmdc_data_all_modal_4", "cmdc_data_all_modal_5"], default="cmdc_data_all_modal_1")

    # Parse arguments 解析命令行参数
    if parse:
        kwargs = parser.parse_args() #kwargs存储命令行参数 name+default
    else:
        kwargs = parser.parse_known_args()[0]

    # print(kwargs.data)
    if kwargs.data == "mosi":
        kwargs.num_classes = 1
        kwargs.batch_size = 64 #64
    elif kwargs.data == "mosei":
        kwargs.num_classes = 1
        kwargs.batch_size = 16
    elif kwargs.data == "ur_funny":
        kwargs.num_classes = 2
        kwargs.batch_size = 32
    else:
        print("No dataset mentioned")
        exit()

    # Namespace => Dictionary
    kwargs = vars(kwargs)  #kwargs字典 
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


def reset_seed(seed=336):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 定义目标函数
def objective(trial):
    # reset_seed1 = trial.suggest_int("reset_seed1", 1, 1000)
    # reset_seed(reset_seed1)  # 必不可少,否则复现不出来
    reset_seed(336)  # 必不可少,否则复现不出来
    # num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16])  # 整数范围
    # batch_size = trial.suggest_categorical("batch_size", [32,64, 128, 256])  # 整数范围
    r_weight = trial.suggest_categorical("r_weight", [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]) 
    # w_silence = trial.suggest_categorical("w_silence", [1.0, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01])
    c_weight = trial.suggest_categorical("c_weight", [0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 0.01])
    s_weight = trial.suggest_categorical("s_weight", [0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    o_weight = trial.suggest_categorical("o_weight", [0.4,0.2, 0.1, 0.05, 0.03, 0.02, 0.01])
    cos_weight = trial.suggest_categorical("cos_weight", [0.1, 0.05, 0.02, 0.015, 0.01, 0.005])

    temp = trial.suggest_categorical("temp", [2.0, 1.2, 1.2, 1.0, 0.7, 0.5, 0.3, 0.1])
    dropout = trial.suggest_categorical("dropout", [0.6, 0.5, 0.4, 0.3, 0.1, 0.0])

    learning_rate = trial.suggest_categorical("learning_rate", [5e-4, 1e-4, 5e-5, 1e-5])
    weight_decay = trial.suggest_categorical("weight_decay", [5e-4, 1e-4, 5e-5, 1e-5])

    # ce_loss_weight = trial.suggest_float("ce_loss_weight", 1, 3)
    optimizer_c = trial.suggest_categorical("optimizer_c", ['RMSprop', 'Adam'])
    # center_score_weight = trial.suggest_float("center_score_weight", 0.00, 0.1)
    activation = trial.suggest_categorical("activation", ["leakyrelu", "prelu", "relu", "rrelu", "tanh", "mish"])  # 离散集合
    rnncell = trial.suggest_categorical("rnncell", ["lstm", "rnn"])  # 离散集合

    # Setting the config for each stage
    train_config = get_config(mode='train')
    # train_config.batch_size = batch_size
    train_config.r_weight = r_weight
    # train_config.w_silence = w_silence
    train_config.c_weight = c_weight
    train_config.s_weight = s_weight
    train_config.o_weight =o_weight
    train_config.cos_weight = cos_weight

    # train_config.num_heads = num_heads
    train_config.temp = temp
    train_config.rnncell = rnncell

    train_config.optimizer_c = optimizer_c
    train_config.learning_rate = learning_rate
    train_config.weight_decay = weight_decay
    train_config.dropout = dropout
    # train_config.pred_center_score_weight = center_score_weight
    train_config.activation = activation_dict[activation]

    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    # Creating pytorch dataloaders  批量加载数据
    train_data_loader = get_loader(train_config, shuffle = True)
    dev_data_loader = get_loader(dev_config, shuffle = False)
    test_data_loader = get_loader(test_config, shuffle = False)

    # Solver is a wrapper for model traiing and testing
    solver = Solver
    solver = solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

    # Build the model
    solver.build()

    # Train the model (test scores will be returned based on dev performance)
    MAE = solver.train()

    return MAE  # 返回目标值



if __name__ == '__main__':
     
    # Setting random seed  伪随机，方便复用
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)      #为CPU随机生成器设定种子
    torch.cuda.manual_seed_all(random_seed)  #为GPU
    torch.backends.cudnn.deterministic = True  #确定性算法
    torch.backends.cudnn.benchmark = False   #性能调优
    np.random.seed(random_seed)   #为 NumPy 库中的随机数生成器设定种子

    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定显卡使用第4卡，（0是第一块）

    study_name = "optimization_study_new"
    storage = f"sqlite:///{study_name}.db"
    # 开始超参数优化
    study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage, sampler=optuna.samplers.TPESampler(seed=336), load_if_exists=True)
    # 如果已经完成优化，直接加载，无需重新优化
    if len(study.trials) == 0:
        # 定义固定参数
        fixed_params = {
            # "batch_size": 256,
            "r_weight": 0.1,
            # "w_silence": 0.3,
            "c_weight": 0.3,
            "s_weight": 0.1,
            "o_weight": 0.05,
            # "num_heads": 4,
            "cos_weight": 0.01,
            "rnncell": 'lstm', 
            "temp": 1.2,
            "learning_rate": 5e-05,
            "weight_decay": 1e-04,
            "dropout": 0.1,
            "optimizer_c": 'Adam',            
            "activation": "rrelu",  # ["leakyrelu", "prelu", "relu", "rrelu", "tanh", "mish"]
        }

        # 手动插入固定参数为一个试验
        study.enqueue_trial(fixed_params)

        study.optimize(objective, n_trials=150)

        # 打印最佳参数
        print("Best parameters:", study.best_params)
        print("Best MAE:", study.best_value)

        # 可视化优化结果
        vis.plot_optimization_history(study).show()
        vis.plot_parallel_coordinate(study).show()
    else:
        print("Loaded existing Study from database.")

        # 加载 Study
        loaded_study = optuna.load_study(study_name="optimization_study_new", storage=storage)

        try:
            # 输出最优参数与结果
            print("Best hyperparameters:", loaded_study.best_trial.params)
            print("Best value (mae):", loaded_study.best_trial.value)
        except Exception as e:  # 看看数据库文件是否损坏，或没保存过最佳结果
            db_path = f"./{study_name}.db"
            print(f"[Warning] Load study failed: {e}")
            # 如果数据库文件存在，就删除
            if os.path.exists(db_path):
                print(f"[Info] Removing corrupted study DB: {db_path}")
                print(f"旧的db文件删除了，重新运行一遍吧")
                os.remove(db_path)
                exit(0)

        # 用最优参数再运行一次
        best_params = loaded_study.best_trial.params
        result = objective(optuna.trial.FixedTrial(best_params))
        print("Re-evaluated result with best params:", result)

        # 1. 选择你要显示的参数
        show_params = [
            "activation", 
            "learning_rate", 
            "dropout",
            "r_weight",
            "c_weight", 
            "s_weight", 
            "o_weight",
            "cos_weight"
        ]

        # 3. 画图（只显示你选的参数）
        fig = plot_parallel_coordinate(loaded_study, params=show_params)
        

        # 5. 线条粗细
        for trace in fig.data:
            if trace.type == 'scatter':
                trace.line.width = 8

        fig.show()




#Best hyperparameters: {'r_weight': 0.5, 'c_weight': 0.1, 's_weight': 0.4, 'o_weight': 0.02, 'cos_weight': 0.01, 'temp': 1.2, 'dropout': 0.5, 'learning_rate': 1e-05, 'weight_decay': 1e-05, 'optimizer_c': 'Adam', 'activation': 'prelu', 'rnncell': 'lstm'}
# Best value (objective): 0.6397603013941002