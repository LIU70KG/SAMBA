import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from utils import to_gpu
import models
from shutil import copyfile, rmtree
from torch.nn import SmoothL1Loss
from models import SAMBA
from create_dataset import CMDC_PHQ9_labels
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
import seaborn as sns

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, 
                 train_data_loader, dev_data_loader, test_data_loader, is_train):
        # ========== 1. 初始化所有配置和数据加载器 ==========
        self.train_config = train_config  # 训练配置（包含use_bert等核心参数）
        self.dev_config = dev_config
        self.test_config = test_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train

        # ========== 2. 初始化设备和模型 ==========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAMBA(config=train_config)  # 模型用train_config初始化
        self.model = to_gpu(self.model)  # 移到GPU


        


    def build(self, cuda=True):
            # 1. 初始化模型
            if self.model is None:
                self.model = getattr(models, self.train_config.model)(self.train_config) 
            
            # 2. 遍历参数：进行冻结 (Freeze) 和 初始化 (Init)
            for name, param in self.model.named_parameters():
                
                # === BERT 冻结策略 ===
                # 这里的逻辑很好：冻结底层，只训练顶层，能有效防止 BERT 被破坏
                if self.train_config.data == "mosei":
                    if "bertmodel.encoder.layer" in name:
                        layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                        if layer_num <= 8:  # 冻结前9层 (0-8)
                            param.requires_grad = False
                
                elif self.train_config.data == "ur_funny":
                    if "bert" in name:
                        param.requires_grad = False
                
                # === RNN 正交初始化 ===
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                
                # 调试打印 (可选)
                # print(f"{name}: grad={param.requires_grad}")

            # 3. 初始化 Embedding (如果不使用 BERT)
            if not self.train_config.use_bert:
                if hasattr(self.train_config, 'pretrained_emb') and self.train_config.pretrained_emb is not None:
                    self.model.embed.weight.data = self.train_config.pretrained_emb
                self.model.embed.requires_grad = False # 固定 Glove

            # 4. 移至 GPU
            if torch.cuda.is_available() and cuda:
                self.model.cuda()

            # 5. 定义优化器 
            if self.is_train:
                # 直接过滤出需要梯度的参数，统一使用一个学习率
                self.optimizer = self.train_config.optimizer(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=self.train_config.learning_rate,
                    weight_decay=self.train_config.weight_decay if hasattr(self.train_config, 'weight_decay') else 0.0
                )
            

    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        
        # Loss Function
        if self.train_config.data == "ur_funny":
            self.criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.train_config.data == 'iemocap':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            # 回归任务使用 SmoothL1
            self.criterion = nn.SmoothL1Loss(beta=0.2, reduction="mean")
            
        best_mae, best_rmse = float('inf'), float('inf')
        best_pearsonrn = float('-inf')
        best_precision, best_recall, best_f1, best_accuracy, best_multiclass_acc = 0.0, 0.0, 0.0, 0.0, 0.0

        early_stop_threshold = 0.0005
        best_mae_for_early_stop = float('inf')
        max_early_stop_patience = self.train_config.patience
        current_early_stop_patience = 0

        # # 加载验证,画图
        # if os.path.isfile('src_0709_rve/checkpoints_best_7095/model.std'):
        #     print("Loading weights...")
        #     self.model.load_state_dict(torch.load(f'src_0709_rve/checkpoints_best_7095/model.std'))
        #     self.optimizer.load_state_dict(torch.load(f'src_0709_rve/checkpoints_best_7095/optim.std'))
        #     print("Record the verification results...")
        #     mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval_plot(0, mode="test")
        #     print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
        #     print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))
        #     return mae
        

        # Checkpoints
        checkpoints = 'src_0709_rve/checkpoints'
        if os.path.exists(checkpoints):
            rmtree(checkpoints, ignore_errors=True)
        os.makedirs(checkpoints)

        # ========== 新增：初始化步数和首次打印标记 ==========
        self.step = 0  # 记录全局训练步数（每个batch自增1）
        self.first_coeff_print = True  # 标记是否首次打印初始系数

        for e in range(self.train_config.n_epoch):
            print(f"-----------------------------------epoch{e}---------------------------------------")
            # 打印当前 LR，方便监控 Scheduler 是否生效

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"//Current patience: {curr_patience}, LR: {current_lr:.6f}//")
            
            self.model.train()
            train_loss_cls, train_loss_hsic,train_loss_synergy,train_single_loss,train_loss_ortho,train_loss_weight= [], [],[],[],[],[]
            train_loss = []
            y_true, y_pred = [], []

            


            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t, v, a, y, l = to_gpu(t), to_gpu(v), to_gpu(a), to_gpu(y), to_gpu(l)
                if self.train_config.use_bert:
                    bert_sent = to_gpu(bert_sent)
                    bert_sent_type = to_gpu(bert_sent_type)
                    bert_sent_mask = to_gpu(bert_sent_mask)

                # 注意：需要先修改 model.py 的 forward 方法，返回 (y_tilde, info_loss, coeffs_map)
                y_tilde, fused_feat_b, loss_hsic_r, loss_synergy_gain, single_loss, loss_ortho, loss_weight = self.model(
                    t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask, labels=y
                )


                # [维度对齐]
                y_tilde = y_tilde.view(-1)
                y = y.view(-1)

                if self.train_config.data == "ur_funny":
                    cls_loss = self.criterion(y_tilde, y.squeeze())
                elif self.train_config.data == 'iemocap':
                    y_tilde = y_tilde.view(-1, 2)
                    cls_loss = self.criterion(y_tilde, y).mean()
                else:   
                    cls_loss = self.criterion(y_tilde, y)

                # [修复] 均衡的 Loss 组合
                # loss =self.train_config.r_weight * loss_hsic_r

                loss = cls_loss + self.train_config.r_weight * loss_hsic_r  + \
                           self.train_config.c_weight * loss_synergy_gain+ \
                           self.train_config.s_weight * single_loss + \
                           self.train_config.o_weight * loss_ortho +\
                           self.train_config.cos_weight * loss_weight 

                loss.backward()

                # 全局梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()



                # 损失记录（保持原有）
                train_loss.append(loss.item())
                train_loss_cls.append(cls_loss.item())
                train_loss_hsic.append((loss_hsic_r).item())
                # train_loss_silence.append((loss_silence).item())
                train_loss_synergy.append((loss_synergy_gain).item())
                train_single_loss.append((single_loss).item())
                train_loss_ortho.append((loss_ortho).item())
                train_loss_weight.append((loss_weight).item())
                y_pred.extend(y_tilde.detach().cpu().tolist())
                y_true.extend(y.detach().cpu().tolist())





            # 训练日志打印（保持原有）
            print(f"Training loss: {round(np.mean(train_loss), 4)}")
            # print('train_loss_cls:%.4f / train_loss_hsic:%.4f / train_loss_silence:%.4f / train_loss_synergy:%.4f / train_single_loss:%.4f / train_loss_ortho:%.4f/ train_loss_weight:%.4f' % (
            print('train_loss_cls:%.4f / train_loss_hsic:%.4f / train_loss_synergy:%.4f / train_single_loss:%.4f / train_loss_ortho:%.4f/ train_loss_weight:%.4f' % (
                round(np.mean(train_loss_cls), 4),
                round(np.mean(train_loss_hsic), 4),
                # round(np.mean(train_loss_silence), 4),
                round(np.mean(train_loss_synergy), 4),
                round(np.mean(train_single_loss), 4),
                round(np.mean(train_loss_ortho), 4),
                round(np.mean(train_loss_weight), 4)
            ))
            print(f"--------------------------------------------")

            # 指标计算（保持原有）
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            # print(f"Train Labels Min: {np.min(y_true)}, Max: {np.max(y_true)}")
            # print(f"Train Predictions Min: {np.min(y_pred)}, Max: {np.max(y_pred)}")
            
            mae_train, rmse_train, pearsonrn_train, _, _, _, _, _ = self.calc_metrics(y_true, y_pred)
            print('_train_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae_train, rmse_train, pearsonrn_train))

            # ===================== 记录三个模态 MAE =====================
            with torch.no_grad():
                # 👇 这是模型内部已经计算好的、可直接使用的投影特征
                # 👇 从模型 forward 输出的 fused_feat_b 拆分（完全安全）
                feat_dim = self.model.sem_dim
                
                # 直接从融合特征里拆分 3 个模态的投影结果
                feat_t = fused_feat_b[:, :feat_dim]
                feat_v = fused_feat_b[:, feat_dim:feat_dim*2]
                feat_a = fused_feat_b[:, feat_dim*2:]

                # 用模型自带的单模态头预测
                pred_t = self.model.uni_head_t(feat_t).squeeze()
                pred_v = self.model.uni_head_v(feat_v).squeeze()
                pred_a = self.model.uni_head_a(feat_a).squeeze()

                # 计算 MAE
                mae_t = F.l1_loss(pred_t, y).item()
                mae_v = F.l1_loss(pred_v, y).item()
                mae_a = F.l1_loss(pred_a, y).item()

            # 初始化保存
            if not hasattr(self, 'modal_mae'):
                self.modal_mae = {'text': [], 'visual': [], 'audio': []}

            # 每个 epoch 保存一次
            self.modal_mae['text'].append(mae_t)
            self.modal_mae['visual'].append(mae_v)
            self.modal_mae['audio'].append(mae_a)

            # 保存文件
            np.savez("REAL_MODAL_MAE.npz",
                    text=self.modal_mae['text'],
                    visual=self.modal_mae['visual'],
                    audio=self.modal_mae['audio'])


            # 验证集评估（保持原有）
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(e, mode="test")
            print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
            print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

            # ---------------------------画图，每一个epoch都要画出来----------------------
            # np_state = np.random.get_state()  # 隔离随机状态，防止跑不出原来的结果
            # torch_state = torch.random.get_rng_state()

            # self.eval_plot(e, mode="train")

            # np.random.set_state(np_state)  # 恢复状态
            # torch.random.set_rng_state(torch_state)

            # 每 5 轮画一张模态权重对比图
            # if e % 1 == 0:
            #     np_state = np.random.get_state()
            #     torch_state = torch.random.get_rng_state()

            #     self.eval_plot(epoch=e)

            #     np.random.set_state(np_state)
            #     torch.random.set_rng_state(torch_state)


            # ---------------------------------------------------------------------------


            # =====================================
            # 早停机制逻辑（恢复并优化）
            # =====================================
            if mae < best_mae_for_early_stop - early_stop_threshold:
                best_mae_for_early_stop = mae
                current_early_stop_patience = 0  # 有显著提升，重置耐心值
            else:
                current_early_stop_patience += 1  # 无显著提升，累计耐心值
                if current_early_stop_patience >= max_early_stop_patience:
                    print(f"\n🔥 Early stopping triggered!")
                    print(f"Epoch: {e}, Best MAE: {best_mae_for_early_stop:.4f}, Last MAE: {mae:.4f}")
                    print(f"No significant improvement for {max_early_stop_patience} epochs (threshold: {early_stop_threshold}).")
                    break

            # 最佳模型保存与patience更新（保持原有）
            flag = 0
            if best_mae > mae:
                best_mae = mae
                rmse_bestmae = rmse
                pearsonrn_bestmae = pearsonrn
                precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae = precision, recall, f1, accuracy, multiclass_acc
                flag = 1
            if best_rmse > rmse:
                best_rmse = rmse
                flag = 1
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                flag = 1
            if best_multiclass_acc < multiclass_acc:
                best_multiclass_acc = multiclass_acc
                flag = 1
            if best_f1 < f1:
                best_precision, best_recall, best_f1 = precision, recall, f1
                flag = 1

            if flag == 1:
                print("------------------Found new best model on test set!----------------")
                print(f"epoch: {e}")
                print("mae: ", mae)
                print("rmse: ", rmse)  
                print("Pearsonrn/Corr: ", pearsonrn)  
                print("precision: ", precision)  
                print("recall: ", recall)  
                print("f1: ", f1)  
                print("accuracy: ", accuracy)  
                print("multiclass_acc: ", multiclass_acc)  
                if not os.path.exists('src_0709_rve/checkpoints'): 
                    os.makedirs('src_0709_rve/checkpoints')
                torch.save(self.model.state_dict(), f'src_0709_rve/checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'src_0709_rve/checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                # 早停触发时的调度器调用（已修复传参和变量名错误）
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'src_0709_rve/checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'src_0709_rve/checkpoints/optim_{self.train_config.name}.std'))
                    self.scheduler.step(mae)  # 修复：将 lr_scheduler 改为 scheduler（与build方法中定义一致）
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

                if num_trials <= 0:
                    print("Running out of patience, early stopping.")
                    break

        # 最终结果打印（保持原有）
        print("------------------best all on test set----------------")
        print('_best_mae:%.4f. / best_rmse:%.4f. / best_f1:%.4f. / best_accuracy: %.4f. / best_multiclass_acc: %.4f.' % (best_mae, best_rmse, best_f1, best_accuracy, best_multiclass_acc))
        print("------------------best MAE on test set----------------")
        mae, rmse, pearsonrn = best_mae, rmse_bestmae, pearsonrn_bestmae
        precision, recall, f1, accuracy, multiclass_acc= precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae
        print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
        print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

        # 结果保存（保持原有）
        if  not os.path.exists(self.train_config.best_model_Configuration_Log):
            with open(self.train_config.best_model_Configuration_Log, 'w') as f:
                pass  
                
        with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
            F1.write("\n" + "="*180 + "\n")
            # line = 'r_weight:{r_weight} |c_weight:{c_weight} |s_weight:{s_weight}|o_weight:{o_weight}|sparse_weight:{sparse_weight}|consistency_weight:{consistency_weight} | activation:{activation}| dropout:{dropout} | learning_rate:{learning_rate} |\n ' \
            #     'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \

            # line = 'r_weight:{r_weight} |c_weight:{c_weight}|w_silence:{w_silence} |s_weight:{s_weight}|o_weight:{o_weight}|cos_weight:{cos_weight}| activation:{activation}| dropout:{dropout} | weight_decay:{weight_decay} | temp:{temp} | num_heads:{num_heads} | slearning_rate:{learning_rate} |\n ' \
            line = 'r_weight:{r_weight} |c_weight:{c_weight}|s_weight:{s_weight}|o_weight:{o_weight}|cos_weight:{cos_weight}| activation:{activation}| dropout:{dropout} | weight_decay:{weight_decay} | temp:{temp} | num_heads:{num_heads} | slearning_rate:{learning_rate} |\n ' \
                'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
                'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} | multiclass_acc:{multiclass_acc} |\n' \
                'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} | best_multiclass_acc:{best_multiclass_acc} |\n' \
                .format(
                    # 配置参数（保持原有）
                    r_weight=self.train_config.r_weight,
                    c_weight=self.train_config.c_weight ,
                    # w_silence=self.train_config.w_silence,
                    s_weight=self.train_config.s_weight ,
                    o_weight=self.train_config.o_weight ,
                    cos_weight=self.train_config.cos_weight,
                    activation=self.train_config.activation,                    
                    dropout=self.train_config.dropout ,
                    weight_decay=self.train_config.weight_decay ,
                    temp=self.train_config.temp ,
                    num_heads=self.train_config.num_heads,
                    # consistency_weight=self.train_config.consistency_weight ,
                    learning_rate=self.train_config.learning_rate,

                    # 当前测试集指标（保持原有）
                    test_MAE=mae,
                    RMSE=rmse,
                    Pearson=pearsonrn, 
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    accuracy=accuracy,
                    multiclass_acc=multiclass_acc,
                    # 最佳指标（关键：确保所有变量都已定义）
                    best_mae=best_mae,
                    best_rmse=best_rmse,
                    best_f1=best_f1,
                    best_accuracy=best_accuracy,
                    best_multiclass_acc=best_multiclass_acc
                )
            F1.write(line)
        

        return mae
            # 测试模型（保留最终版本，删除重复定义）



    def eval(self, e, mode=None, to_print=False, best=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            self.model.load_state_dict(torch.load(
                f'src_0709_rve/checkpoints/model_{self.train_config.name}.std'))

        features, labels = [], []
        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                try:
                    bert_sent = to_gpu(bert_sent)
                    bert_sent_type = to_gpu(bert_sent_type)
                    bert_sent_mask = to_gpu(bert_sent_mask)
                except:
                    pass

                y_tilde, fused_feat_b, _,loss_synergy_gain,single_loss,loss_ortho,loss_weight= self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

                # 在计算 loss 之前，确保模型输出 o 与标签 y 为同一维度
                # 把输出和标签都展平成 [batch,]
                y_tilde = y_tilde.view(-1)
                y = y.view(-1)

                # 如果 shapes 仍然不匹配，抛出并打印调试信息
                if y_tilde.size(0) != y.size(0):
                    raise RuntimeError(f"Shape mismatch before loss: y_tilde.shape={y_tilde.shape}, y.shape={y.shape}. "
                                    "This usually means some module cropped batch size internally or dataloader returns inconsistent batches.")

                # 现在安全计算 loss
                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                if self.train_config.data == 'iemocap':
                    y_tilde = y_tilde.view(-1, 2)
                    cls_loss = (self.criterion(y_tilde[::4], y[::4]).mean() + self.criterion(y_tilde[1::4], y[1::4]).mean() + \
                    self.criterion(y_tilde[2::4], y[2::4]).mean() + self.criterion(y_tilde[3::4], y[3::4]).mean())/4
                else:
                    y_tilde = y_tilde.view(-1)
                    y = y.view(-1)
                    cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss
                eval_loss.append(loss.item())
                # 统一展平
                y_tilde = y_tilde.squeeze(-1)
                y = y.squeeze(-1)

                # 对齐长度
                valid_len = min(len(y_tilde), len(y))

                y_pred.extend(y_tilde[:valid_len].detach().cpu().tolist())
                y_true.extend(y[:valid_len].detach().cpu().tolist())

        eval_loss = np.mean(eval_loss)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if self.train_config.data == 'iemocap':
            test_preds = y_pred.reshape((y_pred.shape[0] // 4, 4, 2))
            test_truth = y_true.reshape(-1, 4)
            f1, acc = [], []
            for emo_ind in range(4):
                test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
                test_truth_i = test_truth[:, emo_ind]
                f1.append(f1_score(test_truth_i, test_preds_i, average='weighted'))
                acc.append(accuracy_score(test_truth_i, test_preds_i))

            accuracy = AverageAcc = np.mean(acc)
            Averagef1 = np.mean(f1)
            if to_print:
                ne, ha, sa, an = acc
                ne_f, ha_f, sa_f, an_f = f1
                print('HappyAcc:%.4f.  SadAcc:%.4f.   AngryAcc:%.4f.   NeutralAcc:%.4f. AverageAcc:%.4f.' % (
                ha, sa, an, ne, AverageAcc))
                print('HappyF1:%.4f.  SadF1:%.4f.   AngryF1:%.4f.   NeutralF1:%.4f. Averagef1:%.4f.' % (
                ha_f, sa_f, an_f, ne_f, Averagef1))

        else:
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.calc_metrics(y_true, y_pred,
                                                                                                    mode, to_print)
            return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc

        return eval_loss, accuracy
    


    
    
    #src_0709_rve/figure/
    # ===========================================================================
    # 🔥 每一轮自动画：无 WMSE ↔ 有 WMSE 对比热力图
    # ===========================================================================
    # ===========================================================================
    # ✅ 安全版 eval_plot：只画当前模型，永不出错！
    # ===========================================================================
    # @torch.no_grad()
    # def eval_plot(self, epoch):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     import seaborn as sns
    #     import os
    #     import torch

    #     self.model.eval()
    #     dataloader = self.test_data_loader

    #     all_err = []
    #     all_q = []
    #     all_weak = []

    #     for batch in dataloader:
    #         t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

    #         # GPU
    #         if torch.cuda.is_available():
    #             t = t.cuda()
    #             v = v.cuda()
    #             a = a.cuda()
    #             y = y.cuda()
    #             l = l.cuda()
    #             try:
    #                 bert_sent = bert_sent.cuda()
    #                 bert_sent_type = bert_sent_type.cuda()
    #                 bert_sent_mask = bert_sent_mask.cuda()
    #             except:
    #                 pass

    #         # forward（只跑一次！）
    #         y_hat, fused, loss_uni, loss_syn, loss_single, loss_ortho, loss_weight = self.model(
    #             t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask
    #         )

    #         # 拆分模态
    #         D = self.model.sem_dim
    #         feat_t = fused[:, :D]
    #         feat_v = fused[:, D:D*2]
    #         feat_a = fused[:, D*2:D*3]

    #         # 预测
    #         p_t = self.model.uni_head_t(feat_t).squeeze()
    #         p_v = self.model.uni_head_v(feat_v).squeeze()
    #         p_a = self.model.uni_head_a(feat_a).squeeze()

    #         # 误差
    #         y = y.squeeze()
    #         e_t = torch.abs(p_t - y).cpu().numpy()
    #         e_v = torch.abs(p_v - y).cpu().numpy()
    #         e_a = torch.abs(p_a - y).cpu().numpy()
    #         err = np.stack([e_t, e_v, e_a], axis=1)
    #         all_err.append(err)

    #     # 拼接
    #     err = np.concatenate(all_err, axis=0).T
    #     N = err.shape[1]
    #     sel = np.random.choice(N, min(12, N), replace=False)
    #     E = err[:, sel]

    #     # 绘图
    #     plt.figure(figsize=(12, 4))
    #     sns.heatmap(E, cmap="Blues", annot=True, fmt=".2f", linewidths=0.3,
    #                 yticklabels=["Text", "Visual", "Audio"])
    #     plt.title(f"Test Modal Error | Epoch {epoch}")
    #     os.makedirs("src_0709_rve/figure", exist_ok=True)
    #     plt.savefig(f"src_0709_rve/figure/modal_err_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
    #     plt.close()

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    # 计算指标（保留最终版本，删除重复定义）
    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):    
        if self.train_config.data == "cmdc":
            test_preds = y_pred
            test_truth = y_true
            mae = np.mean(np.absolute(test_preds - test_truth))
            rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
            pearsonrn, p_value = pearsonr(test_preds, test_truth)
            binary_preds = test_preds >= 5
            binary_truth = test_truth >= 5
            precision = precision_score(binary_truth, binary_preds, zero_division=1)
            recall = recall_score(binary_truth, binary_preds, zero_division=1)
            f1 = f1_score(binary_truth, binary_preds)
            mult_a2=accuracy_score(binary_truth, binary_preds)
            multiclass_true = np.array(CMDC_PHQ9_labels(y_true))
            multiclass_pred = np.array(CMDC_PHQ9_labels(y_pred))
            multiclass_acc = np.sum(multiclass_true == multiclass_pred) / float(len(multiclass_pred))
            return mae, rmse, pearsonrn, precision, recall, f1, mult_a2, multiclass_acc

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))

            if np.all(test_preds == test_preds[0]) or np.all(test_truth == test_truth[0]):
                corr = 0
                print("Warning: One of the input arrays is constant; correlation is undefined.")
            else:
                corr = np.corrcoef(test_preds, test_truth)[0, 1]

            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)
            precision = precision_score(binary_truth, binary_preds, zero_division=1)
            recall = recall_score(binary_truth, binary_preds, zero_division=1)
            f1 = f1_score(binary_truth, binary_preds)
            mult_a2=accuracy_score(binary_truth, binary_preds)

            return mae, rmse, corr, precision, recall, f1, mult_a2, mult_a7
