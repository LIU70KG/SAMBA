import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from transformers import BertModel, BertConfig, AutoModel
from torch.autograd import Function
from scipy.stats import chi2
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model

# =======================================================================================
# 1. 基础工具函数
# =======================================================================================
def to_gpu(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


# =======================================================================================
#  系数预测器 生成 Cr, Cc
# =======================================================================================
class TriCoeffPredictor(nn.Module):
    def __init__(self, dim, dropout=0.2, activation= nn.ReLU(), temp=1.0):
        super().__init__()

        self.r_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        self.c_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
        self.temp = temp

    def forward(self, feat):
        # cr = 1 + torch.tanh(self.r_head(feat))
        # cc = 1 + torch.tanh(self.c_head(feat))
        cr = 1.0 + 1.5 * torch.tanh(self.r_head(feat))    # 放大任务特征  cr的范围是-0.5~2.5
        cc = 1.0 - 1.5 * torch.tanh(self.c_head(feat))   # 协同特征反向！ cc的范围是-0.5~2.5，但反向
        return cr, cc

# =======================================================================================
# 校准模块
# =======================================================================================
class InteractionProbe(nn.Module):
    def __init__(self, dim, dropout=0.2, activation= nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, feat_c, context):
        combined = torch.cat([feat_c, context], dim=-1)
        return self.net(combined)

# =======================================================================================
#  （2）纯数值计算，约束工具 (完整保留 HSIC / CKA)
# =======================================================================================
class QuantifyMetrics(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.epsilon = 1e-6
        self.synergy_threshold = 0.5

    def compute_pairwise_synergy_loss(self, pred_joint, pred_a, pred_b, labels):
        loss_joint = torch.abs(pred_joint - labels)
        loss_a = torch.abs(pred_a - labels)
        loss_b = torch.abs(pred_b - labels)

        min_single_loss = torch.min(loss_a, loss_b)
        diff = loss_joint - min_single_loss + self.synergy_threshold
        loss = torch.relu(diff)
        return loss.mean()

    def synergy_sigle_loss(self, pred_joint, pred_a, pred_b, labels):
        loss_joint = torch.abs(pred_joint - labels)
        loss_a = torch.abs(pred_a - labels)
        loss_b = torch.abs(pred_b - labels)
        sigle_loss = (loss_a.mean() + loss_b.mean() + loss_joint.mean()).mean()
        return sigle_loss

##############################################################################################################
    def unimodal_pred_loss(self, pred_m, labels):
        """EMOE 单模态预测损失 (新增)"""
        return torch.abs(pred_m - labels).mean()

    def compute_modality_importance(self, pred_m, labels):
        """计算模态重要性权重 (新增)"""
        mse = torch.square(pred_m - labels)
        importance = 1.0 / (mse + self.epsilon)
        return importance
###################################################################################################################

# =======================================================================================
# （3）三元校准器 
# =======================================================================================
class TripleCalibrator(nn.Module):
    def __init__(self, config, device, unimodal_heads):
        super().__init__()
        self.config=config
        self.device = device
        self.metrics = QuantifyMetrics(device)

        # 单模态预测头 (EMOE 监督用)
        self.uni_head_t = unimodal_heads['t']
        self.uni_head_v = unimodal_heads['v']
        self.uni_head_a = unimodal_heads['a']

        self.dim=config.sem_dim
        self.dropout=config.dropout
        self.interaction_probe = InteractionProbe(self.dim, dropout=self.dropout).to(device)

    def forward(self, raw_feats, coeffs_map, labels):
        loss_ortho_all = 0.0
        # loss_silence_all = 0.0   
        loss_synergy_gain_all = 0.0
        single_loss_all = 0.0
        loss_uni_all = 0.0
        

        modals = ['t', 'v', 'a']
        I_map = {}

        for m in modals:
            feat = raw_feats[m]
            cr, cc= coeffs_map[m]

            # 1. 互斥约束
            # loss_ortho = torch.mean(torch.abs(torch.sum(cr * cc, dim=-1)))
            loss_ortho = torch.mean(torch.abs(F.cosine_similarity(cr, cc, dim=-1)))
            loss_ortho_all += loss_ortho

            # ====================
            # Cr 分支：EMOE 单模态监督 (已删除 HSIC)
            # ====================
            feat_r = feat* cr
            if m == 't':
                pred_m = self.uni_head_t(feat_r)
            elif m == 'v':
                pred_m = self.uni_head_v(feat_r)
            else:
                pred_m = self.uni_head_a(feat_r)

            loss_uni = self.metrics.unimodal_pred_loss(pred_m, labels)
            loss_uni_all += loss_uni

            I_map[m] = self.metrics.compute_modality_importance(pred_m, labels)



            # # ====================
            # # Cc 分支：静默约束 
            # # ====================
            # feat_c = feat * cc
            # loss_silence = self.metrics.compute_hsic(feat_c, labels)
            # loss_silence_all += loss_silence

        # 协同增益约束
        cc_feats_map = {}
        for m in modals:
            feat = raw_feats[m]
            _, cc = coeffs_map[m]
            cc_feats_map[m] = feat * cc

        pairs = [('t', 'v'), ('t', 'a'), ('v', 'a')]
        zeros = torch.zeros_like(cc_feats_map['t'])
        for (m1, m2) in pairs:
            feat1 = cc_feats_map[m1]
            feat2 = cc_feats_map[m2]
            pred_joint = self.interaction_probe(feat1, feat2)
            pred_single_1 = self.interaction_probe(feat1, zeros)
            pred_single_2 = self.interaction_probe(zeros, feat2)

            single_loss = self.metrics.synergy_sigle_loss(pred_joint, pred_single_1, pred_single_2, labels)
            loss_synergy_gain = self.metrics.compute_pairwise_synergy_loss(pred_joint, pred_single_1, pred_single_2, labels)

            single_loss_all += single_loss
            loss_synergy_gain_all += loss_synergy_gain

        # 返回所有损失，无丢失
        # return (loss_uni_all/3, loss_silence_all/3, loss_synergy_gain_all/3,
        #         single_loss_all/3, loss_ortho_all/3, I_map)
        return (loss_uni_all/3, loss_synergy_gain_all/3,
                single_loss_all/3, loss_ortho_all/3, I_map )

# =======================================================================================
# 3. 自适应精炼器
# =======================================================================================
class AdaptiveRefiner(nn.Module):
    def __init__(self, dim, dropout,activation= nn.ReLU()):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat, coeffs):
        cr, cc = coeffs
        F_in = self.norm(feat)
        w_r = F.softplus(self.alpha)
        w_c = F.softplus(self.beta)
        feat_r = (w_r * cr) * F_in
        feat_c = (w_c * cc) * F_in
        cat_feat = torch.cat([feat_r, feat_c], dim=1)
        gate = self.gate(cat_feat)
        refined_feat = gate * feat_r + (1 - gate) * feat_c
        return refined_feat, cat_feat

# =========================================================
# 样本级弱模态自增强
# =========================================================
class WeakSelfEnhanceBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, activation= nn.ReLU()):
        super().__init__()
        self.sa = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            activation,
            nn.Linear(dim, 1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, other1, other2, qx, q_mean):
        x_in = x.unsqueeze(1)
        x_sa, _ = self.sa(x_in, x_in, x_in)
        x_sa = x_sa.squeeze(1)
        weakness = torch.clamp(q_mean - qx, min=0.0)
        gate_raw = self.gate_mlp(torch.cat([other1, other2], dim=-1))
        # gate = torch.sigmoid(gate_raw) * weakness
        gate = torch.sigmoid(gate_raw) * (weakness + 0.2)
        x_out = (1-gate)*x + gate * x_sa
        return self.norm(x_out)

class UnifiedInteraction(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, activation= nn.ReLU()):
        super().__init__()
        self.enhance_t = WeakSelfEnhanceBlock(dim, num_heads, dropout, activation)
        self.enhance_v = WeakSelfEnhanceBlock(dim, num_heads, dropout, activation)
        self.enhance_a = WeakSelfEnhanceBlock(dim, num_heads, dropout, activation)

    def forward(self, t, v, a, qt, qv, qa):
        qt_mean = 0.33
        qv_mean = 0.33
        qa_mean = 0.33
        t_out = self.enhance_t(t, v, a, qt, qt_mean)
        v_out = self.enhance_v(v, t, a, qv, qv_mean)
        a_out = self.enhance_a(a, t, v, qa, qa_mean)
        return t_out, v_out, a_out

# =======================================================================================
# 预测头
# =======================================================================================
class UniModalPredictor(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
    def forward(self, x):
        return self.net(x)
    
    

class UniversalPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.dense = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.Sigmoid()
        )
        self.res_proj = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim)
        ) if input_dim != hidden_dim else nn.Identity()
        self.out_head = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, x):
        transformed = self.dense(x)
        g = self.gate(x)
        gated_feat = transformed * g
        feat_final = gated_feat + self.res_proj(x)
        return self.out_head(feat_final)
    



# =======================================================================================
# 主模型 SAMBA
# =======================================================================================
class SAMBA(nn.Module):
    def __init__(self, config):
        super(SAMBA, self).__init__()
        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.sem_dim=config.sem_dim

        self.input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = config.num_classes
        self.dropout_rate = config.dropout
        self.num_heads = config.num_heads
        self.activation = self.config.activation()
        self.temp = self.config.temp

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        # ========== 文本特征提取 ==========
        if self.config.use_bert:
            self.bertmodel = DebertaV2Model.from_pretrained('microsoft/deberta-V3-base')
        else:
            self.embed = nn.Embedding(len(config.word2id), self.input_sizes[0])
            self.trnn1 = rnn(self.input_sizes[0], self.hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*self.hidden_sizes[0], self.hidden_sizes[0], bidirectional=True)

        # ========== 视觉 & 声学特征提取 ==========
        self.vrnn1 = rnn(self.input_sizes[1], self.hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*self.hidden_sizes[1], self.hidden_sizes[1], bidirectional=True)
        self.arnn1 = rnn(self.input_sizes[2], self.hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*self.hidden_sizes[2], self.hidden_sizes[2], bidirectional=True)

        # ========== 投影层 ==========
        if self.config.use_bert:
            self.project_t = nn.Sequential(
                nn.Linear(768, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                self.activation
            )
        else:
            self.project_t = nn.Sequential(
                nn.Linear(self.hidden_sizes[0]*4, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                self.activation
            )

        self.project_v = nn.Sequential(
            nn.Linear(self.hidden_sizes[1]*4, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            self.activation
        )
        self.project_a = nn.Sequential(
            nn.Linear(self.hidden_sizes[2]*4, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            self.activation
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tlayer_norm = nn.LayerNorm((self.hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((self.hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((self.hidden_sizes[2]*2,))

        # ========== 核心模块 ==========
        self.coeff_pred_t = TriCoeffPredictor(self.sem_dim, dropout=self.dropout_rate, activation=self.activation, temp=self.temp)
        self.coeff_pred_v = TriCoeffPredictor(self.sem_dim, dropout=self.dropout_rate, activation=self.activation, temp=self.temp)
        self.coeff_pred_a = TriCoeffPredictor(self.sem_dim, dropout=self.dropout_rate, activation=self.activation, temp=self.temp)

        # EMOE 单模态预测头
        self.uni_head_t = UniModalPredictor(input_dim=self.sem_dim).to(self.device)
        self.uni_head_v = UniModalPredictor(input_dim=self.sem_dim).to(self.device)
        self.uni_head_a = UniModalPredictor(input_dim=self.sem_dim).to(self.device)
        

        self.unimodal_heads = nn.ModuleDict({
            't': self.uni_head_t,
            'v': self.uni_head_v,
            'a': self.uni_head_a
        })

        # 校准器
        self.triple_calibrator = TripleCalibrator(config, self.device, self.unimodal_heads)

        # 融合层
        self.fusion1 = nn.Linear(128 * 3, 128)
        self.fusion2 = nn.Sequential(
            nn.LayerNorm(128),
            self.activation,
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.output_size)
        )

        # 精炼器
        self.refiner_t = AdaptiveRefiner(self.sem_dim, dropout=self.dropout_rate, activation=self.activation).to(self.device)
        self.refiner_v = AdaptiveRefiner(self.sem_dim, dropout=self.dropout_rate, activation=self.activation).to(self.device)
        self.refiner_a = AdaptiveRefiner(self.sem_dim, dropout=self.dropout_rate, activation=self.activation).to(self.device)

        # 交互模块
        self.unified_interaction = UnifiedInteraction(self.sem_dim, num_heads=self.num_heads, dropout=self.dropout_rate, activation=self.activation).to(self.device)

        # 权重预测
        self.weight_predictor = nn.Sequential(
            nn.Linear(3 * 2 * self.sem_dim, self.sem_dim),
            nn.LayerNorm(self.sem_dim),
            self.activation,
            nn.Linear(self.sem_dim, 3),
            nn.Softmax(dim=1)
        )

        self.feat_enhance_v = nn.Sequential(nn.LayerNorm(self.hidden_sizes[1]*4), nn.GELU())
        self.feat_enhance_a = nn.Sequential(nn.LayerNorm(self.hidden_sizes[2]*4), nn.GELU())

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        # 🔥 修复：lengths 必须是 1D CPU int64 tensor
        packed_sequence = pack_padded_sequence(sequence, lengths.squeeze().to('cpu'))  # <-- 修复在这里

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.squeeze().to('cpu'))  # <-- 这里也修

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)
        return final_h1, final_h2


    def information_aware_dropout(self, t, v, a, qt, qv, qa, beta=3.0, p_drop=0.3):
        if not self.training:
            return t, v, a
        if torch.rand(1).item() > p_drop:
            return t, v, a

        q = torch.stack([qt.mean(), qv.mean(), qa.mean()])
        prob = torch.softmax(beta * q, dim=0)
        m = torch.multinomial(prob, 1).item()

        if m == 0:
            t = torch.zeros_like(t)
        elif m == 1:
            v = torch.zeros_like(v)
        else:
            a = torch.zeros_like(a)
        return t, v, a

    def forward(self, sentences, visual, acoustic, lengths, bert_sent=None, bert_sent_type=None, bert_sent_mask=None, labels=None, tsne=False):
        batch_size = lengths.size(0)

        # ========== 1. 特征提取 ==========
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)[0]
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
            utterance_text = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
        else:
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)

        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)
        utterance_video = self.feat_enhance_v(utterance_video)  # <-- 增强


        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)
        utterance_audio = self.feat_enhance_a(utterance_audio)  # <-- 增强

        t = self.project_t(utterance_text)
        v = self.project_v(utterance_video)
        a = self.project_a(utterance_audio)


        # ========== 2. 系数预测 ==========
        cr_t, cc_t = self.coeff_pred_t(t)
        cr_v, cc_v = self.coeff_pred_v(v)
        cr_a, cc_a = self.coeff_pred_a(a)
        coeffs_map = {'t':(cr_t,cc_t), 'v':(cr_v,cc_v), 'a':(cr_a,cc_a)}

        # feat_r_t = t * cr_t
        # feat_r_v = v * cr_v
        # feat_r_a = a * cr_a 
        # feat_c_t = t * cc_t
        # feat_c_v = v * cc_v
        # feat_c_a = a * cc_a 

        # ========== 3. 自适应精炼 ==========
        t_refined, cat_t = self.refiner_t(t, coeffs_map['t'])
        v_refined, cat_v = self.refiner_v(v, coeffs_map['v'])
        a_refined, cat_a = self.refiner_a(a, coeffs_map['a'])
        global_descriptors = torch.cat([cat_t, cat_v, cat_a], dim=1)
        modal_weights = self.weight_predictor(global_descriptors)
        q_t_init, q_v_init, q_a_init = modal_weights[:,0:1], modal_weights[:,1:2], modal_weights[:,2:3]

        # ========== 4. 模态丢弃 + 交互 ==========
        t_refined, v_refined, a_refined = self.information_aware_dropout(t_refined, v_refined, a_refined, q_t_init, q_v_init, q_a_init)
        t_enhanced, v_enhanced, a_enhanced = self.unified_interaction(t_refined, v_refined, a_refined, q_t_init, q_v_init, q_a_init)

        # ========== 5. 二次精炼 ==========
        cr_t_new, cc_t_new = self.coeff_pred_t(t_enhanced)
        cr_v_new, cc_v_new = self.coeff_pred_v(v_enhanced)
        cr_a_new, cc_a_new = self.coeff_pred_a(a_enhanced)
        t_rn, cat_tn = self.refiner_t(t_enhanced, (cr_t_new, cc_t_new))
        v_rn, cat_vn = self.refiner_v(v_enhanced, (cr_v_new, cc_v_new))
        a_rn, cat_an = self.refiner_a(a_enhanced, (cr_a_new, cc_a_new))
        global_new = torch.cat([cat_tn, cat_vn, cat_an], dim=1)
        w_final = self.weight_predictor(global_new)
        qt, qv, qa = w_final[:,0:1], w_final[:,1:2], w_final[:,2:3]

        # ========== 6. 融合预测 ==========
        fused = torch.cat([t_enhanced*qt, v_enhanced*qv, a_enhanced*qa], dim=-1)
        fused_tsne = self.fusion1(fused)
        output = self.fusion2(fused_tsne)

        # ========== 7. 损失计算 (所有损失完整返回) ==========
        loss_uni = torch.tensor(0.0, device=self.device)
        # loss_silence = torch.tensor(0.0, device=self.device)
        loss_syn_gain = torch.tensor(0.0, device=self.device)
        loss_single = torch.tensor(0.0, device=self.device)
        loss_orth = torch.tensor(0.0, device=self.device)
        loss_importance = torch.tensor(0.0, device=self.device)


        if self.training and labels is not None:
            raw_feats = {'t':t, 'v':v, 'a':a}
            labels_reg = labels.float().view(-1,1) if labels.dim()==1 else labels.float()

            # 调用校准器，获取所有损失
            # loss_uni, loss_silence, loss_syn_gain, loss_single, loss_orth, I_map = self.triple_calibrator(raw_feats, coeffs_map, labels_reg)
            loss_uni, loss_syn_gain, loss_single, loss_orth, I_map = self.triple_calibrator(raw_feats, coeffs_map, labels_reg)

            # 模态权重对齐损失
            I_tensor = torch.cat([I_map['t'], I_map['v'], I_map['a']], dim=1)
            I_tensor = F.softmax(I_tensor, dim=1)
            w_q_tensor = torch.cat([q_t_init, q_v_init, q_a_init], dim=1)
            cos_sim = F.cosine_similarity(I_tensor, w_q_tensor, dim=1).mean()
            loss_importance = 1.0 - cos_sim


        if tsne==True:
            #3个模态的loss
            # return loss_t, loss_v, loss_a

            # return output, t_refined, v_refined, a_refined, t_enhanced, v_enhanced, a_enhanced, q_t_init, q_v_init, q_a_init, qt, qv, qa

            #####6个子图
            # return output,t,v,a, t_enhanced, v_enhanced, a_enhanced
            # return output, t_refined, v_refined, a_refined, t_enhanced, v_enhanced, a_enhanced,q_t_init, q_v_init, q_a_init, qt, qv, qa

            ###融合特征
            return output, fused_tsne, t_refined, v_refined, a_refined ,t_enhanced, v_enhanced, a_enhanced
        else:
            # 返回所有损失，无丢失
            return output, fused, loss_uni, loss_syn_gain, loss_single, loss_orth, loss_importance
    

