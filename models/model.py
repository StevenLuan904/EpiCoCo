from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet

from .common import *
from .diffusion import *
from .diffusion import get_beta_schedule


class AffinityEncoder(nn.Module):

    def __init__(self, d_model=128, n_freq=16, soft_prompt=True):
        super().__init__()
        self.soft_prompt = soft_prompt
        self.n_freq = n_freq
        self.mlp = nn.Sequential(nn.Linear(n_freq * 2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.d_model = d_model

    def forward(self, s):
        # s: [batch_size] continuous score 0-1
        if self.soft_prompt:
            frequencies = torch.pow(2, torch.arange(self.n_freq)).to(s.device)
            angle = s.unsqueeze(-1) * frequencies * np.pi
            f_feats = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
            return self.mlp(f_feats)  # [batch_size, d_model]
        else:
            return torch.zeros(s.size(0), self.d_model).to(s.device)


class MolDiff(Module):

    def __init__(self, config, num_node_types, num_edge_types, **kwargs):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # whether EGNN uses highway residue mode
        # self.residue_mode = getattr(config.denoiser, 'residue_mode', 'highway')
        # self.residue_mode = 'highway'  # 为了 train_pmhc_20251231_010819 的兼容性强制设为 highway

        self.soft_prompt = getattr(config, 'soft_prompt', True)

        # define beta and alpha
        self.define_betas_alphas(config.diff)

        # 维度定义
        node_dim = config.node_dim  # 256
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        esm_dim = getattr(config, 'esm_dim', 320)

        # [修改 1]: 定义两路特征的维度
        # 要求左右两路各占 node_dim 的一半
        self.half_dim = node_dim // 2  # 128
        # ============================================================
        # 左路: Base Feature (Node + Entity)
        # 目标维度: half_dim (128)
        # ============================================================
        self.node_embedder = nn.Linear(num_node_types, self.half_dim, bias=False)
        self.entity_embedder = nn.Embedding(2, self.half_dim)

        self.use_self_cond = getattr(config, 'self_condition', False)
        if self.use_self_cond:
            # 假设我们利用上一轮预测的 node_type (logits) 作为 condition
            # 维度从 num_node_types 映射回 half_dim
            self.self_cond_projector = nn.Linear(num_node_types, self.half_dim, bias=False)
            nn.init.normal_(self.self_cond_projector.weight, std=0.02)

        # ============================================================
        # 右路: Context Feature (Time + ESM + ESM_AVG + IC50)
        # 目标维度: half_dim (128)
        # 输入维度: time_dim + esm_dim + ic50_dim
        # ============================================================
        # Peptide ESM 占位符
        self.peptide_esm_placeholder = nn.Parameter(torch.randn(esm_dim))
        nn.init.normal_(self.peptide_esm_placeholder, std=0.02)

        # Time Embedding 模块
        self.time_emb = nn.Sequential(GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim,
                                                       type_='linear'),)

        # 定义 IC50 嵌入模块
        self.ic50_encoder = AffinityEncoder(d_model=self.half_dim, n_freq=16, soft_prompt=self.soft_prompt)

        # time_dim: 时间步编码
        # esm_dim * 2: MHC的ESM特征 + 平均池化特征
        # self.half_dim: IC50/Affinity 的 Fourier Embedding 维度
        input_dim = time_dim + esm_dim * 2 + self.half_dim
        hidden_dim = input_dim // 2

        self.time_esm_projector = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
                                                nn.Linear(hidden_dim, self.half_dim), nn.LayerNorm(self.half_dim),
                                                nn.SiLU())

        self.edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False)
        self.edge_projector = nn.Linear(edge_dim + time_dim, edge_dim)
        self.time_emb = nn.Sequential(GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim,
                                                       type_='linear'),)

        # # denoiser
        if config.denoiser.backbone == 'NodeEdgeNet':
            self.denoiser = NodeEdgeNet(node_dim, edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.node_decoder = MLP(node_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)

        self.init_with_real_mhc = getattr(config, 'init_with_real_mhc', True)

    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')

        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])

        # # diffusion for pos
        pos_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_pos)
        if self.categorical_space == 'continuous':
            scaling_node = self.scaling[0]  # should be 1.0
            self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_atom)
        if self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.num_node_types, scaling_node)
        # # diffusion for edge type
        edge_betas = get_beta_schedule(num_timesteps=self.num_timesteps, **config.diff_bond)
        if self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat([time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt

    def add_noise(self,
                  node_type,
                  node_pos,
                  batch_node,
                  halfedge_type,
                  halfedge_index,
                  batch_halfedge,
                  num_mol,
                  t,
                  entity=None,
                  halfedge_entity=None,
                  esm=None,
                  esm_avg=None,
                  **kwargs):
        num_graphs = num_mol
        device = node_pos.device

        time_step = t * torch.ones(num_graphs, device=device).long()

        # 2.1 perturb pos, node, edge
        # node perturbation only on peptide (entity==1)
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)

        if self.categorical_space == 'discrete':
            h_node_pert, log_node_t, log_node_0 = node_pert
            h_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            h_node_pert, h_node_0 = node_pert
            h_halfedge_pert, h_halfedge_0 = halfedge_pert
        return [h_node_pert, pos_pert, h_halfedge_pert]

    def get_loss(self,
                 node_type,
                 node_pos,
                 batch_node,
                 halfedge_type,
                 halfedge_index,
                 batch_halfedge,
                 num_mol,
                 entity=None,
                 halfedge_entity=None,
                 ic50=None,
                 esm=None,
                 esm_avg=None):
        num_graphs = num_mol
        device = node_pos.device

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2.1 perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(node_pos, time_step, batch_node)
        node_pert = self.node_transition.add_noise(node_type, time_step, batch_node)
        # 这里处理的依据是 entity!=0，所以能适用 bond_type
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, batch_halfedge)
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        h_node_pert, h_node_0 = node_pert
        h_halfedge_pert, h_halfedge_0 = halfedge_pert
        h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

        # Self-Conditioning
        if self.use_self_cond:
            # 初始化为全0 (对应 unconditional 的情况)
            # 形状需要匹配 forward 中 self_condition 的期望输入 (num_node_types)
            node_self_cond = torch.zeros(h_node_pert.size(0), self.num_node_types, device=device)

            # 训练时 50% 概率启用 self-condition，且仅当 enabled 时计算 estimate
            if self.training and torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    # 第一次前向：传入全0的 condition 获取猜测值
                    prev_out = self(h_node_pert,
                                    pos_pert,
                                    batch_node,
                                    h_edge_pert,
                                    edge_index,
                                    batch_edge,
                                    time_step,
                                    entity,
                                    halfedge_entity,
                                    ic50,
                                    esm,
                                    esm_avg,
                                    self_condition=node_self_cond)
                    # 将预测的 pred_node (logits) detach 后作为下一次的 condition
                    node_self_cond = prev_out['pred_node'].detach()

        # 3. forward to denoise
        preds = self(h_node_pert,
                     pos_pert,
                     batch_node,
                     h_edge_pert,
                     edge_index,
                     batch_edge,
                     time_step,
                     entity=entity,
                     halfedge_entity=halfedge_entity,
                     ic50=ic50,
                     esm=esm,
                     esm_avg=esm_avg)
        pred_node = preds['pred_node']
        pred_pos = preds['pred_pos']
        pred_halfedge = preds['pred_halfedge']

        # 4. loss
        # 4.1 pos
        loss_pos = F.mse_loss(pred_pos, node_pos)
        loss_len = 0
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(node_pos[bond_index[0]] - node_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_pos[bond_index[0]] - pred_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)

        # continuous
        # 4.2 node type
        loss_node = F.mse_loss(pred_node, h_node_0) * 30
        # 4.3 edge type
        loss_edge = F.mse_loss(pred_halfedge, h_halfedge_0) * 30

        # total
        loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)

        loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len
        return loss_dict

    def forward(self,
                h_node_pert,
                pos_pert,
                batch_node,
                h_edge_pert,
                edge_index,
                batch_edge,
                t,
                entity,
                halfedge_entity,
                ic50=None,
                esm=None,
                esm_avg=None,
                self_condition=None):
        """
        Predict Mol at step `0` given perturbed Mol at step `t`
        """

        # ============================================================
        # 分支 1: 左路 - Base Feature (Node + Entity) -> [N, 128]
        # ============================================================
        node_feat = self.node_embedder(h_node_pert)
        entity_feat = self.entity_embedder(entity.long())

        # 要求的逻辑: h_base = node_feat + entity_feat
        h_base = node_feat + entity_feat

        # ============================================================
        # 分支 2: 右路 - Context Feature (Time + ESM + ESM_AVG + IC50) -> [N, 128] + (optional) Self-Conditioning
        # ============================================================

        # 1. 获取 Time Embedding [N, time_dim]
        time_embed_node = self.time_emb(t.index_select(0, batch_node))

        # 2. 准备 ESM 特征 [N, 320]
        # 使用占位符填充 Peptide，使用真实 ESM 填充 MHC
        esm_full = self.peptide_esm_placeholder.unsqueeze(0).expand(h_base.size(0), -1).clone()
        if esm is not None and esm_avg is not None:
            mask_mhc = (entity == 0)
            esm_full[mask_mhc] = esm.to(esm_full.dtype)
            # [关键修改]：扩展 esm_avg
            # batch_node 是一个索引向量 [0,0,0, ..., 1,1, ..., 4,4]
            # esm_avg 是 [5, 320]
            # 使用 batch_node 进行索引，得到 [Total_Nodes, 320]
            esm_avg_expanded = esm_avg[batch_node]

        # 3. 处理 IC50 特征
        ic50_normalized = torch.clamp(1 - torch.log10(ic50) / torch.log10(torch.tensor(50000.0)), min=0.0, max=1.0)
        ic50_embedding = self.ic50_encoder(ic50_normalized)
        ic50_embedding = ic50_embedding[batch_node]

        # 4. 拼接 Time、ESM 和 IC50 -> [N, time_dim + 320 + ic50_dim]
        time_esm_raw = torch.cat([time_embed_node, esm_full, esm_avg_expanded, ic50_embedding], dim=-1)

        # 5. 投影到 half_dim -> [N, 128]
        h_context = self.time_esm_projector(time_esm_raw)

        # 6. self conditioning
        if self.use_self_cond and self_condition is not None:
            # 将上一把预测的 logits 投影并加到 base feature 上
            # 这里的 self_condition 形状应为 [N, num_node_types]
            h_context = h_context + self.self_cond_projector(self_condition)

        # ============================================================
        # 融合: Concatenation -> [N, 256]
        # h_base (128) + h_context (128) = h_node_input (256)
        # ============================================================
        h_node_input = torch.cat([h_base, h_context], dim=-1)

        # ============================================================
        # Edge Feature
        # edge_dim + time_dim -> edge_dim
        # ============================================================
        time_embed_edge = self.time_emb(t.index_select(0, batch_edge))
        h_edge_input = torch.cat([self.edge_embedder(h_edge_pert), time_embed_edge], dim=-1)
        h_edge_input = self.edge_projector(h_edge_input)

        # 2 diffuse to get the updated node embedding and bond embedding
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_input,
            pos_node=pos_pert,
            h_edge=h_edge_input,
            edge_index=edge_index,
            node_time=t.index_select(0, batch_node).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, batch_edge).unsqueeze(-1) / self.num_timesteps,
            entity=entity,
            halfedge_entity=halfedge_entity)

        n_halfedges = h_edge.shape[0] // 2
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges] + h_edge[n_halfedges:])
        pred_pos = pos_node

        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
        }

    @torch.no_grad()
    def sample(
            self,
            n_graphs,
            batch_node,
            halfedge_index,
            halfedge_type,
            batch_halfedge,
            node_index,
            node_pos,
            entity,
            halfedge_entity,
            esm=None,
            esm_avg=None,
            jump_len=10,
            resample_time=10,
            start_step=990,
            is_resample=True,
            scaffold=None,
            # [新增参数: Bipolar Manifold Steering]
            guidance_scale=1.5,  # w: 引导强度 (推荐 1.0 ~ 3.0)
            target_ic50=1.0,  # [关键] 正样本目标 IC50 (1.0 nM -> Score ~1.0)
            null_ic50=50000.0,  # [关键] 负样本目标 IC50 (50000 nM -> Score ~0.1, 也是归一化的分母)
    ):
        device = batch_node.device

        # ----------------------------------------------------------------------
        # 0. 准备工作：Condition Tensors & Graph Topology
        # ----------------------------------------------------------------------
        # 构造正负样本的 Condition 输入 (假设 forward 会处理 Fourier Embedding 和 Normalize)
        ic_pos_tensor = torch.full((n_graphs,), target_ic50, dtype=torch.float32).to(device)
        ic_neg_tensor = torch.full((n_graphs,), null_ic50, dtype=torch.float32).to(device)

        # 预先构建完整的无向图边索引 (Cat flip)，避免在循环中重复计算
        # 注意：这里假设 model forward 需要完整的 edge_index
        full_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        full_batch_edge = torch.cat([batch_halfedge, batch_halfedge], dim=0)
        if self.use_self_cond:
            # 初始时刻没有预测值，设为全 0
            current_self_cond = torch.zeros(len(batch_node), self.num_node_types, device=device)
        # ----------------------------------------------------------------------
        # 1. 内部核心函数：双极流形操纵 (Bipolar Manifold Steering)
        # ----------------------------------------------------------------------
        def _infer_step(curr_h_node, curr_pos, curr_h_edge, curr_t_tensor):
            """
            执行核心采样步骤 (带缓存)：
            文件命名格式: pos_step_{t}.pt 和 neg_step_{t}.pt
            """
            # 获取当前时间步 (假设 curr_t_tensor 是 [t, t, ...]，取第一个值作为 ID)
            step_idx = curr_t_tensor[0].item()

            out_pos = None
            out_neg = None

            # -----------------------------------------------------------
            # B. 计算与写入缓存
            # -----------------------------------------------------------
            # --- Pass 1: Positive ---
            if out_pos is None:
                out_pos = self(curr_h_node,
                               curr_pos,
                               batch_node,
                               curr_h_edge,
                               full_edge_index,
                               full_batch_edge,
                               curr_t_tensor,
                               entity,
                               halfedge_entity,
                               esm=esm,
                               esm_avg=esm_avg,
                               ic50=ic_pos_tensor,
                               self_condition=current_self_cond if self.use_self_cond else None)

            if guidance_scale <= 0.0:
                return out_pos['pred_node'], out_pos['pred_pos'], out_pos['pred_halfedge']

            # --- Pass 2: Negative ---
            if out_neg is None:
                out_neg = self(curr_h_node,
                               curr_pos,
                               batch_node,
                               curr_h_edge,
                               full_edge_index,
                               full_batch_edge,
                               curr_t_tensor,
                               entity,
                               halfedge_entity,
                               esm=esm,
                               esm_avg=esm_avg,
                               ic50=ic_neg_tensor,
                               self_condition=current_self_cond if self.use_self_cond else None)

            # -----------------------------------------------------------
            # C. 几何引导 (Geometric Steering)
            # -----------------------------------------------------------
            # 这一步通常计算很快，不需要缓存，每次实时算即可
            steered_node = out_pos['pred_node'] + guidance_scale * (out_pos['pred_node'] - out_neg['pred_node'])
            steered_pos = out_pos['pred_pos'] + guidance_scale * (out_pos['pred_pos'] - out_neg['pred_pos'])
            steered_edge = out_pos['pred_halfedge'] + guidance_scale * (out_pos['pred_halfedge'] -
                                                                        out_neg['pred_halfedge'])

            return steered_node, steered_pos, steered_edge

        # ----------------------------------------------------------------------
        # 2. 初始化噪声状态
        # 如果 init_with_real_mhc 为 True，则不初始化 MHC 部分为噪声
        # ----------------------------------------------------------------------
        n_nodes_all = len(batch_node)
        n_halfedges_all = len(batch_halfedge)
        if self.init_with_real_mhc:
            # entity == 0 表示 MHC，保持真实结构
            mhc_mask = (entity == 0)
            node_init = self.node_transition.sample_init(n_nodes_all)  # one-hot noise
            pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
            node_init[mhc_mask] = F.one_hot(node_index[mhc_mask], num_classes=self.num_node_types).float()
            pos_init[mhc_mask] = node_pos[mhc_mask]
        else:
            node_init = self.node_transition.sample_init(n_nodes_all)  # one-hot noise
            pos_init = self.pos_transition.sample_init([n_nodes_all, 3])
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)

        # 轨迹记录容器
        h_node_init = node_init
        h_halfedge_init = halfedge_init
        node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, h_node_init.shape[-1]],
                                dtype=h_node_init.dtype).to(device)
        pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, 3], dtype=pos_init.dtype).to(device)
        halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_all, h_halfedge_init.shape[-1]],
                                    dtype=h_halfedge_init.dtype).to(device)

        node_traj[0] = h_node_init
        pos_traj[0] = pos_init
        halfedge_traj[0] = h_halfedge_init

        # 当前状态变量
        h_node_pert = h_node_init
        pos_pert = pos_init
        h_halfedge_pert = h_halfedge_init

        # ----------------------------------------------------------------------
        # 3. 反向扩散主循环 (Reverse Diffusion Loop)
        # ----------------------------------------------------------------------
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps, desc="Sampling"):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            h_edge_pert = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)
            pred_node, pred_pos, pred_halfedge = _infer_step(h_node_pert, pos_pert, h_edge_pert, time_step)

            # 更新 Self-Conditioning 信息
            if self.use_self_cond:
                current_self_cond = pred_node

            # 计算上一步 t-1 的后验状态 (Posterior Sampling)
            pos_prev = self.pos_transition.get_prev_from_recon(x_t=pos_pert,
                                                               x_recon=pred_pos,
                                                               t=time_step,
                                                               batch=batch_node)

            if self.categorical_space == 'continuous':
                h_node_prev = self.node_transition.get_prev_from_recon(x_t=h_node_pert,
                                                                       x_recon=pred_node,
                                                                       t=time_step,
                                                                       batch=batch_node)
                h_halfedge_prev = self.edge_transition.get_prev_from_recon(x_t=h_halfedge_pert,
                                                                           x_recon=pred_halfedge,
                                                                           t=time_step,
                                                                           batch=batch_halfedge)
            else:
                # 预留给 Discrete Diffusion 的接口
                h_node_prev = pred_node
                h_halfedge_prev = pred_halfedge

            # 记录轨迹
            node_traj[i + 1] = h_node_prev
            pos_traj[i + 1] = pos_prev
            halfedge_traj[i + 1] = h_halfedge_prev

            # 更新当前状态
            pos_pert = pos_prev
            h_node_pert = h_node_prev
            h_halfedge_pert = h_halfedge_prev

            # ------------------------------------------------------------------
            # 4. RePaint / Resampling (时间回溯以增强一致性)
            # ------------------------------------------------------------------
            if is_resample and step < start_step and step % jump_len == 0:
                for _ in range(resample_time):
                    # --- A. 加噪 (Diffusion Forward: t -> s) ---
                    # s = t + jump_len
                    time_t = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
                    time_s = torch.full(size=(n_graphs,), fill_value=step + jump_len, dtype=torch.long).to(device)

                    pos_pert = self.pos_transition.noise_to_step_s(pos_pert, time_t, time_s, batch_node)
                    h_node_pert = self.node_transition.noise_to_step_s(h_node_pert, time_t, time_s, batch_node)
                    h_halfedge_pert = self.edge_transition.noise_to_step_s(h_halfedge_pert, time_t, time_s,
                                                                           batch_halfedge)

                    # --- B. 去噪修正 (Denoise: s -> t) ---
                    for j in range(step, step + jump_len)[::-1]:
                        time_step_j = torch.full(size=(n_graphs,), fill_value=j, dtype=torch.long).to(device)
                        h_edge_pert_j = torch.cat([h_halfedge_pert, h_halfedge_pert], dim=0)

                        # >>>>> 再次调用核心引导推理 (保持引导一致性) <<<<<
                        pred_node_res, pred_pos_res, pred_halfedge_res = _infer_step(
                            h_node_pert, pos_pert, h_edge_pert_j, time_step_j)

                        # --- C. Scaffold Inpainting (强制约束) ---
                        if scaffold:
                            # 为 scaffold 目标生成当前时刻 j 的噪声
                            noised_scaffold_pos = self.pos_transition.add_noise(scaffold.node_pos, time_step_j,
                                                                                scaffold.batch_node)
                            noised_scaffold_node = self.node_transition.add_noise(scaffold.node_type, time_step_j,
                                                                                  scaffold.batch_node)[0]
                            noised_scaffold_halfedge = self.edge_transition.add_noise(
                                scaffold.halfedge_type, time_step_j, scaffold.batch_halfedge)[0]

                            # 强制覆盖模型预测的 x_0 (或 x_recon)
                            # 注意：这里覆盖的是预测的"原图"，随后 transition 会基于这个混合的"原图"算出下一步
                            pred_node_res[scaffold.scaffold_node_mask] = noised_scaffold_node
                            pred_pos_res[scaffold.scaffold_node_mask] = noised_scaffold_pos
                            pred_halfedge_res[scaffold.scaffold_halfedge_mask] = noised_scaffold_halfedge

                        # Step Backward
                        pos_prev_res = self.pos_transition.get_prev_from_recon(x_t=pos_pert,
                                                                               x_recon=pred_pos_res,
                                                                               t=time_step_j,
                                                                               batch=batch_node)

                        if self.categorical_space == 'continuous':
                            h_node_prev_res = self.node_transition.get_prev_from_recon(x_t=h_node_pert,
                                                                                       x_recon=pred_node_res,
                                                                                       t=time_step_j,
                                                                                       batch=batch_node)
                            h_halfedge_prev_res = self.edge_transition.get_prev_from_recon(x_t=h_halfedge_pert,
                                                                                           x_recon=pred_halfedge_res,
                                                                                           t=time_step_j,
                                                                                           batch=batch_halfedge)
                        else:
                            h_node_prev_res = pred_node_res
                            h_halfedge_prev_res = pred_halfedge_res

                        # 更新状态
                        pos_pert = pos_prev_res
                        h_node_pert = h_node_prev_res
                        h_halfedge_pert = h_halfedge_prev_res

        # ----------------------------------------------------------------------
        # 5. 返回结果
        # ----------------------------------------------------------------------
        # 最后一步如果是 Inpainting 任务，理论上可以用 scaffold 的真实值覆盖
        # 但通常保留 Diffusion 的输出以保持物理平滑性
        return {
            'pred': [pred_node, pred_pos, pred_halfedge],  # 最终去噪结果 (x_0 estimate)
            'traj': [node_traj, pos_traj, halfedge_traj],  # 完整轨迹
        }
