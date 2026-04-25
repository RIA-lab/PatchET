import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from models.patchex_backbone import IntraPatchBackbone, InterPatchBackbone
from transformers import EsmModel, EsmTokenizer
from models.loss_func import WeightedRMSELoss
import torch.nn.functional as F
from torch import Tensor
from math import sqrt
from .patchet_pretrained import PatchETPretrained
import yaml
from safetensors import safe_open


def load_config(config_path):
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs

def load_weight(model, checkpoint_path, strict=False):
    state_dict = {}
    with safe_open(checkpoint_path, 'pt') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    model.load_state_dict(state_dict, strict=strict)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

class PredLayers(nn.Module):
    def __init__(self, patch_dim, n_patch_inter_heads, n_RD):
        super(PredLayers, self).__init__()
        self.pred_layers = nn.ModuleList(
            [RDBlock(patch_dim * 2 * n_patch_inter_heads) for _ in range(n_RD)]
        )


    def forward(self, z: Tensor) -> Tensor:  # z: [B x patch_num x D']
        for layer in self.pred_layers:
            z = layer(z)
        return z


class RDBlock(nn.Module):
    '''A dense layer with residual connection'''

    def __init__(self, dim):
        super(RDBlock, self).__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(self, x):
        x0 = x
        x = F.leaky_relu(self.dense(x))
        x = x0 + x
        return x


class PatchPredHead(nn.Module):
    """
    Advanced per-patch prediction head.
    - local_feats: [B, P, D]
    - global_feats: [B, D, P]  (we permute to [B, P, D])
    Returns:
    - patch_preds: [B, P]  (scalar predicted temp per patch)
    - att_weights: [B, n_heads, P, P]  (optional attention weights)
    """
    def __init__(self, dim, n_heads=4, hidden_dim=128, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # linear projections for attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # gating to mix local and attended context
        self.gate_proj = nn.Linear(dim * 2, dim)

        # final MLP to scalar per-patch prediction
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, local_feats, global_feats, return_attn=False):
        # local_feats: [B, P, D]
        # global_feats: [B, D, P] -> permute to [B, P, D]
        B, P, D = local_feats.shape
        # print(local_feats.shape)
        # print(global_feats.shape)
        assert D == self.dim, f"local dim {D} != expected {self.dim}"
        gf = global_feats.permute(0, 2, 1).contiguous()  # [B, P, D]

        # Project Q, K, V
        Q = self.q_proj(local_feats)  # [B, P, D]
        K = self.k_proj(gf)           # [B, P, D]
        V = self.v_proj(gf)           # [B, P, D]

        # reshape for heads: [B, P, n_heads, head_dim] -> [B, n_heads, P, head_dim]
        def split_heads(x):
            x = x.view(B, P, self.n_heads, self.head_dim)
            return x.permute(0, 2, 1, 3)  # [B, n_heads, P, head_dim]

        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # scaled dot-product attention (each patch attends over all patches' global features)
        # att_scores: [B, n_heads, P, P]
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context: [B, n_heads, P, head_dim] -> concat heads -> [B, P, D]
        context_h = torch.matmul(attn, Vh)
        context = context_h.permute(0, 2, 1, 3).contiguous().view(B, P, D)
        context = self.out_proj(context)  # [B, P, D]

        # gating between local and context: gate in [0,1] per element
        cat = torch.cat([local_feats, context], dim=-1)  # [B, P, 2D]
        gate = torch.sigmoid(self.gate_proj(cat))        # [B, P, D]
        fused = gate * context + (1.0 - gate) * local_feats  # [B, P, D]

        # final scalar per patch: use both local and fused (or cat local+fused)
        final = torch.cat([local_feats, fused], dim=-1)  # [B, P, 2D]
        patch_preds = self.mlp(final).squeeze(-1)        # [B, P]

        if return_attn:
            return patch_preds, attn
        return patch_preds


class PatchET(nn.Module):
    def __init__(self, config):
        """
        config must include:
          - context_window, patch_len, patch_dim, max_seq_len, n_layers, d_model,
          - patch_inter_kernel, n_patch_inter_heads, n_RD
          - optional: patch_pred_hidden (int), patch_pred_heads (int)
        """
        super().__init__()
        self.intra_patch_backbone = IntraPatchBackbone(
            c_in=int(config['context_window'] / config['patch_len']),
            context_window=config['context_window'],
            output_dim=config['patch_dim'],
            patch_len=config['patch_len'],
            stride=config['patch_len'],
            max_seq_len=config['max_seq_len'],
            n_layers=config['n_layers'],
            d_model=config['d_model']
        )

        # inter_patch_backbone should return (inter_hidden_state, global_feature)
        # where:
        #   inter_hidden_state: [B, patch_num, patch_dim * 2 * n_patch_inter_heads] (or similar)
        #   global_feature: [B, patch_dim, patch_num]   (patch-aware global features)
        self.inter_patch_backbone = InterPatchBackbone(
            patch_dim=config['patch_dim'],
            patch_inter_kernel=config['patch_inter_kernel'],
            n_patch_inter_heads=config['n_patch_inter_heads']
        )

        # prediction residual layers for inter-patch hidden
        self.pred_layers = PredLayers(
            patch_dim=config['patch_dim'],
            n_patch_inter_heads=config['n_patch_inter_heads'],
            n_RD=config['n_RD']
        )

        # global per-patch prediction head (unchanged)
        pred_dim = config['patch_dim'] * 2 * config['n_patch_inter_heads']
        self.pred_head = nn.Linear(pred_dim, 1)

        # new: attention-based patch prediction head that uses intra_patch slice + global_feature
        patch_pred_hidden = config.get('patch_pred_hidden', 128)
        patch_pred_heads = config.get('patch_pred_heads', max(1, config['n_patch_inter_heads'] // 1))
        self.patch_pred_head = PatchPredHead(
            dim=config['patch_dim'],
            n_heads=patch_pred_heads,
            hidden_dim=patch_pred_hidden,
            dropout=config.get('patch_pred_dropout', 0.1)
        )

    def forward(self, hidden_state: Tensor):
        """
        Inputs:
          hidden_state: ESM features, [B, L, D_esm]  (the input to IntraPatchBackbone)
        Returns:
          global_pred: [B, P, 1]      (per-patch scalar from pred_head on inter features)
          patch_preds: [B, P, 1]      (per-patch scalar from patch_pred_head using local + global_feature)
          inter_hidden: [B, P, pred_dim]  (features after pred_layers, useful for debugging/analysis)
          intra_hidden_state: [B, P, D']  (raw intra patch features)
          global_feature: [B, D', P]    (returned by inter_patch_backbone)
        """
        # intra patch features: [B, P, D']
        intra_hidden_state = self.intra_patch_backbone(hidden_state)

        # inter backbone returns inter-hidden and global patch-aware feature
        # expected shapes:
        #   inter_hidden_state: [B, P, some_dim]  (where some_dim == pred_dim before pred_layers)
        #   global_feature: [B, D', P]
        inter_hidden_state, global_feature = self.inter_patch_backbone(intra_hidden_state)

        # process inter_hidden_state through residual prediction layers
        inter_hidden_state = self.pred_layers(inter_hidden_state)  # [B, P, pred_dim]

        # global per-patch prediction from inter features (old pred_head behavior)
        global_preds = self.pred_head(inter_hidden_state)  # [B, 1]

        # per-patch prediction using intra patch slice + patch-aware global_feature
        # patch_pred_head expects local_feats [B, P, D'] and global_feats [B, D', P]
        patch_preds = self.patch_pred_head(intra_hidden_state, global_feature)  # [B, P, 1]

        # return both, aggregation into the final scalar prediction is done at Model.forward
        return global_preds, patch_preds, inter_hidden_state, intra_hidden_state, global_feature


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pretrained_patchet = False
        if config['pretrained_patchet'] is not None:
            patchet_pretrained_config = load_config(f'{config["pretrained_patchet"]}/model_config.yaml')
            self.patchet = PatchETPretrained(patchet_pretrained_config)
            load_weight(self.patchet, f'{config["pretrained_patchet"]}/model.safetensors')
            self.patchet.inference = True
            freeze_model(self.patchet)
            self.patch_pred_head = PatchPredHead(dim=patchet_pretrained_config.get('target_window', 128),
                                             n_heads=4,
                                             hidden_dim=patchet_pretrained_config.get('target_window', 128),
                                             dropout=0.1)
            self.pretrained_patchet = True
        else:
            self.pretrain_model = EsmModel.from_pretrained(config['pretrain_model'])
            self.patchet = PatchET(config)

        # control flags
        self.inference = False

        # gating / patch loss hyperparams (tunable via config)
        self.threshold = config.get('patch_gate_threshold', 5.0)        # units same as label
        self.lambda_patch = config.get('lambda_patch', 0.1)
        self.gating_mode = config.get('gating_mode', 'soft')           # 'hard' or 'soft'
        self.soft_tau = config.get('gating_soft_tau', 1.0)
        self.patch_loss_type = config.get('patch_loss_type', 'mse')    # 'mse' or 'huber'
        self.reduction_eps = 1e-8

        # optional consistency loss weight (encourage global_pred ~ weighted patch aggregation)
        self.lambda_consistency = config.get('lambda_consistency', 0.0)

        # Focal-style regression hyperparams
        self.focal_gamma = float(config.get('focal_gamma', 0.0))
        self.focal_clip = float(config.get('focal_clip', 10.0))
        self.focal_on_patch = bool(config.get('focal_on_patch', False))

        # place to store diagnostics from last forward for external logging
        self.last_forward_info = {}
        self._keys_to_ignore_on_save = ["pretrain_model.*"]

    def _compute_joint_loss(self, global_pred, patch_preds, labels, patch_mask):
        """
        Compute total loss = main global loss + gated patch loss (normalized) * lambda_patch

        global_pred: [B,1]
        patch_preds: [B,P,1]
        labels: [B,1]
        patch_mask: [B,P] bool tensor (True = valid patch)

        returns: total_loss (scalar tensor), diagnostics dict (tensor values)
        """
        device = global_pred.device
        B = patch_preds.shape[0]
        P = patch_preds.shape[1]

        # reshape
        gp = global_pred.view(B)          # [B]
        pp = patch_preds.squeeze(-1)      # [B,P]
        y = labels.view(B)                # [B]

        # --- FOCAL WEIGHTS (computed from absolute errors, detached for stability) ---
        eps = 1e-8
        if self.focal_gamma > 0.0:
            abs_err = torch.abs(gp - y).detach()  # [B]
            mean_err = abs_err.mean() + eps
            focal_weights = (abs_err / mean_err) ** self.focal_gamma  # [B]
            if self.focal_clip and self.focal_clip > 0:
                focal_weights = torch.clamp(focal_weights, max=self.focal_clip)
            focal_weights = focal_weights / (focal_weights.mean() + eps)
        else:
            focal_weights = torch.ones(B, device=device)

        # 1) main loss on global prediction (RMSE-like)
        per_sample_mse = (gp - y).pow(2)  # [B]
        weighted_mse_mean = (focal_weights * per_sample_mse).mean()
        loss_main = torch.sqrt(weighted_mse_mean + eps)

        # 2) per-element squared error [B,P] for patches
        if self.patch_loss_type == 'mse':
            per_patch_sq = (pp - y.unsqueeze(1)).pow(2)      # [B,P]
        elif self.patch_loss_type == 'huber':
            per_patch_sq = F.smooth_l1_loss(pp, y.unsqueeze(1).expand_as(pp), reduction='none')  # [B,P]
        else:
            raise ValueError("patch_loss_type must be 'mse' or 'huber'")

        # compute per-sample patch loss as mean over *valid* patches only
        valid_mask_f = patch_mask.float()                       # [B,P] float
        valid_counts = valid_mask_f.sum(dim=1)                  # [B]
        per_sample_patch_loss = (per_patch_sq * valid_mask_f).sum(dim=1) / (valid_counts + eps)  # [B]

        # optionally modulate per-sample patch loss by focal weights
        if self.focal_on_patch and self.focal_gamma > 0.0:
            patch_weights_for_samples = focal_weights  # [B]
        else:
            patch_weights_for_samples = torch.ones(B, device=device)

        # 3) gating based on global error
        global_error = torch.abs(gp - y)  # [B]

        if self.gating_mode == 'hard':
            gate = (global_error < self.threshold).float()  # [B]
        elif self.gating_mode == 'soft':
            gate = torch.sigmoid((self.threshold - global_error) / (self.soft_tau + 1e-12))  # [B]
        else:
            raise ValueError("gating_mode must be 'hard' or 'soft'")

        gate_weighted = gate * patch_weights_for_samples  # [B]

        gate_sum = gate_weighted.sum()
        if gate_sum.item() > 0:
            patch_loss = (gate_weighted * per_sample_patch_loss).sum() / (gate_sum + self.reduction_eps)
        else:
            patch_loss = torch.tensor(0.0, device=device, dtype=loss_main.dtype)

        # 4) optional consistency loss: compute weights only over valid patches
        if self.lambda_consistency > 0:
            errors = torch.abs(pp - y.unsqueeze(1))  # [B,P]
            tau_for_weights = self.config.get('patch_tau', 0.1)
            scores = -errors / (tau_for_weights + 1e-12)           # [B,P]
            scores = scores.masked_fill(~patch_mask, -1e9)
            weights = torch.softmax(scores, dim=1)                # [B,P]
            weighted_agg = torch.sum(weights * pp, dim=1, keepdim=True)  # [B,1]
            loss_cons = F.mse_loss(global_pred, weighted_agg)
        else:
            loss_cons = torch.tensor(0.0, device=device, dtype=loss_main.dtype)

        total_loss = loss_main + self.lambda_patch * patch_loss + self.lambda_consistency * loss_cons

        # diagnostics as tensors (kept internal; not returned by forward)
        diag = {
            'loss_main': loss_main.detach(),
            'patch_loss': patch_loss.detach(),
            'loss_consistency': loss_cons.detach(),
            'gate_mean': gate.mean().detach(),
            'global_error_mean': global_error.mean().detach(),
            'per_sample_patch_loss_mean': per_sample_patch_loss.mean().detach(),
            'valid_patch_frac_mean': (valid_counts.float() / float(P)).mean().detach(),
            'focal_gamma': torch.tensor(float(self.focal_gamma), device=device),
            'focal_weight_mean': focal_weights.mean().detach()
        }

        return total_loss, diag

    def forward(self, input_ids, attention_mask, labels=None):
        if self.pretrained_patchet:
            outputs = self.patchet(input_ids, attention_mask, labels=None)
            intra_hidden_state = outputs.intra_hidden_state  # [B, P, D]
            global_hidden_state = outputs.global_hidden_state  # [B, D, P]
            global_pred = outputs.pred
            patch_preds = self.patch_pred_head(intra_hidden_state, global_hidden_state)
        else:
            # 1) get ESM features (frozen)
            with torch.no_grad():
                outputs = self.pretrain_model(input_ids, attention_mask)
            hidden_state = outputs.last_hidden_state  # (B, L, D_esm)

            # 2) run PatchET
            # PatchET must return: global_pred [B,1], patch_preds [B,P,1], and optionally other features
            global_pred, patch_preds, inter_hidden_state, intra_hidden_state, global_feature = self.patchet(
                hidden_state)

        # shapes
        patch_preds_s = patch_preds.squeeze(-1)  # [B,P]
        global_pred_s = global_pred.view(-1, 1)  # [B,1]

        # ---------- compute patch_mask from attention_mask (respect padding) ----------
        device = attention_mask.device
        seq_lens = attention_mask.sum(dim=1).long()        # [B], number of real tokens per example
        patch_len = int(self.config['patch_len'])
        valid_patch_nums = ((seq_lens + patch_len - 1) // patch_len).clamp(min=0)  # [B]

        B = patch_preds.shape[0]
        P = patch_preds.shape[1]
        valid_patch_nums = torch.minimum(valid_patch_nums, torch.tensor(P, device=valid_patch_nums.device))

        idx = torch.arange(P, device=valid_patch_nums.device).unsqueeze(0)  # [1, P]
        patch_mask = idx < valid_patch_nums.unsqueeze(1)                   # [B, P], True=valid

        # Training mode: labels provided -> compute gated joint loss
        if labels is not None and not self.inference:
            total_loss, diag = self._compute_joint_loss(global_pred_s, patch_preds, labels, patch_mask)

            # Build patch_weights (for logging/diagnostics) using errors vs true label
            labels_flat = labels.view(-1)  # [B]
            errors = torch.abs(patch_preds_s - labels_flat.unsqueeze(1))  # [B,P]
            tau_for_weights = self.config.get('patch_tau', 0.1)
            scores = -errors / (tau_for_weights + 1e-12)
            scores = scores.masked_fill(~patch_mask, -1e9)
            weights = torch.softmax(scores, dim=1)  # [B,P]
            weights = weights * patch_mask.float()
            patch_weights = weights.unsqueeze(-1)  # [B,P,1]

            # store diagnostics for external logging (move tensors to cpu to avoid GPU retention)
            self.last_forward_info = {k: (v.detach().cpu().item() if torch.is_tensor(v) and v.numel()==1 else v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                                       for k, v in diag.items()}
            # optionally store small arrays like patch_weights? avoid storing large tensors by default
            self.last_forward_info['patch_weights_shape'] = patch_weights.shape

            # Return only loss and final prediction (clean for HF Trainer)
            global_pred_s = global_pred_s.squeeze(-1)  # [B]
            return ModelOutput(loss=total_loss, pred=global_pred_s)

        # Inference mode (or labels is None): compute weights w.r.t global_pred
        else:
            global_pred_flat = global_pred_s.view(-1)  # [B]
            errors_inf = torch.abs(patch_preds_s - global_pred_flat.unsqueeze(1))  # [B,P]
            tau_for_weights = self.config.get('patch_tau', 0.1)
            scores_inf = -errors_inf / (tau_for_weights + 1e-12)
            scores_inf = scores_inf.masked_fill(~patch_mask, -1e9)
            weights_inf = torch.softmax(scores_inf, dim=1)  # [B,P]
            weights_inf = weights_inf * patch_mask.float()
            weights_inf_exp = weights_inf  # [B,P]

            aggregated_from_patches = torch.sum(weights_inf * patch_preds_s, dim=1, keepdim=True)

            # store diagnostics for external use (cpu)
            diag_inf = {
                'aggregated_from_patches_mean': aggregated_from_patches.mean().detach(),
                'valid_patch_frac_mean': (valid_patch_nums.float() / float(P)).mean().detach(),
            }
            self.last_forward_info = {k: (v.detach().cpu().item() if torch.is_tensor(v) and v.numel()==1 else v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                                      for k, v in diag_inf.items()}
            self.last_forward_info['patch_weights_shape'] = weights_inf_exp.shape

            # Return only final prediction
            return ModelOutput(pred=global_pred_s, patch_weights=weights_inf)



class Collator:
    def __init__(self, pretrain_model):
        self.tokenizer = EsmTokenizer.from_pretrained(pretrain_model)

    def __call__(self, batch):
        seqs = [_.sequence for _ in batch]
        labels = [_.label for _ in batch]
        labels = torch.tensor(labels).float()
        inputs = self.tokenizer(list(seqs), return_tensors="pt", padding='max_length', truncation=True, max_length=1000)
        inputs['labels'] = labels
        return inputs
