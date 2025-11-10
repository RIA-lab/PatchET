import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from models.patchet_backbone import PatchIntraBackbone
from transformers import EsmModel, EsmTokenizer
from models.loss_func import WeightedRMSELoss
import torch.nn.functional as F


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


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_inter_heads = config['n_patch_inter_heads']
        self.pretrain_model = EsmModel.from_pretrained(config['pretrain_model'])

        self.patch_intra_conv = nn.Conv1d(640, config['target_window'], kernel_size=config['patch_len'],
                                          stride=config['patch_len'])
        self.patch_intra_layers = PatchIntraBackbone(c_in=int(config['context_window'] / config['patch_len']),
                                                     context_window=config['context_window'],
                                                     target_window=config['target_window'],
                                                     patch_len=config['patch_len'],
                                                     stride=config['patch_len'],
                                                     max_seq_len=config['max_seq_len'],
                                                     n_layers=config['n_layers'],
                                                     d_model=config['d_model'])


        self.patch_inter_conv = nn.Conv1d(config['target_window'], config['target_window'],
                                          kernel_size=2 * config['patch_inter_kernel'] + 1,
                                          padding=config['patch_inter_kernel'])
        self.patch_inter_layers = nn.ModuleList(
            [nn.Conv1d(config['target_window'], config['target_window'],
                       kernel_size=2 * config['patch_inter_kernel'] + 1, padding=config['patch_inter_kernel'])
             for _ in range(config['n_patch_inter_heads'])]
        )

        self.pred_layers = nn.ModuleList(
            [RDBlock(config['target_window'] * 2 * config['n_patch_inter_heads']) for _ in range(config['n_RD'])]
        )
        self.pred_head = nn.Linear(config['target_window'] * 2 * config['n_patch_inter_heads'], 1)

        self.loss_fct = WeightedRMSELoss()
        self._keys_to_ignore_on_save = []
        self.inference = False

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            outputs = self.pretrain_model(input_ids, attention_mask)

        hidden_state = outputs.last_hidden_state  # (B, L, D)
        patch_intra_values = self.patch_intra_conv(hidden_state.transpose(1, 2)) # (B, D', patch_num)
        hidden_state = self.patch_intra_layers(hidden_state) # (B, patch_num, D')
        hidden_state = F.softmax(hidden_state, dim=-1)
        hidden_state = hidden_state * patch_intra_values.transpose(1, 2)

        patch_inter_values = self.patch_inter_conv(hidden_state.transpose(1, 2)) # (B, D', patch_num)
        for i in range(self.patch_inter_heads):
            weights = F.softmax(self.patch_inter_layers[i](hidden_state.transpose(1, 2)), dim=-1)
            x_sum = torch.sum(patch_inter_values * weights, dim=-1)  # (B, D')
            x_max, _ = torch.max(patch_inter_values * weights, dim=-1) # (B, D')
            if i == 0:
                cat_xsum = x_sum
                cat_xmax = x_max
            else:
                cat_xsum = torch.cat([cat_xsum, x_sum], dim=1)
                cat_xmax = torch.cat([cat_xmax, x_max], dim=1)

        hidden_state = torch.cat([cat_xsum, cat_xmax], dim=1)  # Concat features for regression (B, D' * 2 * n_patch_inter_heads)
        for layer in self.pred_layers:
            hidden_state = layer(hidden_state)
        pred = self.pred_head(hidden_state) # (B, 1)
        pred = pred.squeeze(-1)

        if self.inference:
            return ModelOutput(pred=pred, hidden_state=hidden_state)

        loss = self.loss_fct(pred, labels) if labels is not None else None
        return ModelOutput(loss=loss, pred=pred)



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
