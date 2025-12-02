from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math
import numpy as np

from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv
from .dynamics import NeuralDynamicsModule, DynamicsFeatureProjector, DynamicsGateProjector

from libs.generalized_coordinates import get_derivatives

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)
        self.dropout = nn.Dropout(drop_out) #0.05 dropout

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1
        
    def forward(self, queries, keys, values, mask):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
        values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2) #（n,head,t,c）

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values) # （n,head,t,c）,（n,head,t,c）->(n,head,c,c)
        Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6) #（n,head,t,c）,（n,head,c） ->(n,head,t)

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) #(n,head,c,c),(n,head,t,c),(n,head,t)->(n,head,t,c)

        x = x.reshape(B, L, -1) #4 head to （n,t,c）
        x = self.out_projection(x)
        x = self.dropout(x) #0.05的dropout

        return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            ) #膨胀卷积
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))
        self.conv_fusion = nn.Conv1d(2 * n_features, n_features, 1)
        
    def forward(self, feature_s, feature_t, mask): #feature_s （n,t,v) feature_t (n,t,c)
        n, c, t, v = feature_s.shape
        feature_s = feature_s.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)  # (n,8,t,v) -->(n,v*8,t)
        feature_s = self.conv_s(feature_s) #(n,v,t)->(n,c,t)
        # map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t)
        # feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)
        # feature_cross = feature_cross + feature_t
        feature_cross = self.conv_fusion(torch.cat([feature_t, feature_s], dim=1))
        feature_cross = feature_cross.permute(0, 2, 1) #(n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask


class DynamicFeatureFusion(nn.Module):
    def __init__(self, in_channel, n_features, dilation):
        super().__init__()
        self.conv_in = nn.Conv1d(3, 3, 1, groups=3)  # 19->64
        self.conv_dilated = nn.Conv1d(3, 3, 3, padding=1, groups=3)
        self.sigmoid = nn.Sigmoid()
        self.conv_out = nn.Conv1d(n_features * 3, n_features, 1)

    def forward(self, feature_st, feature_d, mask):  # feature_s （n,t,v) feature_t (n,t,c)
        # N, D, T = feature_d.shape

        feature_d = self.conv_in(feature_d)
        feature_d = self.conv_dilated(feature_d)
        feature_d = self.sigmoid(feature_d)

        gated_feature_st1 = feature_d[:,:1,:] * feature_st
        gated_feature_st2 = feature_d[:,1:2,:] * feature_st
        gated_feature_st3 = feature_d[:,2:,:] * feature_st

        gated_feature_st = torch.cat([gated_feature_st1, gated_feature_st2, gated_feature_st3], dim=1)

        gated_feature_st = self.conv_out(gated_feature_st) + feature_st

        return gated_feature_st * mask, feature_d * mask




    
class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer #（1,2,3,4,5,6,7,8,9）
        num_SFI_layers = len(SFI_layer) #9
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate) #0.3 dropout

        self.conv_in = nn.Conv2d(in_channel, (num_SFI_layers+1)*8, kernel_size=1)
        self.conv_t = nn.Conv1d(node * 8, n_features, 1)
        self.SFI_layers = nn.ModuleList(
            [SFI(node * 8, n_features) for i in range(num_SFI_layers)])
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in 
                range(num_layers)]) #10层扩张注意力
        self.dynamic_gates = nn.ModuleList(
            [DynamicFeatureFusion(n_features, n_features, 2 ** i) for i in range(num_layers)]) #10
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feature_d, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        x = self.conv_in(x) #c=64->80
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers)*8, 8), dim=1)

        N, C, T ,V = feature_t.shape
        feature_t = feature_t.permute(0, 3, 1, 2).contiguous().view(N, V*C, T) #(n,8,t,v) -->(n,v*8,t)
        feature_st = self.conv_t(feature_t) #(n,v*8,t)->(n,64,t)

        for index, (layer, dynamic_gate) in enumerate(zip(self.layers, self.dynamic_gates)):
            if index in self.SFI_layer:
                feature_st =  self.SFI_layers[count](feature_s[:,count*8:(count+1)*8,:], feature_st, mask)
                count+=1
            feature_st = layer(feature_st, None, mask)
            feature_st, feature_d = dynamic_gate(feature_st, feature_d, mask) # temporal modulation

        feature_st = self.conv_out(feature_st)
        return feature_st * mask
       
class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature


    
class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int],
        n_stages_brb: Optional[int],
        SFI_layer: Optional[int],
        dataset: str,
        node: int,
        dof: int,
        num_people: int,
        use_Friction: bool,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()


        self.in_channel = in_channel

        self.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / 0.07))  # 2.6593

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset, node)

        # stream B：the dynamic model
        self.dynamics_stream = NeuralDynamicsModule(dof, use_Friction)
        self.dynamics_projector = DynamicsFeatureProjector(dof * num_people, n_features, node)

        self.dyn_gate_projector = DynamicsGateProjector(dof * num_people, n_features, node)


        self.STI = STI(node, n_features * 2, n_features, n_features, n_layers, SFI_layer)
 
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)
        self.conv_feature = nn.Conv1d(n_features, 768, 1)

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features, 768, 1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor, q: torch.Tensor, mask: torch.Tensor, joint_graph) -> Tuple[torch.Tensor, torch.Tensor]:

        N, C, V, T = x.shape

        F_s0 = self.SP(x, joint_graph) * mask.unsqueeze(3) #（n,c,t,v）

        q, q_dot, q_ddot = get_derivatives(q, 1)  # (N_m, T, d)

        dyn_outputs = self.dynamics_stream(q, q_dot, q_ddot)
        tau_hat = dyn_outputs["tau_hat"]  # (N_m, T, d)

        if q.shape[0] != N:
            _, T, D =  q.shape
            tau_hat = tau_hat.view(N, -1, T, D)
            q_dot = q_dot.view(N, -1, T, D)

            # 2. Concatenate along the feature dimension 'd'  (N, 2, T, d) -> (N, T, 2*d)
            tau_hat = tau_hat.permute(0, 2, 1, 3).contiguous().view(N, T, 2 * D)
            q_dot = q_dot.permute(0, 2, 1, 3).contiguous().view(N, T, 2 * D)

        F_dyn = self.dynamics_projector(tau_hat)  # (N_m, C_dyn, T, V)

        Dyn_gate = self.dyn_gate_projector(tau_hat, q_dot)

        # --- spatial modulation ---
        F_fused = torch.cat([F_s0, F_dyn], dim=1)

        feature = self.STI(F_fused, Dyn_gate, mask)
        
        out_cls = self.conv_cls(feature)
        out_bound = self.conv_bound(feature)
        out_feature = self.conv_feature(feature)
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature = as_stage(self.activation_asb(out_cls) * mask, feature * mask, mask)
                out_feature = conv_stage(feature)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature, dyn_outputs, self.logit_scale)

        else: # eval
            for as_stage in self.asb:
                out_cls, feature = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound)
