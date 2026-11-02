"""YOUR FILE: prediction heads for L-NER (course rule 9: model design is yours; plumbing only calls this).

Contract used by train_ner.py
-----------------------------
build_head(d_model, n_tags, name, **cfg) -> nn.Module    called with the config's "head" dict as keywords,
                                                           e.g. {"name": "linear"} or {"name": "custom", "hidden": ...}

head.forward(h) -> logits                                  h: (B, T, d_model) backbone states, logits: (B, T, n_tags)

Optional methods (used automatically when present):
head.loss(logits, tags) -> scalar      tags: (B, T) long, -100 = ignore. If absent: (class-weighted) cross-entropy.
                                       Implement this for e.g. a CRF head.
head.decode(logits) -> tag ids         logits: (1, N, n_tags) for one whole document (windows already merged),
                                       returns (1, N) long. If absent: argmax.

Tag ids: 0 = "O", then for each label in labels.txt order: B-, I- (BIO) or B-, I-, E-, S- (BIOES).
The head is always trained (requires_grad=True), whatever the technique.
"""
import torch
import torch.nn as nn
from torchcrf import CRF
class Linear_head(nn.Module):
    def __init__(self,d_model,n_tags,**cfg):
        super().__init__()
        self.in_features = d_model
        self.out_features = n_tags
        self.p = cfg.get("dropout",0.0)
        self.linear = nn.Linear(self.in_features,self.out_features)
        self.dropout = nn.Dropout(p=self.p)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C = x.shape
        x = self.dropout(x)
        logits = self.linear(x)
        return logits

class Mlp_head(nn.Module):
    def __init__(self,d_model,n_tags,**cfg):
        super().__init__()
        self.in_features = d_model
        self.out_features = n_tags
        self.hidden = cfg.get("hidden",256)
        self.activation = nn.GELU()
        self.p = cfg.get("dropout",0.0)
        self.linear1 = nn.Linear(self.in_features,self.hidden)
        self.linear2 = nn.Linear(self.hidden,self.out_features)
        self.dropout = nn.Dropout(p=self.p)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C = x.shape
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits

class Mlp_Crf__head(nn.Module):
    def __init__(self,d_model,n_tags,**cfg):
        super().__init__()
        self.in_features = d_model
        self.out_features = n_tags
        self.hidden = cfg.get("hidden",256)
        self.activation = nn.GELU()
        self.p = cfg.get("dropout",0.0)
        self.linear1 = nn.Linear(self.in_features,self.hidden)
        self.linear2 = nn.Linear(self.hidden,self.out_features)
        self.dropout = nn.Dropout(p=self.p)
        self.crf = CRF(n_tags, batch_first=True)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        B,T,C = x.shape
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits
    def loss(self,logits:torch.Tensor ,tags:torch.Tensor):
        # need to use this logits and find crossentropy error
        mask = tags!=-100
        safe_tags=tags.masked_fill(~mask,0)
        log_likelihood = self.crf(logits,safe_tags,mask=mask,reduction="mean")
        return -log_likelihood
    def decode(self, logits):
        mask = torch.ones(
            logits.shape[:2],
        dtype=torch.bool,
        device=logits.device
    )

        predictions = self.crf.decode(logits, mask=mask)

        return torch.tensor(
        predictions,
        dtype=torch.long,
        device=logits.device
    )






















def build_head(d_model, n_tags, name, **cfg):
    if(name == "linear"):
        return Linear_head(d_model,n_tags,**cfg)
    elif(name ==  "mlp"):
        return Mlp_head(d_model,n_tags,**cfg)
    elif(name == "mlp_crf"):
        return Mlp_Crf__head(d_model,n_tags,**cfg)
    else:
        raise ValueError(f"please provide name of classification head in cfg , availible options : linear , mlp , mlp_crf")
   