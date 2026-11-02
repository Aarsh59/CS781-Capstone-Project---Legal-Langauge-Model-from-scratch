from dataclasses import dataclass 
import torch  
import torch.nn as nn  
from encoder import EncBlock, sinusoidal_positional_encoding  
from decoder4 import DecBlock

@dataclass
class EncLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"   # "sinusoidal" or "rope"
    attention_scale: float = None      # preserve Q1.4 hook
    norm_first: bool = False


@dataclass
class DecLMConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    pos_emb_type: str = "sinusoidal"   # "sinusoidal" or "rope"
    attention_scale: float = None      # preserve Q1.4 hook
    norm_first: bool = False


class EncDecLM(nn.Module):
    def __init__(self,cfg1:EncLMConfig,cfg2:DecLMConfig):
        super().__init__()
        # Encoder Block Initializations 
        self.cfg1 = cfg1
        self.tok_emb1 = nn.Embedding(cfg1.vocab_size, cfg1.d_model)
        if cfg1.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg1.max_seq_len, cfg1.d_model)
            self.register_buffer("pos_emb_enc", pe, persistent=False)
   
        self.encblocks = nn.ModuleList([EncBlock(cfg1) for _ in range(cfg1.n_layers)])
        self.ln_f_enc = nn.LayerNorm(cfg1.d_model)
        # Decoder Block Initializations
        self.cfg2 = cfg2
        self.tok_emb2 = nn.Embedding(cfg2.vocab_size, cfg2.d_model)
        if cfg2.pos_emb_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(cfg2.max_seq_len, cfg2.d_model)
            self.register_buffer("pos_emb_dec", pe, persistent=False)
       
        self.decblocks = nn.ModuleList([DecBlock(cfg2) for _ in range(cfg2.n_layers)])
        self.ln_f_dec = nn.LayerNorm(cfg2.d_model)
        self.head = nn.Linear(cfg2.d_model, cfg2.vocab_size, bias=False)
        self.head.weight = self.tok_emb2.weight
    def encode(self,src:torch.Tensor)->torch.Tensor:
        # passing it through encoder 
        B,S = src.shape
        
        x_enc = self.tok_emb1(src)
        if(self.cfg1.pos_emb_type == "sinusoidal"):
            x_enc=x_enc+self.pos_emb_enc[:S]
        for block in self.encblocks:
            x_enc = block(x_enc)
        x_enc = self.ln_f_enc(x_enc)
        memory = x_enc
        return memory

    def decode(self,tgt:torch.Tensor,memory:torch.Tensor)->torch.Tensor:
        B,T = tgt.shape 
        x_dec = self.tok_emb2(tgt)
        if(self.cfg2.pos_emb_type == "sinusoidal"):
            x_dec=x_dec+self.pos_emb_dec[:T]
        for block in self.decblocks:
            x_dec = block(x_dec,memory)
        x_dec = self.ln_f_dec(x_dec)
        x_dec = self.head(x_dec)
        return x_dec


    def forward(self ,src:torch.Tensor , tgt:torch.Tensor)->torch.Tensor:
        return self.decode(tgt,self.encode(src))
        
         
        






        
    
        
