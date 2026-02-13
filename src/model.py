import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import ModelConfig, TrainConfig  

class AdaptivePascalBiasGenerator(nn.Module):
    def __init__(self, n: int, max_seq_len: int, bias_warmup_steps: int):
        super().__init__()
        self.n, self.max_seq_len, self.bias_warmup_steps = n, max_seq_len, bias_warmup_steps
        self.skip_penalty_param = nn.Parameter(torch.tensor(-1.5))
        self.pascal_coeffs_map = self._calculate_multinomial_coeffs(n)
        self.base_bias_1_gram = math.log(1 + self.pascal_coeffs_map.get((n, 0, 0), 1))
        self.base_bias_2_gram = math.log(1 + self.pascal_coeffs_map.get((n-1, 1, 0), n))
        self._bias_cache = {}
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))

    def _calculate_multinomial_coeffs(self, n: int) -> dict:
        coeffs, f = {}, math.factorial
        for k1 in range(n + 1):
            for k2 in range(n - k1 + 1):
                k3 = n - k1 - k2
                coeffs[(k1, k2, k3)] = f(n) // (f(k1) * f(k2) * f(k3))
        return coeffs

    def get_adaptive_bias_strength(self) -> torch.Tensor:
        if self.training_step < self.bias_warmup_steps:
            strength = self.training_step.float() / self.bias_warmup_steps
        else:
            strength = torch.tensor(1.0, device=self.training_step.device)
        min_strength = 0.01
        return min_strength + strength * (1.0 - min_strength)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.training: self.training_step += 1
        cache_key = f"{seq_len}_{self.n}"
        if cache_key in self._bias_cache:
            base_bias = self._bias_cache[cache_key].to(device, non_blocking=True)
        else:
            base_bias = self._compute_base_bias_matrix(seq_len, device)
            if len(self._bias_cache) < 20:
                self._bias_cache[cache_key] = base_bias.cpu()
        adaptive_strength = self.get_adaptive_bias_strength()
        penalty_factor = torch.exp(-torch.exp(self.skip_penalty_param))
        final_bias = self._apply_skip_penalty(base_bias.clone(), seq_len, penalty_factor, device)
        return final_bias * adaptive_strength

    def _compute_base_bias_matrix(self, seq_len: int, device: torch.device) -> torch.Tensor:
        bias_values = {}
        window_size = 3
        for i in range(max(0, seq_len - window_size + 1)):
            positions = [i + j for j in range(min(window_size, seq_len - i))]
            for pos in positions:
                if pos < seq_len: bias_values[(pos, pos)] = self.base_bias_1_gram
            for j in range(len(positions)):
                for k in range(j + 1, len(positions)):
                    if positions[j] < seq_len and positions[k] < seq_len:
                        pair = tuple(sorted([positions[j], positions[k]]))
                        bias_values[pair] = self.base_bias_2_gram
        B = torch.zeros(seq_len, seq_len, dtype=torch.float32, device=device)
        for (i, j), bias in bias_values.items():
            B[i, j] = bias
            if i != j: B[j, i] = bias
        return B

    def _apply_skip_penalty(self, B: torch.Tensor, seq_len: int, penalty_factor: float, device: torch.device) -> torch.Tensor:
        if seq_len >= 3:
            skip_bias = self.base_bias_2_gram * penalty_factor
            idx = torch.arange(0, seq_len - 2, device=device)
            B[idx, idx + 2], B[idx + 2, idx] = skip_bias, skip_bias
        return B

class PascalBiasedAttention(nn.Module):
    def __init__(self, config: ModelConfig, bias_generator: AdaptivePascalBiasGenerator):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head, self.n_embd, self.dropout_p = config.n_head, config.n_embd, config.dropout
        self.bias_generator = bias_generator
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + self.bias_generator(T, x.device).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout_p, training=self.training)
        
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)

class Block(nn.Module):
    def __init__(self, config: ModelConfig, bias_generator: AdaptivePascalBiasGenerator):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = PascalBiasedAttention(config, bias_generator)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PascalLanguageModel(nn.Module):
    def __init__(self, m_config: ModelConfig, t_config: TrainConfig):
        super().__init__()
        self.config = m_config
        self.bias_generator = AdaptivePascalBiasGenerator(n=m_config.pascal_n, max_seq_len=m_config.block_size, bias_warmup_steps=t_config.bias_warmup_iters)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(m_config.vocab_size, m_config.n_embd),
            wpe=nn.Embedding(m_config.block_size, m_config.n_embd),
            drop=nn.Dropout(m_config.dropout),
            h=nn.ModuleList([Block(m_config, self.bias_generator) for _ in range(m_config.n_layer)]),
            ln_f=nn.LayerNorm(m_config.n_embd)
        ))
        self.lm_head = nn.Linear(m_config.n_embd, m_config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.register_buffer('pos', torch.arange(m_config.block_size, dtype=torch.long))
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def get_bias_info(self) -> dict:
        return {'bias_strength': self.bias_generator.get_adaptive_bias_strength().item(),
                'skip_penalty': torch.exp(-torch.exp(self.bias_generator.skip_penalty_param)).item(),
                'training_step': self.bias_generator.training_step.item()}
                
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        b, t = idx.size()
        pos = self.pos[:t]
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits, loss = self.lm_head(x[:, [-1], :]), None
        return logits, loss