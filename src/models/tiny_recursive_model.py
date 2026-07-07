import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from types import SimpleNamespace


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DropPath(nn.Module):
    """Stochastic depth / LayerDrop."""
    def __init__(self):
        super().__init__()

    def forward(self, x, drop_prob: float = 0.0):
        if drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - drop_prob
        if x.dim() == 4:
            shape = [x.shape[0], 1, 1, 1]
        else:
            shape = [x.shape[0]] + [1] * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor


class AdapterExpert(nn.Module):
    """Single adapter expert for task routing in Mixture of Adapters."""
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, dim // 4)
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.up = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.dropout(self.act(self.down(x))))


class MixtureOfAdapters(nn.Module):
    """Mixture-of-Adapters routing module with lightweight expert selection."""
    def __init__(self, dim, num_experts=4, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, dim // 4)
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            AdapterExpert(dim, hidden_dim, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.router = nn.Linear(dim, num_experts, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        # x: [B, T, dim]
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return (outputs * weights.unsqueeze(-2)).sum(dim=-1)


class NumericalProjector(nn.Module):
    """Embed auxiliary numeric values directly into the model latent space."""
    def __init__(self, dim, hidden_dim=None, input_dim=1):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, dim // 4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        numeric = self.net(x)
        return numeric * torch.sigmoid(self.gate(numeric))


class NumericalFusion(nn.Module):
    """Fuse numeric embeddings with the latent reasoning state."""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim if hidden_dim is not None else max(1, dim // 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Linear(dim, dim, bias=False)

    def forward(self, z, num_emb):
        fused = self.net(torch.cat([z, num_emb], dim=-1))
        return z + torch.sigmoid(self.gate(z)) * fused


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())

    def forward(self, x):
        seq_len = x.shape[1]
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # Original cos/sin shape: [seq_len, head_dim]
    # q/k shape: [batch_size, n_heads, seq_len, head_dim]
    # We need cos/sin to be [1, 1, seq_len, head_dim] for proper broadcasting
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function with optional dropout"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional GQA"""
    def __init__(self, dim, n_heads, n_kv_heads=None, max_seq_len=512, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        assert self.n_kv_heads <= n_heads
        assert dim % self.n_kv_heads == 0
        self.use_grouped_q = self.n_kv_heads != self.n_heads

        if self.use_grouped_q:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.kv_proj = nn.Linear(dim, 2 * self.n_kv_heads * self.head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

        self.attn_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        if self.use_grouped_q:
            q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2, self.head_dim)
            k, v = kv.unbind(dim=-2)
            k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1, -1)
            v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1, -1)
            k = k.reshape(B, self.n_heads, T, self.head_dim)
            v = v.reshape(B, self.n_heads, T, self.head_dim)
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=-1)
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""
    def __init__(
        self,
        dim,
        n_heads,
        mlp_ratio=4,
        max_seq_len=512,
        dropout=0.0,
        is_adapter_layer=False,
        adapter_dropout=0.0,
        alpha=1.0,
        beta=1.0,
        rezero_init=False,
        n_kv_heads=None,
        ffn_multiplier=None,
        use_task_adapter=True,
    ):
        super().__init__()
        self.beta = beta
        self.rezero_init = rezero_init
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads=n_kv_heads, max_seq_len=max_seq_len, dropout=dropout)
        self.norm2 = RMSNorm(dim)
        hidden_dim = int(ffn_multiplier * dim) if ffn_multiplier is not None else int((8.0 / 3.0) * dim)
        self.mlp = SwiGLU(dim, hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.is_adapter_layer = is_adapter_layer and use_task_adapter
        self.adapter = MixtureOfAdapters(dim, num_experts=4, hidden_dim=max(1, dim // 4), dropout=adapter_dropout) if self.is_adapter_layer else None
        self.adapter_scalar = nn.Parameter(torch.zeros(1)) if self.is_adapter_layer else None

        if self.rezero_init:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        self.drop_path = DropPath()

    def forward(self, x, drop_prob: float = 0.0):
        residual = x
        x = self.norm1(x)
        x = self.attn(x) * self.beta
        x = self.dropout(x)
        x = residual + self.alpha * x
        x = self.drop_path(x, drop_prob)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x) * self.beta
        x = self.dropout(x)
        x = residual + self.alpha * x
        x = self.drop_path(x, drop_prob)

        if self.is_adapter_layer:
            x = x + self.adapter_scalar * self.adapter(x)
        return x


# ============================================================================
# Tiny Recursive Model
# ============================================================================

class TinyRecursiveNetwork(nn.Module):
    """
    The core tiny network used in TRM.
    This supports depth scaling, interleaved adapters, DeepNorm, LayerDrop, and optional checkpointing.
    """
    def __init__(
        self,
        dim,
        n_heads=8,
        n_layers=2,
        mlp_ratio=4,
        max_seq_len=512,
        dropout=0.0,
        adapter_dropout=0.0,
        adapter_every_k=2,
        n_kv_heads=None,
        ffn_multiplier=None,
        use_task_adapter=True,
        use_checkpoint=False,
    ):
        super().__init__()
        assert n_layers > 0, "n_layers must be positive"
        self.n_layers = n_layers
        self.adapter_every_k = adapter_every_k
        self.use_checkpoint = use_checkpoint

        deepnorm_alpha = (2 * n_layers) ** 0.25
        deepnorm_beta = (2 * n_layers) ** -0.25

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim,
                n_heads,
                mlp_ratio,
                max_seq_len,
                dropout=dropout,
                is_adapter_layer=(layer_idx % adapter_every_k == 0),
                adapter_dropout=adapter_dropout,
                alpha=deepnorm_alpha,
                beta=deepnorm_beta,
                rezero_init=(layer_idx % 2 == 1),
                n_kv_heads=n_kv_heads,
                ffn_multiplier=ffn_multiplier,
                use_task_adapter=use_task_adapter,
            )
            for layer_idx in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

    def forward(self, x):
        for layer_idx, layer in enumerate(self.layers):
            p = 0.15 * layer_idx / max(1, self.n_layers - 1)
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, drop_prob=p)
            else:
                x = layer(x, drop_prob=p)
        return self.norm(x)


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for Text Generation

    Architecture based on TRM paper:
    - Single tiny 2-layer network
    - Recursive reasoning with latent z and prediction y
    - Deep supervision across multiple improvement steps

    For text generation:
    - x: embedded input sequence (context)
    - y: current token predictions (embedded)
    - z: latent reasoning state

    The model recursively improves its latent z, then updates y.
    """
    def __init__(
        self,
        vocab_size,
        dim=768,
        n_heads=12,
        n_kv_heads=None,
        n_layers=4,
        mlp_ratio=4,
        ffn_multiplier=None,
        max_seq_len=2048,
        n_latent_recursions=2,  # reduced loops for performance
        n_improvement_cycles=1,  # lighter improvement cycle
        dropout=0.0,
        adapter_dropout=0.0,
        adapter_every_k=2,
        numerical_input_dim=1,
        use_task_adapter=False,
        use_checkpoint=False,
        use_less_is_more=False,
        ema_decay=0.0,
        tie_embeddings=False,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_latent_recursions = n_latent_recursions
        self.n_improvement_cycles = n_improvement_cycles
        self.use_checkpoint = use_checkpoint
        self.use_less_is_more = use_less_is_more

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.emb_dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()

        self.numerical_input_dim = numerical_input_dim
        self.num_projector = NumericalProjector(dim, input_dim=numerical_input_dim)
        self.num_fusion = NumericalFusion(dim)

        # Single tiny network (key insight: one network is better than two)
        effective_n_layers = n_layers * 2
        self.net = TinyRecursiveNetwork(
            dim,
            n_heads,
            effective_n_layers,
            mlp_ratio,
            max_seq_len,
            dropout=dropout,
            adapter_dropout=adapter_dropout,
            adapter_every_k=adapter_every_k,
            n_kv_heads=n_kv_heads,
            ffn_multiplier=ffn_multiplier,
            use_task_adapter=use_task_adapter,
            use_checkpoint=use_checkpoint,
        )

        # Projection layers for combining x, y, z
        self.combine_xyz = nn.Linear(dim * 3, dim, bias=False)
        self.combine_yz = nn.Linear(dim * 2, dim, bias=False)

        # Output head
        self.output_head = nn.Linear(dim, vocab_size, bias=False)

        # Halting head for ACT (simplified - no Q-learning)
        self.halt_head = nn.Linear(dim, 1, bias=False)

        # Learnable initial states for y and z
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_gate = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(1, n_latent_recursions))))
        self.ema_decay = ema_decay

        self.tie_embeddings = tie_embeddings

        self._init_weights()

        # Optionally tie embedding and output weights (helps sample efficiency)
        if self.tie_embeddings:
            try:
                self.output_head.weight = self.token_emb.weight
            except Exception:
                pass

    def _init_weights(self):
        init_std = 0.02 / math.sqrt(max(1, self.net.n_layers))
        output_std = 0.02 / math.sqrt(self.dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.output_head or module is self.halt_head:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=output_std)
                else:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, TransformerBlock) and module.beta != 1.0:
                if hasattr(module.attn, 'proj'):
                    module.attn.proj.weight.data.mul_(module.beta)
                if hasattr(module.attn, 'q_proj'):
                    module.attn.q_proj.weight.data.mul_(module.beta)
                    module.attn.kv_proj.weight.data.mul_(module.beta)
                module.mlp.w2.weight.data.mul_(module.beta)
                module.mlp.w3.weight.data.mul_(module.beta)

    def get_embeddings(self, input_ids, attention_mask=None):
        """Get token + position embeddings, accounting for left-padding"""
        B, T = input_ids.shape
        # Clamp input_ids to valid range
        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        # Clamp position to max_seq_len
        T = min(T, self.max_seq_len)
        
        # Calculate true positions considering left padding
        if attention_mask is not None:
            # e.g., mask = [0, 0, 1, 1], cumsum = [0, 0, 1, 2] -> pos = [0, 0, 0, 1]
            pos = torch.cumsum(attention_mask[:, :T], dim=1) - 1
            pos = pos.clamp(min=0)
        else:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            
        return self.emb_dropout(self.token_emb(input_ids[:, :T]) + self.pos_emb(pos))

    def get_depth_scaled_optimizer_groups(self, base_lr, decay_rate=0.85):
        """Create optimizer groups with layer-wise learning rate decay and adapter/full lr exceptions."""
        layer_ids = []
        for name, module in self.named_modules():
            if isinstance(module, TransformerBlock):
                match = re.search(r'net\.layers\.(\d+)', name)
                if match:
                    layer_ids.append(int(match.group(1)))
        max_layer = max(layer_ids) if layer_ids else 0

        groups = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                'adapter' in name
                or 'combine_' in name
                or 'output_head' in name
                or 'halt_head' in name
                or 'token_emb' in name
                or 'pos_emb' in name
            ):
                lr = base_lr
            else:
                match = re.search(r'net\.layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    depth_from_top = max_layer - layer_idx
                    lr = base_lr * (decay_rate ** depth_from_top)
                else:
                    lr = base_lr

            groups.setdefault(lr, []).append(param)

        return [{'params': params, 'lr': lr} for lr, params in groups.items()]

    def latent_recursion(self, x, y, z, num_emb=None):
        """
        Single recursion cycle:
        1. Update z given (x, y, z)
        2. Update y once given (y, z)
        """
        z_prev = z
        if self.use_less_is_more:
            combined = self.combine_xyz(torch.cat([x, y, z], dim=-1))
            delta_z = self.net(combined)
            z = z + self.z_gate * delta_z
            if num_emb is not None:
                z = self.num_fusion(z, num_emb)
        else:
            for _ in range(self.n_latent_recursions):
                combined = self.combine_xyz(torch.cat([x, y, z], dim=-1))
                delta_z = self.net(combined)
                z = z + self.z_gate * delta_z
                if num_emb is not None:
                    z = self.num_fusion(z, num_emb)

        combined_yz = self.combine_yz(torch.cat([y, z], dim=-1))
        y = self.net(combined_yz)

        if self.ema_decay > 0.0:
            z = self.ema_decay * z_prev + (1.0 - self.ema_decay) * z

        return y, z

    def deep_recursion(self, x, y, z, use_grad=True, num_emb=None):
        """
        Deep recursion with T improvement cycles.
        First T-1 cycles without gradients, last cycle with gradients.
        """
        if not use_grad:
            with torch.no_grad():
                for _ in range(self.n_improvement_cycles):
                    y, z = self.latent_recursion(x, y, z, num_emb=num_emb)
            return y.detach(), z.detach()

        with torch.no_grad():
            for _ in range(self.n_improvement_cycles - 1):
                y, z = self.latent_recursion(x, y, z, num_emb=num_emb)

        y, z = self.latent_recursion(x, y, z, num_emb=num_emb)
        return y.detach(), z.detach(), self.output_head(y), self.halt_head(y.mean(dim=1))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        targets=None,
        n_supervision_steps=4,
        numerical_values=None,
        numerical_mask=None,
        **kwargs,
    ):
        """
        Forward pass with deep supervision.

        Args:
            input_ids: [B, T] input token IDs
            attention_mask: optional [B, T] attention mask
            targets: [B, T] target token IDs (for training)
            numerical_values: optional numeric features [B, T, F]
            numerical_mask: optional numeric mask [B, T]
            n_supervision_steps: number of deep supervision steps

        Returns:
            If training: loss
            If inference: object with logits
        """
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T].clamp(0, self.vocab_size - 1)

        x = self.get_embeddings(input_ids, attention_mask=attention_mask)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :T].unsqueeze(-1).to(dtype=x.dtype, device=x.device)
            x = x * attention_mask

        num_emb = None
        if numerical_values is not None:
            num_emb = self.num_projector(numerical_values[:, :T])
            if numerical_mask is not None:
                numerical_mask = numerical_mask[:, :T].unsqueeze(-1).to(dtype=num_emb.dtype, device=num_emb.device)
                num_emb = num_emb * numerical_mask
            x = x + num_emb

        y = self.y_init.expand(B, T, -1).clone().to(dtype=x.dtype, device=x.device)
        z = self.z_init.expand(B, T, -1).clone().to(dtype=x.dtype, device=x.device)

        if targets is None:
            y, z = self.deep_recursion(x, y, z, use_grad=False, num_emb=num_emb)
            return SimpleNamespace(logits=self.output_head(y))

        targets = targets[:, :T].clamp(0, self.vocab_size - 1)

        total_loss = 0.0
        for step in range(n_supervision_steps):
            y, z, logits, halt_logit = self.deep_recursion(x, y, z, use_grad=True, num_emb=num_emb)

            ce_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100
            )

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (targets != -100)
                correct = ((preds == targets) & mask).float().sum() / mask.float().sum().clamp(min=1)
            halt_loss = F.binary_cross_entropy_with_logits(
                halt_logit.squeeze(-1),
                correct.expand(B)
            )

            total_loss = total_loss + ce_loss + 0.1 * halt_loss

        return total_loss / n_supervision_steps

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40,
        pad_token_id=None,
        do_sample=False,
        stopping_criteria=None,
        **kwargs,
    ):
        """Generate text autoregressively with HuggingFace-style args."""
        self.eval()

        if max_length is None:
            max_length = input_ids.shape[1] + max_new_tokens
        max_length = min(max_length, self.max_seq_len)

        for _ in range(max_length - input_ids.shape[1]):
            # Crop to max_seq_len - 1 to leave room for prediction
            idx_cond = input_ids[:, -(self.max_seq_len - 1):]
            if attention_mask is not None:
                mask_cond = attention_mask[:, -(self.max_seq_len - 1):]
            else:
                mask_cond = None

            # Clamp input ids to valid vocab range
            idx_cond = idx_cond.clamp(0, self.vocab_size - 1)

            # Get predictions
            output = self(idx_cond, attention_mask=mask_cond)
            logits = output.logits if hasattr(output, 'logits') else output
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            if stopping_criteria is not None:
                stop = stopping_criteria(input_ids, logits)
                if isinstance(stop, torch.Tensor):
                    if stop.numel() == 1:
                        stop = bool(stop.item())
                    else:
                        stop = stop.all().item()
                if stop:
                    break

        return input_ids
