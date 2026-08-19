import jax
import jax.numpy as jnp
from flax import linen as nn

from wassersteinflowmatching.autoregressivewasserstein.DefaultConfig import ARWFMConfig


class FeedForward(nn.Module):
    """
    Transformer MLP / feed-forward block.
    (Pre-norm compatible version)
    """
    hidden_dim: int
    dropout_rate: float
    @nn.compact
    def __call__(self, inputs, deterministic: bool = True, dropout_rng=None):
        # Use mlp_hidden_dim from your config
        x = nn.Dense(features=self.hidden_dim)(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic, rng=dropout_rng)
        x = nn.leaky_relu(x) 
        output = nn.Dense(inputs.shape[-1])(x)
        return output
    
class ContextEncoderBlock(nn.Module):
    """Single Transformer encoder block.  Mask shape and semantics are the
    caller's responsibility — supports [B, 1, N, N] causal masks."""
    embedding_dim: int
    num_heads: int
    mlp_hidden_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, mask=None, dropout_rng = None, deterministic=True):
        attn_mask = mask

        # Pre-norm self-attention
        attn_rng, ff_rng = jax.random.split(dropout_rng) if dropout_rng is not None else (None, None)
        #normed_inputs = SetLayerNorm()(conditioned_inputs, mask=masks)
        normed_inputs = nn.LayerNorm()(x)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(
            normed_inputs, 
            mask=attn_mask, 
            deterministic=deterministic,
            dropout_rng=attn_rng
        )
        
        x = x + attn_output
        #normed_x = SetLayerNorm()(x, mask=masks)
        normed_x = nn.LayerNorm()(x)
        ff_output = FeedForward(hidden_dim=self.mlp_hidden_dim, dropout_rate=self.dropout_rate)(normed_x, deterministic=deterministic, dropout_rng=ff_rng)
        output = x + ff_output
        
        return output
    

class CausalContextEncoder(nn.Module):
    """Causal set encoder c_phi for AR training.

    Processes all N target points in a single forward pass.  Position k's
    output is conditioned only on points 0..k-1, implemented via a shift-by-1
    input transform plus a lower-triangular attention mask.  This lets the
    model train on all N positions simultaneously (like an LLM) rather than
    sampling one target per step.

    At inference, run the full encoder on the partially-filled sequence and
    take the embedding at the next position to generate.
    """
    config: ARWFMConfig

    @nn.compact
    def __call__(self, target_points, padding_mask, deterministic=True, dropout_rng=jax.random.PRNGKey(0)):
        # target_points: [B, N, d]  — OT-matched targets in AR generation order
        # padding_mask:  [B, N]     — True = real point, False = padding

        B, N, _ = target_points.shape
        embedding_dim = self.config.context_embedding_dim
        num_layers    = self.config.context_num_layers
        num_heads     = self.config.context_num_heads
        mlp_hidden_dim = self.config.context_mlp_hidden_dim
        dropout_rate  = self.config.context_dropout_rate

        # Project all N target points to embedding space first.
        projected = nn.Dense(embedding_dim)(target_points)  # [B, N, embedding_dim]

        # Learned null-context embedding used as the start-of-sequence token at
        # position 0 of the shifted input.  Must NOT be zero: a zero start token
        # produces an all-zero Dense output (zero bias at init), which gives
        # LayerNorm variance=0 and gradient ≈ 1/sqrt(eps) ≈ 1000 per layer.
        # With 12 layers this overflows float32 to Inf → Inf/Inf = NaN in gradient
        # clipping → NaN parameters after the very first update.
        # Initialising at std=1 keeps the LayerNorm gradient ≈ 1 throughout.
        null_emb = self.param(
            'null_context', nn.initializers.normal(stddev=1.0), (embedding_dim,)
        )

        # Shift by 1: position k receives projected(target_{k-1}); position 0
        # receives the learned null embedding (context = empty).
        x = jnp.concatenate([
            jnp.broadcast_to(null_emb[None, None, :], (B, 1, embedding_dim)),
            projected[:, :-1, :],
        ], axis=1)  # [B, N, embedding_dim]

        # Padding mask for the shifted sequence:
        #   position 0 = null token (always valid)
        #   position k (k>=1) = projected target_{k-1}, valid iff padding_mask[k-1]
        shifted_valid = jnp.concatenate(
            [jnp.ones([B, 1], dtype=bool), padding_mask[:, :-1]], axis=1
        )  # [B, N]

        # Combined attention mask: [B, 1, N, N]
        # mask[b, 0, q, k] = (k <= q)  AND  shifted_valid[b, k]
        causal = jnp.tril(jnp.ones((N, N), dtype=bool))                 # [N, N]
        attn_mask = causal[None, None, :, :] & shifted_valid[:, None, None, :]  # [B, 1, N, N]

        for _ in range(num_layers):
            layer_rng, dropout_rng = jax.random.split(dropout_rng)
            x = ContextEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout_rate=dropout_rate,
            )(x, mask=attn_mask, deterministic=deterministic, dropout_rng=layer_rng)

        # Per-position residual MLP refinement (no mean pooling — keep all N embeddings)
        x = nn.LayerNorm()(x)
        for _ in range(num_layers):
            mlp_rng, dropout_rng = jax.random.split(dropout_rng)
            residual = x
            normed_x = nn.LayerNorm()(x)
            ff_output = FeedForward(hidden_dim=mlp_hidden_dim, dropout_rate=dropout_rate)(
                normed_x, deterministic=deterministic, dropout_rng=mlp_rng)
            x = ff_output + residual

        x = nn.LayerNorm()(x)
        x = nn.Dense(embedding_dim)(x)

        return x  # [B, N, context_embedding_dim]


class FlowMLP(nn.Module):
    """Single-point flow predictor v_theta.

    Predicts the velocity for one point x_t at time t, given a context embedding.
    Uses Fourier time features (same convention as the existing AttentionNN) and
    residual MLP blocks for depth.
    """
    config: ARWFMConfig
    output_dim: int

    @nn.compact
    def __call__(self, x, t, context, deterministic=True, dropout_rng=jax.random.PRNGKey(0)):
        # x:       [B, d]
        # t:       [B]
        # context: [B, context_embedding_dim]

        hidden_dim = self.config.flow_hidden_dim
        dropout_rate = self.config.flow_dropout_rate
        # Fourier time embedding — same formula as existing AttentionNN
        freqs = jnp.arange(hidden_dim // 2) * (2.0 * jnp.pi / hidden_dim)
        t_freq = freqs[None, :] * t[:, None]                        # [B, hidden_dim//2]
        t_four = jnp.concatenate([jnp.cos(t_freq), jnp.sin(t_freq)], axis=-1)  # [B, hidden_dim]
        t_emb = nn.Dense(hidden_dim)(t_four)                        # [B, hidden_dim]

        # Project point and context into hidden_dim, fuse with time
        x_emb = nn.Dense(hidden_dim)(x)
        context_emb = nn.Dense(hidden_dim)(context)
        
        emb_concat = jnp.concatenate([x_emb, t_emb, context_emb], axis=-1)
        h = nn.Dense(hidden_dim)(emb_concat)

        # Residual MLP blocks
        for _ in range(self.config.flow_num_layers):
            
            block_rng, dropout_rng = jax.random.split(dropout_rng)

            residual = h
            normed_h = nn.LayerNorm()(h) 

            ff_output = FeedForward(hidden_dim=hidden_dim, dropout_rate=dropout_rate)(normed_h, deterministic=deterministic, dropout_rng=block_rng)
            h = ff_output + residual
            
        h = nn.LayerNorm()(h)
        h = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.variance_scaling(
                1e-5, mode='fan_in', distribution='truncated_normal'),
            bias_init=nn.initializers.zeros,
        )(h)
        return h  # [B, d]


class ARFlowModel(nn.Module):
    """Combined AR flow model: causal context encoder + flow MLP.

    Forward pass (training — all N points in one shot):
        1. Encode all N target points causally: position k's embedding depends
           only on points 0..k-1, so every position can be trained simultaneously.
        2. Predict velocities for all N interpolated points via the flow MLP.
    """
    config: ARWFMConfig
    space_dim: int

    @nn.compact
    def __call__(self, x_t, t, target_points, padding_mask, deterministic=True, dropout_rng=jax.random.PRNGKey(0)):
        # x_t:          [B, N, d]  — interpolated points at their respective times
        # t:            [B, N]     — per-point interpolation times
        # target_points:[B, N, d]  — OT-matched targets in permutation order (context source)
        # padding_mask: [B, N]     — True = real point

        B, N, d = x_t.shape
        encoder_rng, flow_rng = jax.random.split(dropout_rng)

        context_embs = CausalContextEncoder(config=self.config)(
            target_points, padding_mask, deterministic=deterministic, dropout_rng=encoder_rng
        )  # [B, N, context_embedding_dim]

        # Apply FlowMLP to all N positions at once by flattening the N axis
        v_flat = FlowMLP(config=self.config, output_dim=self.space_dim)(
            x_t.reshape(B * N, d),
            t.reshape(B * N),
            context_embs.reshape(B * N, -1),
            deterministic=deterministic,
            dropout_rng=flow_rng,
        )  # [B*N, d]

        return v_flat.reshape(B, N, d)


# ---------------------------------------------------------------------------
# Spatial ALiBi RPE — isotropic relative positional encoding
# ---------------------------------------------------------------------------

def alibi_slopes(num_heads: int) -> jnp.ndarray:
    """Fixed ALiBi head slopes: 2^(-8h/H) for h = 1..H."""
    h = jnp.arange(1, num_heads + 1)
    return 2.0 ** (-8.0 * h / num_heads)  # [H]


class SpatialContextEncoderBlock(nn.Module):
    """Transformer encoder block with manual MHA so a float attention bias can be injected.

    Flax's nn.MultiHeadDotProductAttention only accepts boolean masks; ALiBi
    requires adding a float bias (causal -inf + spatial distance penalty) to the
    raw logits before softmax, so we implement QKV attention by hand.
    """
    embedding_dim: int
    num_heads: int
    mlp_hidden_dim: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, attn_bias, dropout_rng=None, deterministic=True):
        # x:        [B, N, D]
        # attn_bias:[B, H, N, N]  float — combines causal (-1e9 / 0) + ALiBi RPE
        B, N, D = x.shape
        H = self.num_heads
        head_dim = D // H

        attn_rng, ff_rng = jax.random.split(dropout_rng) if dropout_rng is not None else (None, None)

        normed = nn.LayerNorm()(x)

        # Q / K / V projections — no bias term (common in modern transformers)
        Q = nn.Dense(D, use_bias=False)(normed).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        K = nn.Dense(D, use_bias=False)(normed).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        V = nn.Dense(D, use_bias=False)(normed).reshape(B, N, H, head_dim).transpose(0, 2, 1, 3)
        # shapes: [B, H, N, head_dim]

        logits = jnp.einsum('bhqd,bhkd->bhqk', Q, K) * (head_dim ** -0.5) + attn_bias  # [B,H,N,N]
        weights = jax.nn.softmax(logits, axis=-1)

        if self.dropout_rate > 0:
            weights = nn.Dropout(rate=self.dropout_rate)(
                weights, deterministic=deterministic, rng=attn_rng)

        attn_out = jnp.einsum('bhqk,bhkd->bhqd', weights, V)          # [B, H, N, head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)    # [B, N, D]
        attn_out = nn.Dense(D)(attn_out)                               # output projection

        x = x + attn_out

        normed_x = nn.LayerNorm()(x)
        ff_out = FeedForward(hidden_dim=self.mlp_hidden_dim, dropout_rate=self.dropout_rate)(
            normed_x, deterministic=deterministic, dropout_rng=ff_rng)
        return x + ff_out


class SpatialCausalContextEncoder(nn.Module):
    """Causal context encoder with isotropic ALiBi spatial RPE.

    Identical to CausalContextEncoder except the attention bias combines:
      - causal masking  (−1e9 for future / padding, 0 for valid past)
      - ALiBi spatial bias  (−slope_h * ||pos_i − pos_j||₂)

    The BOS position (slot 0 of the shifted sequence) is set to the centroid
    of the cloud, supplied by the caller via positions_shifted[:, 0, :].
    """
    config: ARWFMConfig

    @nn.compact
    def __call__(self, target_points, positions_shifted, current_positions, padding_mask,
                 deterministic=True, dropout_rng=jax.random.PRNGKey(0)):
        # target_points:      [B, N, d]
        # positions_shifted:  [B, N, s]  — shifted coords (BOS=centroid at idx 0); drives ALiBi
        # current_positions:  [B, N, s]  — unshifted coords; tells the model WHERE to generate
        # padding_mask:       [B, N]     — True = real point

        B, N, _ = target_points.shape
        embedding_dim  = self.config.context_embedding_dim
        num_layers     = self.config.context_num_layers
        num_heads      = self.config.context_num_heads
        mlp_hidden_dim = self.config.context_mlp_hidden_dim
        dropout_rate   = self.config.context_dropout_rate

        projected = nn.Dense(embedding_dim)(target_points)  # [B, N, embedding_dim]

        null_emb = self.param(
            'null_context', nn.initializers.normal(stddev=1.0), (embedding_dim,)
        )

        # Shift by 1: position 0 = BOS (null_emb), position k = projected(target_{k-1})
        x = jnp.concatenate([
            jnp.broadcast_to(null_emb[None, None, :], (B, 1, embedding_dim)),
            projected[:, :-1, :],
        ], axis=1)  # [B, N, embedding_dim]

        shifted_valid = jnp.concatenate(
            [jnp.ones([B, 1], dtype=bool), padding_mask[:, :-1]], axis=1
        )  # [B, N]

        # Causal + padding mask as float bias [B, 1, N, N]
        causal = jnp.tril(jnp.ones((N, N), dtype=bool))
        causal_mask = causal[None, None, :, :] & shifted_valid[:, None, None, :]
        causal_bias = jnp.where(causal_mask, 0.0, -1e9).astype(jnp.float32)  # [B, 1, N, N]

        # ALiBi spatial RPE [B, H, N, N] — asymmetric:
        #   query[k] = current_positions[k]  (where we ARE generating)
        #   key[j]   = positions_shifted[j]  (where past context points are)
        # This gives: how far is the current target point from each past context point.
        slopes = alibi_slopes(num_heads)  # [H]
        diff = current_positions[:, :, None, :] - positions_shifted[:, None, :, :]  # [B,N,N,s]
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)                        # [B, N, N]
        rpe_bias = -slopes[None, :, None, None] * dist[:, None, :, :]              # [B, H, N, N]

        # Combined bias broadcasts [B,1,N,N] + [B,H,N,N] → [B, H, N, N]
        combined_bias = causal_bias + rpe_bias

        for _ in range(num_layers):
            layer_rng, dropout_rng = jax.random.split(dropout_rng)
            x = SpatialContextEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout_rate=dropout_rate,
            )(x, attn_bias=combined_bias, deterministic=deterministic, dropout_rng=layer_rng)

        # Per-position residual MLP refinement (same tail as CausalContextEncoder)
        x = nn.LayerNorm()(x)
        for _ in range(num_layers):
            mlp_rng, dropout_rng = jax.random.split(dropout_rng)
            residual = x
            normed_x = nn.LayerNorm()(x)
            ff_output = FeedForward(hidden_dim=mlp_hidden_dim, dropout_rate=dropout_rate)(
                normed_x, deterministic=deterministic, dropout_rng=mlp_rng)
            x = ff_output + residual

        x = nn.LayerNorm()(x)
        x = nn.Dense(embedding_dim)(x)
        return x  # [B, N, context_embedding_dim]


class SpatialARFlowModel(nn.Module):
    """AR flow model with spatial ALiBi RPE in the context encoder."""
    config: ARWFMConfig
    space_dim: int

    @nn.compact
    def __call__(self, x_t, t, target_points, positions_shifted, current_positions,
                 padding_mask, deterministic=True, dropout_rng=jax.random.PRNGKey(0)):
        # x_t:               [B, N, d]
        # t:                 [B, N]
        # target_points:     [B, N, d]
        # positions_shifted: [B, N, s]  — past positions (ALiBi context)
        # current_positions: [B, N, s]  — target positions (where to generate)
        # padding_mask:      [B, N]

        B, N, d = x_t.shape
        encoder_rng, flow_rng = jax.random.split(dropout_rng)

        context_embs = SpatialCausalContextEncoder(config=self.config)(
            target_points, positions_shifted, current_positions, padding_mask,
            deterministic=deterministic, dropout_rng=encoder_rng,
        )  # [B, N, context_embedding_dim]

        v_flat = FlowMLP(config=self.config, output_dim=self.space_dim)(
            x_t.reshape(B * N, d),
            t.reshape(B * N),
            context_embs.reshape(B * N, -1),
            deterministic=deterministic,
            dropout_rng=flow_rng,
        )  # [B*N, d]

        return v_flat.reshape(B, N, d)
