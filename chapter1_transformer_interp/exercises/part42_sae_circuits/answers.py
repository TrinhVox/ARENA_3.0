# %%

import gc
import os
import sys
from collections import Counter, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, TypeAlias

import einops
import numpy as np
import plotly.express as px
import torch as t
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from IPython.display import IFrame, display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from sae_lens import SAE, ActivationsStore, HookedSAETransformer
from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
from tabulate import tabulate
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy

dtype = t.float32  # t.bfloat16
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
device = str(device)  # SAELens expects device as string; this is easier!


def _get_hook_layer(sae: SAE) -> int:
    """Extract the layer number from an SAE's hook name (e.g. 'blocks.7.hook_resid_pre' → 7)."""
    return int(sae.cfg.metadata.hook_name.split(".")[1])


# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part42_sae_circuits"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part42_sae_circuits.tests as tests
import part42_sae_circuits.utils as utils

MAIN = __name__ == "__main__"

# %%
gpt2 = HookedSAETransformer.from_pretrained("gpt2-small", device=device, dtype=dtype)

gpt2_saes = {
    layer: SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_pre",
        device=device,
        dtype=dtype,
    )
    for layer in tqdm(range(gpt2.cfg.n_layers))
}

# %%
class SparseTensor:
    """
    Handles 2D tensor data (assumed to be non-negative) in 2 different formats:
        dense:  The full tensor, which contains zeros. Shape is (n1, ..., nk).
        sparse: A tuple of nonzero values with shape (n_nonzero,), nonzero indices with shape
                (n_nonzero, k), and the shape of the dense tensor.
    """

    sparse: tuple[Tensor, Tensor, tuple[int, ...]]
    dense: Tensor

    def __init__(self, sparse: tuple[Tensor, Tensor, tuple[int, ...]], dense: Tensor):
        self.sparse = sparse
        self.dense = dense

    @classmethod
    def from_dense(cls, dense: Tensor) -> "SparseTensor":
        """Creates a SparseTensor from a dense tensor, extracting the positive entries."""
        values = dense[dense>0]
        indices = dense.nonzero()
        shape = tuple(dense.shape)
        return cls((values,indices,shape), dense)

    @classmethod
    def from_sparse(cls, sparse: tuple[Tensor, Tensor, tuple[int, ...]]) -> "SparseTensor":
        """Creates a SparseTensor from a (values, indices, shape) tuple, reconstructing the dense tensor."""
        values, indices, shape = sparse
        dense = t.zeros(shape, dtype=values.dtype, device=values.device)
        dense[indices.unbind(-1)] = values
        return cls(sparse, dense)


    @property
    def values(self) -> Tensor:
        return self.sparse[0].squeeze()

    @property
    def indices(self) -> Tensor:
        return self.sparse[1].squeeze()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.sparse[2]


# Test `from_dense`
x = t.zeros(10_000)
nonzero_indices = t.randint(0, 10_000, (10,)).sort().values
nonzero_values = t.rand(10)
x[nonzero_indices] = nonzero_values
sparse_tensor = SparseTensor.from_dense(x)
t.testing.assert_close(sparse_tensor.sparse[0], nonzero_values)
t.testing.assert_close(sparse_tensor.sparse[1].squeeze(-1), nonzero_indices)
t.testing.assert_close(sparse_tensor.dense, x)

# Test `from_sparse`
sparse_tensor = SparseTensor.from_sparse((nonzero_values, nonzero_indices.unsqueeze(-1), tuple(x.shape)))
t.testing.assert_close(sparse_tensor.dense, x)

# Verify other properties
t.testing.assert_close(sparse_tensor.values, nonzero_values)
t.testing.assert_close(sparse_tensor.indices, nonzero_indices)

tests.test_sparse_tensor(SparseTensor)

# %%
def latent_acts_to_later_latent_acts(
    latent_acts_nonzero: Float[Tensor, " nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given some latent activations for a residual stream SAE earlier in the model, computes the
    latent activations of a later SAE. It does this by mapping the latent activations through the
    path SAE decoder -> intermediate model layers -> later SAE encoder.

    This function must input & output sparse information (i.e. nonzero values and their indices)
    rather than dense tensors, because latent activations are sparse but jacrev() doesn't support
    gradients on real sparse tensors.
    """
    # Convert to dense, map through SAE decoder
    latent_acts = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)).dense
    resid_stream_from = sae_from.decode(latent_acts)

    # Map through model layers
    resid_stream_next = model.forward(
        resid_stream_from,
        start_at_layer=_get_hook_layer(sae_from),
        stop_at_layer=_get_hook_layer(sae_to),
    )

    # Map through SAE encoder, and turn back into SparseTensor
    latent_acts_next_recon = sae_to.encode(resid_stream_next)
    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon)

    return latent_acts_next_recon.sparse[0], (latent_acts_next_recon.dense,)


# %%
def latent_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae_from: SAE,
    sae_to: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor, SparseTensor, SparseTensor]:
    """
    Computes the gradients between all active pairs of latents belonging to two SAEs.

    Returns:
        latent_latent_gradients:    The gradients between all active pairs of latents
        latent_acts_prev:           The latent activations of the first SAE
        latent_acts_next:           The latent activations of the second SAE
        latent_acts_next_recon:     The reconstructed latent activations of the second SAE (i.e.
                                    based on the first SAE's reconstructions)
    """
    acts_prev_name = f"{sae_from.cfg.metadata.hook_name}.hook_sae_acts_post"
    acts_next_name = f"{sae_to.cfg.metadata.hook_name}.hook_sae_acts_post"
    sae_from.use_error_term = True  # so we can get both true latent acts at once

    with t.no_grad():
        # Get the true activations for both SAEs
        _, cache = model.run_with_cache_with_saes(
            tokens,
            names_filter=[acts_prev_name, acts_next_name],
            stop_at_layer=_get_hook_layer(sae_to) + 1,
            saes=[sae_from, sae_to],
            remove_batch_dim=False,
        )
        latent_acts_prev = SparseTensor.from_dense(cache[acts_prev_name])
        latent_acts_next = SparseTensor.from_dense(cache[acts_next_name])

    # Compute jacobian between earlier and later latent activations (and also get the activations
    # of the later SAE which are downstream of the earlier SAE's reconstructions)
    latent_latent_gradients, (latent_acts_next_recon_dense,) = t.func.jacrev(
        latent_acts_to_later_latent_acts, has_aux=True
    )(
        *latent_acts_prev.sparse,
        sae_from,
        sae_to,
        model,
    )

    latent_acts_next_recon = SparseTensor.from_dense(latent_acts_next_recon_dense)

    # Set SAE state back to default
    sae_from.use_error_term = False

    return (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    )
#%%
prompt = "The Eiffel tower is in Paris"
tokens = gpt2.to_tokens(prompt)
str_toks = gpt2.to_str_tokens(prompt)
layer_from = 0
layer_to = 3

# Get latent-to-latent gradients
t.cuda.empty_cache()
with t.enable_grad():
    (
        latent_latent_gradients,
        latent_acts_prev,
        latent_acts_next,
        latent_acts_next_recon,
    ) = latent_to_latent_gradients(tokens, gpt2_saes[layer_from], gpt2_saes[layer_to], gpt2)

# Verify that ~the same latents are active in both, and the MSE loss is small
nonzero_latents = [tuple(x) for x in latent_acts_next.indices.tolist()]
nonzero_latents_recon = [tuple(x) for x in latent_acts_next_recon.indices.tolist()]
alive_in_one_not_both = set(nonzero_latents) ^ set(nonzero_latents_recon)
print(f"# nonzero latents (true): {len(nonzero_latents)}")
print(f"# nonzero latents (reconstructed): {len(nonzero_latents_recon)}")
print(f"# latents alive in one but not both: {len(alive_in_one_not_both)}")

fig = px.imshow(
    to_numpy(latent_latent_gradients.T),
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    x=[f"F{layer_to}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_next_recon.indices],
    y=[f"F{layer_from}.{latent}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts_prev.indices],
    labels={"x": f"To layer {layer_to}", "y": f"From layer {layer_from}"},
    title=f'Gradients between SAE latents in layer {layer_from} and SAE latents in layer {layer_to}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
    width=1600,
    height=1000,
)
fig.show()

# %%
def tokens_to_latent_acts(
    token_scales: Float[Tensor, "batch seq"],
    tokens: Int[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Given scale factors for model's embeddings (i.e. scale factors applied after we compute the sum
    of positional and token embeddings), returns the SAE's latents.

    Returns:
        latent_acts_sparse: The SAE's latents in sparse form (i.e. the tensor of values)
        latent_acts_dense:  The SAE's latents in dense tensor, in a length-1 tuple
    """
    resid_after_embed = model(tokens, stop_at_layer=0)
    resid_after_embed = einops.einsum(resid_after_embed, token_scales, "... seq d_model, ... seq -> ... seq d_model")
    resid_stream = model.forward(
        resid_after_embed,
        start_at_layer=0,
        stop_at_layer = _get_hook_layer(sae)
    )

    sae_latents = sae.encode(resid_stream)
    sae_latents = SparseTensor.from_dense(sae_latents)

    return sae_latents.sparse[0], (sae_latents.dense,)


def token_to_latent_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
) -> tuple[Tensor, SparseTensor]:
    """
    Computes the gradients between an SAE's latents and all input tokens.

    Returns:
        token_latent_grads: The gradients between input tokens and SAE latents
        latent_acts:        The SAE's latent activations
    """
    token_scales = t.ones(tokens.shape, device=model.cfg.device, requires_grad=True)


    token_latent_grads, (latent_acts_dense,) = t.func.jacrev(
        tokens_to_latent_acts, has_aux=True
    )(
        token_scales, tokens, sae, model
    )

    token_latent_grads = einops.rearrange(token_latent_grads, "d_sae_nonzero batch seq -> batch seq d_sae_nonzero")

    latent_acts = SparseTensor.from_dense(latent_acts_dense)
    return (token_latent_grads, latent_acts)


tests.test_token_to_latent_gradients(token_to_latent_gradients, gpt2, gpt2_saes)

sae_layer = 3
token_latent_grads, latent_acts = token_to_latent_gradients(tokens, sae=gpt2_saes[sae_layer], model=gpt2)

fig = px.imshow(
    to_numpy(token_latent_grads[0]),
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    x=[f"F{sae_layer}.{latent:05}, {str_toks[seq]!r} ({seq})" for (_, seq, latent) in latent_acts.indices],
    y=[f"{str_toks[i]!r} ({i})" for i in range(len(str_toks))],
    labels={"x": f"To layer {sae_layer}", "y": "From tokens"},
    title=f'Gradients between input tokens and SAE latents in layer {sae_layer}<br><sup>   Prompt: "{"".join(str_toks)}"</sup>',
    width=1900,
    height=450,
)
fig.show()

# %%
def latent_acts_to_logits(
    latent_acts_nonzero: Float[Tensor, " nonzero_acts"],
    latent_acts_nonzero_inds: Int[Tensor, "nonzero_acts n_indices"],
    latent_acts_shape: tuple[int, ...],
    sae: SAE,
    model: HookedSAETransformer,
    token_ids: list[int] | None = None,
) -> tuple[Tensor, tuple[Tensor]]:
    """
    Computes the logits as a downstream function of the SAE's reconstructed residual stream. If we
    supply `token_ids`, it means we only compute & return the logits for those specified tokens.
    """
    latent_acts = SparseTensor.from_sparse((latent_acts_nonzero, latent_acts_nonzero_inds, latent_acts_shape)).dense
    latent_resid_stream = sae.decode(latent_acts)

    logits_recon = model.forward(
        latent_resid_stream,
        start_at_layer=_get_hook_layer(sae))[0,-1]

    
    return logits_recon[token_ids], (logits_recon,)


def latent_to_logit_gradients(
    tokens: Float[Tensor, "batch seq"],
    sae: SAE,
    model: HookedSAETransformer,
    k: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, list[int] | None, SparseTensor]:
    """
    Computes the gradients between active latents and some top-k set of logits (we
    use k to avoid having to compute the gradients for all tokens).

    Returns:
        latent_logit_gradients:  The gradients between the SAE's active latents & downstream logits
        logits:                  The model's true logits
        logits_recon:            The model's reconstructed logits (i.e. based on SAE reconstruction)
        token_ids:               The tokens we computed the gradients for
        latent_acts:             The SAE's latent activations
    """
    assert tokens.shape[0] == 1, "Only supports batch size 1 for now"

    acts_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    sae.use_error_term=True

    with t.no_grad():
        logits, cache = model.run_with_cache_with_saes(
            tokens,
            names_filter=[acts_name],
            saes=[sae],
            remove_batch_dim=False
        )

        latents_act = SparseTensor.from_dense(cache[acts_name])
    
    token_ids = None if k is None else logits.topk(k=k).indices.tolist()

    latent_logit_gradients, (logits_recon,) = t.func.jacrev(
        latent_acts_to_logits, has_aux=True
    )(
        *latents_act.sparse,
        sae,
        model,
        token_ids
    )
    sae.use_error_term=False

    return (
        latent_logit_gradients,
        logits,
        logits_recon,
        token_ids,
        latent_acts,
    )

# %%
