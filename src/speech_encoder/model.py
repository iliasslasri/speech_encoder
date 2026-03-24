from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict, overload

import fsspec
import joblib
import numpy as np
import torch
from jaxtyping import Float, Int64
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torchaudio.models import Wav2Vec2Model, hubert_base

import torch.nn.functional as F
from .assets import load_hubert_fairseq_state_dict, match_name_or_path_with_textlesslib_names, textlesslib_checkpoints


class KMeansQuantizer(nn.Module):
    def __init__(self, fitted_kmeans: KMeans) -> None:
        super().__init__()
        self._kmeans = fitted_kmeans

    @property
    def n_clusters(self) -> int:
        return self._kmeans.n_clusters

    @property
    def cluster_centers(self) -> Float[Tensor, "k dim"]:
        return torch.from_numpy(self._kmeans.cluster_centers_)

    @overload
    def forward(self, x: Float[Tensor, "batch seq dim"]) -> Int64[Tensor, "batch seq"]: ...

    @overload
    def forward(self, x: Float[Tensor, "seq dim"]) -> Int64[Tensor, " seq"]: ...

    @torch.inference_mode()
    def forward(self, x):
        x_np = x.cpu().numpy()
        match x.ndim:
            case 2:
                predictions = self._kmeans.predict(x_np)
            case 3:
                predictions = np.stack([self._kmeans.predict(y) for y in x_np])
            case _:
                raise ValueError(f"Invalid number of dimensions: {x.ndim}")
        return torch.from_numpy(predictions).to(x.device)

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path) -> "KMeansQuantizer":
        path = match_name_or_path_with_textlesslib_names(name_or_path)
        with fsspec.open(str(path), "rb") as f:
            return KMeansQuantizer(joblib.load(f))

    @classmethod
    def available_checkpoints(cls) -> tuple[str, ...]:
        return tuple(
            name
            for name in textlesslib_checkpoints()
            if "kmeans" in name and "tacotron" not in name and "hifigan" not in name
        )


class HuBERT(nn.Module):
    def __init__(self, pretrained_hubert: Wav2Vec2Model, *, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.model = pretrained_hubert
        self._hidden_state: torch.Tensor | None = None
        self.model.encoder.transformer.layers[layer - 1].feed_forward.register_forward_hook(self._hook)  # ty:ignore[unresolved-attribute, not-subscriptable]

    def _hook(self, _: nn.Module, __: tuple[Any, ...], output: torch.Tensor) -> None:
        """Register a forward hook to match fairseq behavior (different from torchaudio and huggingface transformers).

        Extract hidden state just after the feed_forward, before layer_norm and residual.
        References:
            fairseq:
                - https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L1135
                - https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py#L1329-L1375
            torchaudio:
                - https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L445-L463
            transformers:
                - https://github.com/huggingface/transformers/blob/main/src/transformers/models/hubert/modeling_hubert.py#L418-L477
        """
        if self._hidden_state is not None:
            raise ValueError("Hidden state should be 'None' at this stage. Did not get cleaned up")
        self._hidden_state = output.detach()

    @overload
    def forward(
        self,
        waveforms: Float[Tensor, "batch time"],
        lengths: None,
    ) -> tuple[Float[Tensor, "batch seq"], None]: ...

    @overload
    def forward(
        self,
        waveforms: Float[Tensor, "batch time"],
        lengths: Int64[Tensor, " batch"],
    ) -> tuple[Float[Tensor, "batch seq"], Int64[Tensor, " batch"]]: ...

    @torch.inference_mode()
    def forward(self, waveforms, lengths=None):
        _, output_lengths = self.model.extract_features(waveforms, lengths, num_layers=self.layer)
        hidden_state = self._hidden_state
        if hidden_state is None:
            raise ValueError("Hidden state has not been set, forward hook failure.")
        self._hidden_state = None
        return hidden_state, output_lengths

    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, layer: int) -> "HuBERT":
        path = match_name_or_path_with_textlesslib_names(name_or_path)
        state_dict = load_hubert_fairseq_state_dict(path)
        pretrained_hubert = hubert_base().eval()
        pretrained_hubert.load_state_dict(state_dict)
        return HuBERT(pretrained_hubert, layer=layer)

    @classmethod
    def available_checkpoints(cls) -> tuple[str, ...]:
        return ("hubert-base-ls960", "mhubert-base-vp_en_es_fr", "mhubert-base-vp_mls_cv_8lang")


class SpidRWrapper(nn.Module):
    def __init__(self, model_name: str, *, layer: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.layer = layer
        self.model = torch.hub.load("facebookresearch/spidr", model_name)
        self.conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    @staticmethod
    def conv_length(lengths: Tensor, conv_layer_config: list[tuple[int, int, int]]) -> Tensor:
        for _, kernel_size, stride in conv_layer_config:
            # handle float or int
            lengths = torch.div(lengths - kernel_size, stride, rounding_mode="floor") + 1
            lengths = torch.max(torch.zeros_like(lengths), lengths)
        return lengths

    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        # Normalize: SpidR expects standardized audio (mean 0, var 1)
        # waveforms is [batch, time]
        waveforms = F.layer_norm(waveforms, (waveforms.shape[-1],))
        
        feats = self.model.feature_extractor(waveforms)
        feats = self.model.feature_projection(feats)
        
        # Get intermediate outputs from student model
        # layer is 1-indexed, get_intermediate_outputs expects num_layers
        hidden_states = self.model.student.get_intermediate_outputs(feats, num_layers=self.layer)
        hidden_state = hidden_states[-1]

        if lengths is not None:
             lengths = self.conv_length(lengths, self.conv_layer_config)
        
        return hidden_state, lengths


class SpidRQuantizer(nn.Module):
    def __init__(self, spidr_model: nn.Module, *, layer: int) -> None:
        super().__init__()
        self.model = spidr_model
        self.layer = layer  # 1-indexed, matches SpidRWrapper convention
        self.conv_layer_config = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    @torch.inference_mode()
    def forward(self, waveforms: Tensor, lengths: Tensor | None = None) -> Tensor:
        # Standardize audio
        waveforms = F.layer_norm(waveforms, (waveforms.shape[-1],))
        
        attention_mask = None
        if lengths is not None:
            ds_lengths = SpidRWrapper.conv_length(lengths, self.conv_layer_config)
            
            with torch.no_grad():
                feats = self.model.feature_extractor(waveforms)
                max_len = feats.shape[1]
            
            padding_mask = torch.arange(max_len, device=lengths.device).expand(lengths.shape[0], max_len) >= ds_lengths[:, None]
            attention_mask = ~padding_mask[:, None, None, :].expand(lengths.shape[0], 1, max_len, max_len)

        # get_codebooks returns list of [batch, seq, vocab] probabilities when onehot=False
        codeprobs_list = self.model.get_codebooks(waveforms, attention_mask=attention_mask, onehot=False)
        
        valid_codeprobs = [c for c in codeprobs_list if c is not None]
        if not valid_codeprobs:
            raise ValueError("No valid codebooks returned from SpidR model.")
        
        # codeprobs_list is layer-indexed (0-based). self.layer is 1-indexed.
        # Layers without a codebook are None; index directly to respect the configured layer.
        layer_idx = self.layer - 1
        if layer_idx >= len(codeprobs_list) or codeprobs_list[layer_idx] is None:
            available = [i + 1 for i, c in enumerate(codeprobs_list) if c is not None]
            raise ValueError(
                f"Layer {self.layer} does not have a codebook in this SpidR model. "
                f"Available codebook layers: {available}"
            )
        target_codebook = codeprobs_list[layer_idx]
        
        if target_codebook.ndim == 2: # [seq, vocab] if batch_size=1
             target_codebook = target_codebook.unsqueeze(0)
             
        units = target_codebook.argmax(dim=-1) # [batch, seq]
        return units


class AvailableConfig(NamedTuple):
    name: str
    layer: int
    vocab_size: int
    kind_kmeans: Literal["kmeans", "kmeans-expresso"]


class DiscreteUnits(TypedDict):
    units: list[int]
    counts: list[int]


class SpeechEncoder(nn.Module):
    def __init__(self, dense: nn.Module, quantizer: nn.Module, *, deduplicate: bool) -> None:
        super().__init__()
        self.dense = dense
        self.quantizer = quantizer
        self.deduplicate = deduplicate

    @overload
    def forward(
        self,
        waveforms: Float[Tensor, "batch time"],
        lengths: Int64[Tensor, " batch"] | None,
        *,
        formatted: Literal[True],
    ) -> list[DiscreteUnits]: ...

    @overload
    def forward(
        self,
        waveforms: Float[Tensor, "batch time"],
        lengths: Int64[Tensor, " batch"],
        *,
        formatted: Literal[False],
    ) -> tuple[Float[Tensor, "batch seq"], Int64[Tensor, " batch"]]: ...

    @overload
    def forward(
        self,
        waveforms: Float[Tensor, "batch time"],
        lengths: None,
        *,
        formatted: Literal[False],
    ) -> tuple[Float[Tensor, "batch seq"], None]: ...

    def forward(self, waveforms, lengths=None, *, formatted=True):
        hidden_state, lengths = self.dense(waveforms, lengths)
        if isinstance(self.quantizer, SpidRQuantizer):
            units = self.quantizer(waveforms, lengths)
        else:
            units = self.quantizer(hidden_state)

        if not formatted:
            if lengths is not None:
                mask = torch.arange(units.shape[-1]).unsqueeze(0) >= lengths.unsqueeze(1)
                units[mask] = -1
            return units, lengths
        if lengths is None:
            lengths = torch.full(size=(units.shape[0],), fill_value=units.shape[1], device=units.device)
        if self.deduplicate:
            output = [torch.unique_consecutive(u[:n], return_counts=True) for u, n in zip(units, lengths.tolist(), strict=True)]
        else:
            output = [(u[:n], torch.ones(n, dtype=torch.int64)) for u, n in zip(units, lengths.tolist(), strict=True)]
        return [{"units": u.tolist(), "counts": c.tolist()} for u, c in output]

    @classmethod
    def from_textlesslib(
        cls,
        name: str,
        *,
        layer: int,
        vocab_size: int,
        deduplicate: bool,
        kind_kmeans: str = "kmeans",
    ) -> "SpeechEncoder":
        if name.startswith("spidr") or name.startswith("dinosr"):
            model = SpidRWrapper(name, layer=layer)
            quantizer = SpidRQuantizer(model.model, layer=layer)
            return SpeechEncoder(model, quantizer, deduplicate=deduplicate).eval()

        if (name, layer, vocab_size, kind_kmeans) not in cls.available_checkpoints_list():
            available = "\n".join(str(c) for c in cls.available_checkpoints_list())
            raise ValueError(f"Invalid combination of arguments. Pick one of:\n{available}")
        hubert = HuBERT.from_pretrained(name, layer=layer)
        kmeans = KMeansQuantizer.from_pretrained(f"{name}-layer-{layer}-{kind_kmeans}-{vocab_size}")
        return SpeechEncoder(hubert, kmeans, deduplicate=deduplicate).eval()

    @classmethod
    def available_checkpoints_list(cls) -> tuple[AvailableConfig, ...]:
        return (
            AvailableConfig("hubert-base-ls960", 6, 50, "kmeans"),
            AvailableConfig("hubert-base-ls960", 6, 100, "kmeans"),
            AvailableConfig("hubert-base-ls960", 6, 200, "kmeans"),
            AvailableConfig("hubert-base-ls960", 6, 500, "kmeans"),
            AvailableConfig("hubert-base-ls960", 9, 500, "kmeans"),
            AvailableConfig("hubert-base-ls960", 9, 2000, "kmeans-expresso"),
            AvailableConfig("mhubert-base-vp_en_es_fr", 11, 1000, "kmeans"),
            AvailableConfig("mhubert-base-vp_mls_cv_8lang", 12, 2000, "kmeans"),
            AvailableConfig("mhubert-base-vp_mls_cv_8lang", 12, 2000, "kmeans-expresso"),
            # SpidR / DinoSR
            AvailableConfig("spidr_base", 6, 256, "spidr"),
            AvailableConfig("dinosr_base_reproduced", 5, 256, "spidr"),
            AvailableConfig("dinosr_base_original", 5, 256, "spidr"),
        )

    @classmethod
    def available_checkpoints(cls) -> tuple[AvailableConfig, ...]:
        return cls.available_checkpoints_list()
