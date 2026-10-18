# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from optimum.utils import (
    DummyAudioInputGenerator,
    DummyPastKeyValuesGenerator,
    DummyTransformerTextInputGenerator,
    NormalizedTextConfig,
    is_transformers_version,
)


class GPTBigCodeDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    def __init__(self, task: str, normalized_config: NormalizedTextConfig, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, **kwargs)
        self.multi_query = normalized_config.multi_query

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if is_transformers_version("<", "4.54"):
            if self.multi_query:
                shape = (
                    self.batch_size,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads * 2,
                )
            else:
                shape = (
                    self.batch_size,
                    self.num_attention_heads,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads * 2,
                )
            pkv = [
                self.random_float_tensor(shape, framework=framework, dtype=float_dtype) for _ in range(self.num_layers)
            ]

        else:
            if self.multi_query:
                shape = (
                    self.batch_size,
                    1,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads,
                )
            else:
                shape = (
                    self.batch_size,
                    self.num_attention_heads,
                    self.sequence_length,
                    self.hidden_size // self.num_attention_heads,
                )
            pkv = [
                (
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(shape, framework=framework, dtype=float_dtype),
                )
                for _ in range(self.num_layers)
            ]

        return pkv


class DummyMoonshineAudioInputGenerator(DummyAudioInputGenerator):
    SUPPORTED_INPUT_NAMES = ("input_values", "attention_mask")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "input_values":  # raw waveform
            return self.random_float_tensor(
                shape=[self.batch_size, self.sequence_length],
                min_value=-1,
                max_value=1,
                framework=framework,
                dtype=float_dtype,
            )
        elif input_name == "attention_mask":  # attention mask
            return self.random_mask_tensor(
                shape=[self.batch_size, self.sequence_length],
                framework=framework,
                dtype=int_dtype,
            )
        else:
            raise ValueError(f"Unsupported input name: {input_name}")


class DummySanaTransforemerTextInputGenerator(DummyTransformerTextInputGenerator):
    SUPPORTED_INPUT_NAMES = ("encoder_hidden_states", "encoder_attention_mask")

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "encoder_attention_mask":
            return self.random_mask_tensor(
                shape=[self.batch_size, self.sequence_length],
                framework=framework,
                dtype=int_dtype,
            )
        else:
            return super().generate(
                input_name=input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype
            )


class Qwen3_5DummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """Dummy state generator for the Qwen3.5 hybrid (linear + full) attention cache.

    Qwen3.5 interleaves two layer kinds, selected per-layer by ``config.layer_types``:

    - ``full_attention`` layers keep a standard transformer KV cache, exported as
      ``past_key_values.{i}.key`` / ``past_key_values.{i}.value`` with a dynamic
      ``past_sequence_length`` axis, like any GQA decoder.
    - ``linear_attention`` layers (Gated-DeltaNet) keep a *fixed-size* recurrent state
      instead of a growing KV cache, exported as two tensors per layer:
        * ``past_conv.{i}``      shape (batch, conv_dim, conv_kernel_dim)
        * ``past_recurrent.{i}`` shape (batch, v_heads, k_head_dim, v_head_dim)
      Neither has a sequence axis, so neither grows during decoding.

    ``i`` is the *global* layer index (0..num_layers-1), so the names line up 1:1 with
    the ONNX graph produced by torch.onnx and with the names transformers.js expects.
    """

    SUPPORTED_INPUT_NAMES = ("past_key_values", "past_conv", "past_recurrent")

    def __init__(self, task, normalized_config, **kwargs):
        super().__init__(task=task, normalized_config=normalized_config, **kwargs)
        # NormalizedConfig proxies unknown attributes to the underlying config
        self.num_key_value_heads = normalized_config.num_key_value_heads
        self.head_dim = getattr(normalized_config, "head_dim", self.hidden_size // self.num_attention_heads)
        self.layer_types = list(normalized_config.layer_types)
        # linear-attention (Gated-DeltaNet) geometry
        self.conv_kernel_dim = normalized_config.linear_conv_kernel_dim
        lin_k_heads = normalized_config.linear_num_key_heads
        lin_k_dim = normalized_config.linear_key_head_dim
        lin_v_heads = normalized_config.linear_num_value_heads
        lin_v_dim = normalized_config.linear_value_head_dim
        # conv acts on the concatenated [q, k, v] linear projections
        self.conv_dim = 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim
        self.rec_shape = (self.batch_size, lin_v_heads, lin_k_dim, lin_v_dim)
        self.conv_shape = (self.batch_size, self.conv_dim, self.conv_kernel_dim)

    def generate(self, input_name, framework="pt", int_dtype="int64", float_dtype="fp32"):
        full_idx = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        lin_idx = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        kv_shape = (
            self.batch_size,
            self.num_key_value_heads,
            self.sequence_length,
            self.head_dim,
        )
        if input_name == "past_key_values":
            return [
                (
                    self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype),
                    self.random_float_tensor(kv_shape, framework=framework, dtype=float_dtype),
                )
                for _ in full_idx
            ]
        if input_name == "past_conv":
            return [
                self.random_float_tensor(self.conv_shape, framework=framework, dtype=float_dtype) for _ in lin_idx
            ]
        if input_name == "past_recurrent":
            return [
                self.random_float_tensor(self.rec_shape, framework=framework, dtype=float_dtype) for _ in lin_idx
            ]
        raise ValueError(f"Unsupported input name: {input_name}")
