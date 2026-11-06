"""Component vocabulary and weight-slice mapping for Cohere2 / Tiny Aya.

The component space follows the spec (section 4.2):

* ``L{l}.H{h}`` covers a query head, the corresponding GQA key/value rows, and
  the output-projection columns for that head.
* ``L{l}.MLP`` covers ``gate_proj``, ``up_proj`` and ``down_proj`` in that layer.

K/V slices are shared by every head in a GQA group. When a subset of a group is
selected, the K/V slice receives a strength multiplier equal to the selected
fraction of the group, as specified in section 4.2.

The architecture facts below were checked against the installed Transformers
``cohere2`` implementation on 2026-11-05:

* attention/MLP live at ``model.layers.{l}.self_attn.*`` and
  ``model.layers.{l}.mlp.*``;
* Cohere2 uses *parallel* attention and MLP, so there is only
  ``input_layernorm`` (no post-attention norm);
* ``lm_head.weight`` is tied to ``model.embed_tokens.weight``;
* ``head_dim`` is ``hidden_size // num_attention_heads`` unless overridden.

The live Tiny Aya checkpoint check (``assert_matches_model``) is part of the
D1 smoke test and requires the gated weights.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

ComponentKind = Literal["head", "mlp"]

_COMPONENT_RE = re.compile(r"^L(?P<layer>\d+)\.(?:H(?P<head>\d+)|(?P<mlp>MLP))$")

ATTENTION_TENSORS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TENSORS = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Component:
    """A single node in the shared circuit vocabulary."""

    layer: int
    kind: ComponentKind
    head: int | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be >= 0")
        if self.kind == "head":
            if self.head is None or self.head < 0:
                raise ValueError("head components require a non-negative head index")
        elif self.head is not None:
            raise ValueError("MLP components cannot carry a head index")

    @property
    def id(self) -> str:
        return f"L{self.layer}.H{self.head}" if self.kind == "head" else f"L{self.layer}.MLP"

    @property
    def sort_key(self) -> tuple[int, str, int]:
        return (self.layer, self.kind, -1 if self.head is None else self.head)

    def __str__(self) -> str:
        return self.id

    @classmethod
    def parse(cls, component_id: str) -> Component:
        match = _COMPONENT_RE.match(component_id)
        if not match:
            raise ValueError(f"invalid component id: {component_id!r}")
        head = match.group("head")
        return cls(
            layer=int(match.group("layer")),
            kind="head" if head is not None else "mlp",
            head=int(head) if head is not None else None,
        )


@dataclass(frozen=True)
class TensorEdit:
    """A slice of one weight tensor and the strength to apply to it."""

    tensor: str
    dim: int
    start: int
    end: int
    strength: float
    component: str

    @property
    def length(self) -> int:
        return self.end - self.start

    def key(self) -> tuple[str, int, int, int]:
        return (self.tensor, self.dim, self.start, self.end)


@dataclass(frozen=True)
class Cohere2Layout:
    """Static architecture description used for slice arithmetic."""

    num_layers: int = 36
    hidden_size: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 11008
    prefix: str = "model"

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads * self.head_dim != self.hidden_size:
            raise ValueError(
                "num_heads * head_dim must equal hidden_size "
                f"({self.num_heads} * {self.head_dim} != {self.hidden_size})"
            )
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def num_components(self) -> int:
        return self.num_layers * (self.num_heads + 1)

    @classmethod
    def from_preset(cls) -> Cohere2Layout:
        """Tiny Aya's published architecture (section 1.1 of the spec)."""
        return cls()

    @classmethod
    def from_config(cls, config: object, prefix: str = "model") -> Cohere2Layout:
        """Build from a ``Cohere2Config``-like object."""
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        return cls(
            num_layers=int(config.num_hidden_layers),
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=int(num_kv_heads) if num_kv_heads is not None else num_heads,
            head_dim=head_dim,
            intermediate_size=int(config.intermediate_size),
            prefix=prefix,
        )

    # -- naming ---------------------------------------------------------

    def layer_prefix(self, layer: int) -> str:
        self._check_layer(layer)
        return f"{self.prefix}.layers.{layer}"

    def attn_tensor(self, layer: int, name: str) -> str:
        if name not in ATTENTION_TENSORS:
            raise ValueError(name)
        return f"{self.layer_prefix(layer)}.self_attn.{name}"

    def mlp_tensor(self, layer: int, name: str) -> str:
        if name not in MLP_TENSORS:
            raise ValueError(name)
        return f"{self.layer_prefix(layer)}.mlp.{name}"

    def editable_tensor_names(self) -> list[str]:
        names: list[str] = []
        for layer in range(self.num_layers):
            names.extend(self.attn_tensor(layer, name) for name in ATTENTION_TENSORS)
            names.extend(self.mlp_tensor(layer, name) for name in MLP_TENSORS)
        return names

    def _editable_name_set(self) -> set[str]:
        return set(self.editable_tensor_names())

    def non_editable_tensor_names(self) -> list[str]:
        names = [f"{self.prefix}.embed_tokens.weight"]
        for layer in range(self.num_layers):
            names.append(f"{self.layer_prefix(layer)}.input_layernorm.weight")
        names.append(f"{self.prefix}.norm.weight")
        names.append("lm_head.weight")
        return names

    # -- components -----------------------------------------------------

    def all_components(self) -> list[Component]:
        components: list[Component] = []
        for layer in range(self.num_layers):
            components.extend(Component(layer, "head", head) for head in range(self.num_heads))
            components.append(Component(layer, "mlp"))
        return components

    def head_components(self) -> Iterator[Component]:
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                yield Component(layer, "head", head)

    def mlp_components(self) -> Iterator[Component]:
        for layer in range(self.num_layers):
            yield Component(layer, "mlp")

    def kv_group(self, head: int) -> int:
        self._check_head(head)
        return head // self.group_size

    def heads_in_group(self, layer: int, group: int) -> list[Component]:
        self._check_layer(layer)
        if not 0 <= group < self.num_kv_heads:
            raise ValueError(f"kv group out of range: {group}")
        start = group * self.group_size
        return [Component(layer, "head", head) for head in range(start, start + self.group_size)]

    # -- slicing --------------------------------------------------------

    def edits_for_components(self, components: Iterable[str | Component]) -> list[TensorEdit]:
        """Aggregate selected components into disjoint + K/V strength edits."""
        parsed = [c if isinstance(c, Component) else Component.parse(c) for c in components]
        by_layer: dict[int, list[Component]] = {}
        for component in parsed:
            self._check_component(component)
            by_layer.setdefault(component.layer, []).append(component)

        edits: list[TensorEdit] = []
        for layer in sorted(by_layer):
            layer_components = by_layer[layer]
            heads = {c.head for c in layer_components if c.kind == "head"}
            has_mlp = any(c.kind == "mlp" for c in layer_components)

            for head in sorted(head for head in heads if head is not None):
                start = head * self.head_dim
                end = start + self.head_dim
                edits.append(
                    TensorEdit(
                        self.attn_tensor(layer, "q_proj"), 0, start, end, 1.0,
                        Component(layer, "head", head).id,
                    )
                )
                edits.append(
                    TensorEdit(
                        self.attn_tensor(layer, "o_proj"), 1, start, end, 1.0,
                        Component(layer, "head", head).id,
                    )
                )

            group_counts: dict[int, int] = {}
            for head in heads:
                if head is None:
                    continue
                group = self.kv_group(head)
                group_counts[group] = group_counts.get(group, 0) + 1
            for group, count in sorted(group_counts.items()):
                start = group * self.head_dim
                end = start + self.head_dim
                strength = count / self.group_size
                label = Component(layer, "head", group * self.group_size).id
                edits.append(
                    TensorEdit(self.attn_tensor(layer, "k_proj"), 0, start, end, strength, label)
                )
                edits.append(
                    TensorEdit(self.attn_tensor(layer, "v_proj"), 0, start, end, strength, label)
                )

            if has_mlp:
                label = Component(layer, "mlp").id
                edits.append(
                    TensorEdit(self.mlp_tensor(layer, "gate_proj"), 0, 0,
                               self.intermediate_size, 1.0, label)
                )
                edits.append(
                    TensorEdit(self.mlp_tensor(layer, "up_proj"), 0, 0,
                               self.intermediate_size, 1.0, label)
                )
                edits.append(
                    TensorEdit(self.mlp_tensor(layer, "down_proj"), 0, 0,
                               self.hidden_size, 1.0, label)
                )

        return sorted(edits, key=lambda e: (e.tensor, e.dim, e.start))

    def edits_by_tensor(
        self, components: Iterable[str | Component]
    ) -> dict[str, list[TensorEdit]]:
        grouped: dict[str, list[TensorEdit]] = {}
        for edit in self.edits_for_components(components):
            grouped.setdefault(edit.tensor, []).append(edit)
        return grouped

    def component_parameter_count(self, component: str | Component) -> int:
        component = component if isinstance(component, Component) else Component.parse(component)
        self._check_component(component)
        if component.kind == "mlp":
            return 3 * self.hidden_size * self.intermediate_size
        q_o = 2 * self.hidden_size * self.head_dim
        kv_share = 2 * self.hidden_size * self.head_dim / self.group_size
        return int(q_o + kv_share)

    def tensor_parameter_count(self, tensor: str) -> int:
        """Number of elements in an editable tensor (all dims are full)."""
        layer, kind, name = self._parse_tensor_name(tensor)
        if kind == "attn":
            if name in ("q_proj", "o_proj"):
                return self.hidden_size * (self.num_heads * self.head_dim)
            return self.hidden_size * (self.num_kv_heads * self.head_dim)
        return self.hidden_size * self.intermediate_size

    def tensor_dim_size(self, tensor: str, dim: int) -> int:
        """Length of one dimension of an editable tensor.

        Slice coverage is checked against this, not against the total element
        count: MLP components edit whole tensors along dimension 0.
        """
        _, kind, name = self._parse_tensor_name(tensor)
        if dim not in (0, 1):
            raise ValueError(f"editable tensors are 2D, got dim {dim}")
        if kind == "attn":
            if name == "q_proj":
                # (num_heads * head_dim, hidden_size)
                return self.num_heads * self.head_dim if dim == 0 else self.hidden_size
            if name in ("k_proj", "v_proj"):
                # (num_kv_heads * head_dim, hidden_size)
                return self.num_kv_heads * self.head_dim if dim == 0 else self.hidden_size
            # o_proj: (hidden_size, num_heads * head_dim)
            return self.hidden_size if dim == 0 else self.num_heads * self.head_dim
        if name in ("gate_proj", "up_proj"):
            return self.intermediate_size if dim == 0 else self.hidden_size
        return self.hidden_size if dim == 0 else self.intermediate_size

    def total_editable_parameters(self) -> int:
        return sum(self.tensor_parameter_count(t) for t in self.editable_tensor_names())

    # -- validation -----------------------------------------------------

    def validate_partition(self) -> None:
        """Every editable tensor must be covered exactly once by all components."""
        all_components = self.all_components()
        edits_by_tensor = self.edits_by_tensor(all_components)
        expected_tensors = set(self.editable_tensor_names())
        assert set(edits_by_tensor) == expected_tensors, (
            f"edit tensor set mismatch: {set(edits_by_tensor) ^ expected_tensors}"
        )

        for tensor, edits in edits_by_tensor.items():
            _, _, name = self._parse_tensor_name(tensor)
            # o_proj is sliced along input columns; everything else (including
            # the full-tensor MLP slices) is represented on dimension 0.
            dim = 1 if name == "o_proj" else 0
            size = self.tensor_dim_size(tensor, dim)
            ordered = sorted(edits, key=lambda e: e.start)
            cursor = 0
            for edit in ordered:
                if edit.dim != dim:
                    raise AssertionError(f"{tensor}: dim {edit.dim} != expected {dim}")
                if edit.start != cursor:
                    raise AssertionError(
                        f"{tensor}: gap/overlap at {cursor}, next slice starts {edit.start}"
                    )
                if edit.end <= edit.start:
                    raise AssertionError(f"{tensor}: empty slice {edit}")
                cursor = edit.end
            if cursor != size:
                raise AssertionError(f"{tensor}: coverage ended at {cursor}, expected {size}")

        forbidden = set(edits_by_tensor) & set(self.non_editable_tensor_names())
        if forbidden:
            raise AssertionError(f"editable mapping touched protected tensors: {forbidden}")

    def assert_matches_model(self, model: object) -> None:
        """Verify real HF parameter names/shapes against this layout.

        Uses ``named_parameters()`` on the loaded checkpoint. This is the D1
        guard against relying on the spec's names without checking the model.
        """
        parameters = dict(model.named_parameters())  # type: ignore[attr-defined]
        expected: dict[str, tuple[int, ...]] = {}
        for layer in range(self.num_layers):
            expected[self.attn_tensor(layer, "q_proj")] = (
                self.num_heads * self.head_dim,
                self.hidden_size,
            )
            expected[self.attn_tensor(layer, "k_proj")] = (
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            )
            expected[self.attn_tensor(layer, "v_proj")] = (
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            )
            expected[self.attn_tensor(layer, "o_proj")] = (
                self.hidden_size,
                self.num_heads * self.head_dim,
            )
            expected[self.mlp_tensor(layer, "gate_proj")] = (
                self.intermediate_size,
                self.hidden_size,
            )
            expected[self.mlp_tensor(layer, "up_proj")] = (
                self.intermediate_size,
                self.hidden_size,
            )
            expected[self.mlp_tensor(layer, "down_proj")] = (
                self.hidden_size,
                self.intermediate_size,
            )

        missing = sorted(set(expected) - set(parameters))
        if missing:
            raise AssertionError(f"model is missing {len(missing)} expected tensors: {missing[:5]}")
        wrong = {
            name: (tuple(parameters[name].shape), shape)
            for name, shape in expected.items()
            if tuple(parameters[name].shape) != shape
        }
        if wrong:
            raise AssertionError(f"shape mismatch on {len(wrong)} tensors: {list(wrong.items())[:3]}")

        # The lm_head is tied; it must never be treated as editable.
        if "lm_head.weight" in parameters:
            tied = parameters.get("model.embed_tokens.weight")
            if tied is not None and parameters["lm_head.weight"].data_ptr() != tied.data_ptr():
                raise AssertionError("lm_head is not tied to embed_tokens on this checkpoint")
        for name in self.non_editable_tensor_names():
            if name not in parameters and name != "lm_head.weight":
                raise AssertionError(f"expected protected tensor is missing: {name}")

    def model_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        """Shapes for a synthetic tiny model in tests (full-size by default)."""
        shapes: dict[str, tuple[int, ...]] = {
            f"{self.prefix}.embed_tokens.weight": (262144, self.hidden_size),
            f"{self.prefix}.norm.weight": (self.hidden_size,),
            "lm_head.weight": (262144, self.hidden_size),
        }
        for layer in range(self.num_layers):
            shapes[f"{self.layer_prefix(layer)}.input_layernorm.weight"] = (self.hidden_size,)
            shapes[self.attn_tensor(layer, "q_proj")] = (
                self.num_heads * self.head_dim,
                self.hidden_size,
            )
            shapes[self.attn_tensor(layer, "k_proj")] = (
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            )
            shapes[self.attn_tensor(layer, "v_proj")] = (
                self.num_kv_heads * self.head_dim,
                self.hidden_size,
            )
            shapes[self.attn_tensor(layer, "o_proj")] = (
                self.hidden_size,
                self.num_heads * self.head_dim,
            )
            shapes[self.mlp_tensor(layer, "gate_proj")] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[self.mlp_tensor(layer, "up_proj")] = (
                self.intermediate_size,
                self.hidden_size,
            )
            shapes[self.mlp_tensor(layer, "down_proj")] = (
                self.hidden_size,
                self.intermediate_size,
            )
        return shapes

    # -- internals ------------------------------------------------------

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise ValueError(f"layer out of range: {layer}")

    def _check_head(self, head: int) -> None:
        if not 0 <= head < self.num_heads:
            raise ValueError(f"head out of range: {head}")

    def _check_component(self, component: Component) -> None:
        self._check_layer(component.layer)
        if component.kind == "head":
            assert component.head is not None
            self._check_head(component.head)

    def _parse_tensor_name(self, tensor: str) -> tuple[int, Literal["attn", "mlp"], str]:
        parts = tensor.split(".")
        try:
            layer = int(parts[2])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"not an editable Cohere2 tensor: {tensor}") from exc
        self._check_layer(layer)
        if len(parts) != 5 or parts[0] != self.prefix or parts[1] != "layers":
            raise ValueError(f"not an editable Cohere2 tensor: {tensor}")
        if parts[3] == "self_attn" and parts[4] in ATTENTION_TENSORS:
            return layer, "attn", parts[4]
        if parts[3] == "mlp" and parts[4] in MLP_TENSORS:
            return layer, "mlp", parts[4]
        raise ValueError(f"not an editable Cohere2 tensor: {tensor}")

    def validate_model_state_dict(self, state: Mapping[str, object]) -> None:
        """Check that a state dict contains every tensor the editor will touch."""
        missing = [name for name in self.editable_tensor_names() if name not in state]
        if missing:
            raise KeyError(f"state dict is missing {len(missing)} editable tensors: {missing[:5]}")
