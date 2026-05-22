# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

"""Compare MultiMolecule model ports against the golden upstream fixtures.

Fixtures live in the ``MultiMolecule/golden`` submodule under
``models/<model>/<case>/``. Set ``GOLDEN_ROOT`` to override the submodule path.
If no fixtures are visible, the suite collects zero parameters and the file is
skipped rather than failed, so a clone without the submodule does not break CI.

Each fixture's ``meta.json`` may declare ``multimolecule.checkpoint`` (a
non-default HF repo id) and ``multimolecule.automodel`` (a class name in
``multimolecule`` or ``transformers``) to override the convention defaults of
``multimolecule/<model>(-<case>)`` and ``AutoModel`` respectively.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from safetensors.torch import load_file

import multimolecule  # noqa: F401  registers MultiMolecule configs with transformers
import transformers
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = Path(os.environ.get("GOLDEN_ROOT", REPO_ROOT / "golden"))


def _hf_id(meta: dict) -> str:
    explicit = meta.get("multimolecule", {}).get("checkpoint")
    if explicit:
        return explicit
    if meta["case"] == "default":
        return f"multimolecule/{meta['model']}"
    return f"multimolecule/{meta['model']}-{meta['case']}"


def _model_class(meta: dict) -> Any:
    name = meta.get("multimolecule", {}).get("automodel")
    if not name:
        return AutoModel
    cls = getattr(multimolecule, name, None) or getattr(transformers, name, None)
    if cls is None:
        raise ImportError(f"Cannot resolve AutoModel class {name!r}")
    return cls


def _output_handlers() -> dict[str, Callable]:
    return {
        "attentions": lambda out: torch.stack(out.attentions, dim=0),
        "hidden_states": lambda out: torch.stack(out.hidden_states, dim=0),
    }


def discover_fixtures() -> list[Path]:
    if not GOLDEN_ROOT.exists():
        return []
    return sorted(path for path in (GOLDEN_ROOT / "models").glob("*/*") if path.is_dir())


@pytest.mark.parametrize(
    "fixture_dir",
    discover_fixtures(),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_faithfulness(fixture_dir: Path) -> None:
    meta = json.loads((fixture_dir / "meta.json").read_text())
    inputs = load_file(str(fixture_dir / "inputs.safetensors"))
    expected = load_file(str(fixture_dir / "expected.safetensors"))

    if set(expected) != set(meta["outputs"]):
        pytest.fail(
            f"{fixture_dir}: meta.outputs {sorted(meta['outputs'])} does not match "
            f"expected.safetensors keys {sorted(expected)}"
        )

    handlers = _output_handlers()
    needs_attentions = "attentions" in expected
    needs_hidden_states = "hidden_states" in expected

    model = _model_class(meta).from_pretrained(_hf_id(meta))
    model.eval()
    with torch.no_grad():
        actual = model(
            **{key: value for key, value in inputs.items()},
            output_attentions=needs_attentions,
            output_hidden_states=needs_hidden_states,
            return_dict=True,
        )

    atol = float(meta["tolerance"]["atol"])
    rtol = float(meta["tolerance"]["rtol"])
    for key, expected_tensor in expected.items():
        getter = handlers.get(key, lambda out, key=key: getattr(out, key))
        actual_tensor = getter(actual)
        if actual_tensor is None:
            pytest.fail(f"{fixture_dir}: model output {key!r} is None")
        if actual_tensor.shape != expected_tensor.shape:
            pytest.fail(
                f"{fixture_dir}: {key} shape mismatch "
                f"actual={tuple(actual_tensor.shape)} expected={tuple(expected_tensor.shape)}"
            )
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)
