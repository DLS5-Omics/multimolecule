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

from __future__ import annotations

import string

import numpy as np
from scipy.spatial import cKDTree

from multimolecule.utils.rna.secondary_structure.notations import dot_bracket_to_pairs

_PRIMARY_SPACE = 1.0
_PAIR_SPACE = 1.0
_RELAX_MAX_LENGTH = 1000
_RELAX_MIN_DISTANCE = 0.9
_RELAX_ITERATIONS = 120
_RELAX_STEP = 0.08
_RELAX_SPRING = 0.10
_RELAX_ANCHOR = 0.01
_RELAX_CONVERGENCE_TOL = 5e-4
_LAYOUT_BRACKETS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")] + list(
    zip(string.ascii_uppercase, string.ascii_lowercase)
)


def secondary_structure_layout(
    dot_bracket: str,
    *,
    normalize: bool = True,
    rotate: float = 0.0,
    half_span: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D coordinates for a dot-bracket RNA secondary structure.

    Args:
        dot_bracket: Dot-bracket notation.
        normalize: Whether to center and isotropically scale coordinates.
        rotate: Counter-clockwise rotation in degrees.
        half_span: Half-span of the normalized coordinate box.

    Returns:
        Two numpy arrays containing x and y coordinates.

    Examples:
        >>> from multimolecule.visualization.rna import secondary_structure_layout
        >>> xs, ys = secondary_structure_layout("(((...)))")
        >>> xs.shape, ys.shape
        ((9,), (9,))
    """
    pairs = dot_bracket_to_pairs(dot_bracket)
    layout_pairs = _primary_layout_pairs(dot_bracket, pairs)
    xs, ys = _tree_layout(len(dot_bracket), layout_pairs)
    xs, ys = _rotate_coords(xs, ys, rotate)
    if normalize:
        xs, ys = _normalize_coords(xs, ys, half_span=half_span)
    return xs, ys


class _LayoutNode:
    def __init__(self, index: int = -1, pair: tuple[int, int] | None = None) -> None:
        self.index = index
        self.pair = pair
        self.children: list[_LayoutNode] = []
        self.x = 0.0
        self.y = 0.0
        self.direction_x = 0.0
        self.direction_y = 1.0


def _primary_layout_pairs(dot_bracket: str, pairs: np.ndarray) -> np.ndarray:
    for opener, closer in _LAYOUT_BRACKETS:
        tier_pairs = _bracket_pairs(dot_bracket, opener, closer)
        if tier_pairs.size:
            return tier_pairs
    return pairs


def _tree_layout(length: int, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if length == 0:
        return np.empty(0), np.empty(0)
    if length == 1:
        return np.array([0.0]), np.array([0.0])

    pairmap = _pairmap(length, pairs)
    root = _build_layout_tree(pairmap, 0, length - 1)
    _place_layout_node(root, None, 0.0, 0.0, 0.0, 1.0)

    xs = np.zeros(length, dtype=float)
    ys = np.zeros(length, dtype=float)
    _collect_layout_coords(root, xs, ys)
    xs, ys = _relax_layout(xs, ys, pairs)
    return xs, ys


def _pairmap(length: int, pairs: np.ndarray) -> list[int]:
    pairmap = [-1] * length
    for i, j in pairs:
        pairmap[int(i)] = int(j)
        pairmap[int(j)] = int(i)
    return pairmap


def _bracket_pairs(dot_bracket: str, opener: str, closer: str) -> np.ndarray:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, symbol in enumerate(dot_bracket):
        if symbol == opener:
            stack.append(index)
        elif symbol == closer and stack:
            pairs.append((stack.pop(), index))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    pairs.sort()
    return np.asarray(pairs, dtype=int)


def _build_layout_tree(pairmap: list[int], start: int, stop: int) -> _LayoutNode:
    root = _LayoutNode()
    index = start
    while index <= stop:
        partner = pairmap[index]
        if partner > index:
            _add_layout_segment(pairmap, root, index, partner)
            index = partner + 1
        elif partner < 0:
            root.children.append(_LayoutNode(index=index))
            index += 1
        else:
            index += 1
    return root


def _add_layout_segment(pairmap: list[int], parent: _LayoutNode, start: int, stop: int) -> None:
    if start > stop:
        return

    if pairmap[start] == stop:
        node = _LayoutNode(pair=(start, stop))
        _add_layout_segment(pairmap, node, start + 1, stop - 1)
        parent.children.append(node)
        return

    node = _LayoutNode()
    index = start
    while index <= stop:
        partner = pairmap[index]
        if partner > index:
            _add_layout_segment(pairmap, node, index, partner)
            index = partner + 1
        elif partner < 0:
            node.children.append(_LayoutNode(index=index))
            index += 1
        else:
            index += 1
    parent.children.append(node)


def _place_layout_node(
    node: _LayoutNode,
    parent: _LayoutNode | None,
    start_x: float,
    start_y: float,
    direction_x: float,
    direction_y: float,
) -> None:
    direction_x, direction_y = _normalize_vector(direction_x, direction_y)
    node.direction_x = direction_x
    node.direction_y = direction_y

    if len(node.children) <= 1:
        node.x = start_x
        node.y = start_y
        if node.children:
            child = node.children[0]
            if child.pair is None and child.index < 0:
                child_x = start_x
                child_y = start_y
            else:
                child_x = start_x + direction_x * _PRIMARY_SPACE
                child_y = start_y + direction_y * _PRIMARY_SPACE
            _place_layout_node(
                child,
                node,
                child_x,
                child_y,
                direction_x,
                direction_y,
            )
        return

    pair_children = sum(child.pair is not None for child in node.children)
    circumference = (len(node.children) + 1) * _PRIMARY_SPACE + (pair_children + 1) * _PAIR_SPACE
    radius = circumference / (2 * np.pi)
    if parent is None:
        node.x = direction_x * radius
        node.y = direction_y * radius
    else:
        node.x = parent.x + direction_x * radius
        node.y = parent.y + direction_y * radius

    cross_x = -direction_y
    cross_y = direction_x
    walker = _PAIR_SPACE / 2
    for child in node.children:
        walker += _PRIMARY_SPACE
        if child.pair is not None:
            walker += _PAIR_SPACE / 2
        angle = walker / circumference * 2 * np.pi - np.pi / 2
        child_x = node.x + np.cos(angle) * cross_x * radius + np.sin(angle) * direction_x * radius
        child_y = node.y + np.cos(angle) * cross_y * radius + np.sin(angle) * direction_y * radius
        child_direction_x, child_direction_y = _normalize_vector(child_x - node.x, child_y - node.y)
        _place_layout_node(child, node, child_x, child_y, child_direction_x, child_direction_y)
        if child.pair is not None:
            walker += _PAIR_SPACE / 2


def _relax_layout(xs: np.ndarray, ys: np.ndarray, pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = xs.size
    if length < 4 or length > _RELAX_MAX_LENGTH:
        return xs, ys

    pos = np.column_stack((xs, ys)).astype(float, copy=True)
    initial = pos.copy()

    # Pre-compute an int64 key set of (i, j) pairs (i < j) whose mutual distance is
    # already constrained by other forces — backbone neighbours (|i - j| ≤ 2) and
    # paired bases. These must be excluded from the repulsion term.
    ignore_keys: set[int] = set()
    for i in range(length):
        for j in range(i + 1, min(length, i + 3)):
            ignore_keys.add(_pair_key(i, j, length))
    edge_i: list[int] = list(range(length - 1))
    edge_j: list[int] = list(range(1, length))
    targets: list[float] = [_PRIMARY_SPACE] * (length - 1)
    for i, j in pairs:
        a = int(min(i, j))
        b = int(max(i, j))
        ignore_keys.add(_pair_key(a, b, length))
        edge_i.append(a)
        edge_j.append(b)
        targets.append(_PAIR_SPACE)

    ignore_arr = np.array(sorted(ignore_keys), dtype=np.int64)
    edge_i_array = np.asarray(edge_i, dtype=int)
    edge_j_array = np.asarray(edge_j, dtype=int)
    target_array = np.asarray(targets, dtype=float)

    step = _RELAX_STEP
    for _ in range(_RELAX_ITERATIONS):
        # Spatial index reduces repulsion from O(L²) to O(L log L + k) where k is
        # the number of close pairs returned by query_pairs.
        tree = cKDTree(pos)
        close_pairs = tree.query_pairs(_RELAX_MIN_DISTANCE, output_type="ndarray")
        force = np.zeros_like(pos)
        if close_pairs.size:
            lo = close_pairs[:, 0].astype(np.int64)
            hi = close_pairs[:, 1].astype(np.int64)
            keys = lo * length + hi
            keep = ~np.isin(keys, ignore_arr, assume_unique=False)
            if keep.any():
                lo = lo[keep]
                hi = hi[keep]
                delta = pos[lo] - pos[hi]
                distance = np.linalg.norm(delta, axis=1) + 1e-9
                magnitude = (_RELAX_MIN_DISTANCE - distance) / _RELAX_MIN_DISTANCE
                impulse = (delta / distance[:, None]) * magnitude[:, None]
                np.add.at(force, lo, impulse)
                np.add.at(force, hi, -impulse)

        edge_delta = pos[edge_j_array] - pos[edge_i_array]
        edge_length = np.linalg.norm(edge_delta, axis=1) + 1e-9
        spring = (
            _RELAX_SPRING * ((edge_length - target_array) / target_array)[:, None] * edge_delta / edge_length[:, None]
        )
        np.add.at(force, edge_i_array, spring)
        np.add.at(force, edge_j_array, -spring)

        force += _RELAX_ANCHOR * (initial - pos)
        max_displacement = float(step * np.max(np.linalg.norm(force, axis=1))) if length else 0.0
        pos += step * force
        pos -= pos.mean(axis=0, keepdims=True) - initial.mean(axis=0, keepdims=True)
        step *= 0.995
        if max_displacement < _RELAX_CONVERGENCE_TOL:
            break

    return pos[:, 0], pos[:, 1]


def _pair_key(i: int, j: int, length: int) -> int:
    return int(i) * length + int(j)


def _collect_layout_coords(node: _LayoutNode, xs: np.ndarray, ys: np.ndarray) -> None:
    if node.pair is not None:
        i, j = node.pair
        cross_x = -node.direction_y
        cross_y = node.direction_x
        xs[i] = node.x + cross_x * _PAIR_SPACE / 2
        ys[i] = node.y + cross_y * _PAIR_SPACE / 2
        xs[j] = node.x - cross_x * _PAIR_SPACE / 2
        ys[j] = node.y - cross_y * _PAIR_SPACE / 2
    elif node.index >= 0:
        xs[node.index] = node.x
        ys[node.index] = node.y

    for child in node.children:
        _collect_layout_coords(child, xs, ys)


def _normalize_vector(x: float, y: float) -> tuple[float, float]:
    norm = float(np.hypot(x, y))
    if norm <= 1e-12:
        return 0.0, 1.0
    return x / norm, y / norm


def _rotate_coords(xs: np.ndarray, ys: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    if angle_degrees == 0 or xs.size == 0:
        return xs, ys
    theta = np.radians(angle_degrees)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    center_x, center_y = float(xs.mean()), float(ys.mean())
    centered_x, centered_y = xs - center_x, ys - center_y
    return (
        cos_theta * centered_x - sin_theta * centered_y + center_x,
        sin_theta * centered_x + cos_theta * centered_y + center_y,
    )


def _normalize_coords(xs: np.ndarray, ys: np.ndarray, half_span: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    if xs.size == 0:
        return xs, ys
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    scale = (2 * half_span) / span
    return (xs - center_x) * scale, (ys - center_y) * scale
