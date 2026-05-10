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

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple, TypeAlias, Union

import numpy as np

from ..palettes import _STRUCTURAL_CLASS_TO_LOOP_TYPE, PAIR_PROBABILITY_CMAP, STRUCTURAL_CLASS_PALETTE

LW_EDGES: tuple[str, ...] = ("WC", "H", "S")
LW_ORIENTATIONS: tuple[str, ...] = ("cis", "trans")
_TOPOLOGY_REGION_KINDS: frozenset[str] = frozenset({"hairpin", "bulge", "internal", "multiloop", "external"})


@dataclass(frozen=True)
class LeontisWesthof:
    """Leontis-Westhof annotation for a single base pair.

    Glyph mapping used by the renderer: WC -> circle, H -> square, S -> triangle.
    cis fills the glyph; trans leaves it hollow.

    Args:
        edge_5p: Edge identifier on the 5' nucleotide. One of ``"WC"`` (Watson-Crick),
            ``"H"`` (Hoogsteen), ``"S"`` (Sugar).
        edge_3p: Edge identifier on the 3' nucleotide. Same set as ``edge_5p``.
        orientation: ``"cis"`` (filled glyph) or ``"trans"`` (hollow glyph). Defaults to
            ``"cis"`` because canonical Watson-Crick pairs are cis by convention.

    Raises:
        ValueError: If any edge is not in ``{"WC", "H", "S"}`` or orientation is not in
            ``{"cis", "trans"}``.

    Examples:
        >>> LeontisWesthof("WC", "WC").orientation
        'cis'
    """

    edge_5p: str
    edge_3p: str
    orientation: str = "cis"  # Canonical Watson-Crick pairs are cis by convention.

    def __post_init__(self) -> None:
        if self.edge_5p not in LW_EDGES:
            raise ValueError(f"edge_5p must be one of {LW_EDGES}, got {self.edge_5p!r}.")
        if self.edge_3p not in LW_EDGES:
            raise ValueError(f"edge_3p must be one of {LW_EDGES}, got {self.edge_3p!r}.")
        if self.orientation not in LW_ORIENTATIONS:
            raise ValueError(f"orientation must be one of {LW_ORIENTATIONS}, got {self.orientation!r}.")


class PairProbabilityTrack:
    """
    L×L matrix of base-pair probabilities.

    Renders as: faint chords/arcs under MFE pairs on planar; arc color + width on arc;
    chord color + width on circular. Treated as a no-op on contact_map
    because the contact map already shows the matrix directly.

    The matrix may be symmetric or upper-triangular; the renderer only inspects the upper
    triangle. Values must be finite probabilities in ``[0, 1]``. Pairs with probability below
    ``threshold`` are skipped. If ``haze_threshold`` is set, pairs with probability in
    ``[haze_threshold, threshold)`` are still drawn but as a faint background "haze" layer,
    keeping low-probability pairs visible without cluttering the main rendering.

    Args:
        matrix: 2-D array-like ``(L, L)`` of pair probabilities. Symmetric or upper-triangular.
        cmap: Matplotlib colormap name for the probability scale.
        threshold: Lower probability cutoff for main pair rendering. Pairs below this are
            skipped or sent to the haze layer.
        width_by_probability: When ``True``, line width scales with probability so high-confidence
            pairs visually dominate.
        haze_threshold: Optional secondary cutoff in ``[0, threshold)``. Pairs at or above
            this cutoff but below ``threshold`` render as a faint haze layer instead of being
            dropped entirely.
        name: Display name used in the colorbar.

    Raises:
        ValueError: From [validate][] when matrix shape disagrees with structure length,
            values are non-finite or outside ``[0, 1]``, or ``haze_threshold >= threshold``.
    """

    matrix: Any
    cmap: str
    threshold: float
    width_by_probability: bool
    haze_threshold: float | None
    name: str

    def __init__(
        self,
        matrix: Any,
        *,
        cmap: str = PAIR_PROBABILITY_CMAP,
        threshold: float = 0.01,
        width_by_probability: bool = True,
        haze_threshold: float | None = None,
        name: str = "pair probability",
    ) -> None:
        self.matrix = matrix
        self.cmap = cmap
        self.threshold = float(threshold)
        self.width_by_probability = bool(width_by_probability)
        self.haze_threshold = None if haze_threshold is None else float(haze_threshold)
        self.name = name

    def matrix_array(self) -> np.ndarray:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"PairProbabilityTrack matrix must be 2D, but got shape {matrix.shape}.")
        return matrix

    def validate(self, length: int) -> None:
        matrix = self.matrix_array()
        if matrix.shape != (length, length):
            raise ValueError(
                f"PairProbabilityTrack matrix shape {matrix.shape} != (length, length) = ({length}, {length})."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("PairProbabilityTrack matrix values must be finite.")
        if (matrix < 0.0).any() or (matrix > 1.0).any():
            raise ValueError("PairProbabilityTrack matrix values must be between 0.0 and 1.0.")
        _check_probability_value(self.threshold, "PairProbabilityTrack.threshold")
        if self.haze_threshold is not None:
            _check_probability_value(self.haze_threshold, "PairProbabilityTrack.haze_threshold")
            if self.haze_threshold >= self.threshold:
                raise ValueError("PairProbabilityTrack.haze_threshold must be smaller than threshold.")


class SequenceDiffTrack:
    """
    Highlight positions where the plot's sequence differs from a reference sequence.

    Renders as: emphasized bead outline on planar, emphasized base label on arc and circular,
    and emphasized tick label on contact_map. Useful for visualizing single-nucleotide variants
    (the riboSNitch case study in Léger et al. 2019).

    Args:
        reference_sequence: Reference RNA sequence to diff against. Comparison is
            case-insensitive.
        highlight_color: Color used for emphasized outlines / labels / ticks.
        bold_label: When ``True``, base labels at differing positions are rendered bold.
    """

    reference_sequence: str
    highlight_color: str
    bold_label: bool

    def __init__(
        self,
        reference_sequence: str,
        *,
        highlight_color: str = "#d62728",
        bold_label: bool = True,
    ) -> None:
        self.reference_sequence = reference_sequence
        self.highlight_color = highlight_color
        self.bold_label = bool(bold_label)

    def diff_mask(self, sequence: str | None, length: int) -> np.ndarray:
        if sequence is None:
            raise ValueError("sequence is required when using SequenceDiffTrack.")
        if len(self.reference_sequence) != length:
            raise ValueError(
                "SequenceDiffTrack reference_sequence length "
                f"{len(self.reference_sequence)} != sequence length {length}."
            )
        mask = np.zeros(length, dtype=bool)
        for i in range(length):
            if sequence[i].upper() != self.reference_sequence[i].upper():
                mask[i] = True
        return mask


class BaseValueTrack:
    """
    Per-position scalar overlay (SHAPE / DMS reactivity, attention, model confidence, ...).

    Renders as: bead fill color on planar, line markers on arc, outer-ring wedges on circular,
    sidebar on contact_map. NaN values render as a neutral gray.

    Args:
        values: Length-L array-like of floats. NaN renders as neutral gray.
        cmap: Matplotlib colormap name.
        vmin: Lower color-scale bound. ``None`` uses the data min.
        vmax: Upper color-scale bound. ``None`` uses the data max.
        name: Display name used in the colorbar.
        show_colorbar: Whether the renderer attaches a colorbar.

    Raises:
        ValueError: From [validate][] when ``values`` is not 1-D or its length disagrees
            with the structure length.
    """

    values: Sequence[float]
    cmap: str
    vmin: float | None
    vmax: float | None
    name: str
    show_colorbar: bool

    def __init__(
        self,
        values: Sequence[float],
        *,
        cmap: str = "Reds",
        vmin: float | None = None,
        vmax: float | None = None,
        name: str = "value",
        show_colorbar: bool = True,
    ) -> None:
        self.values = values
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.name = name
        self.show_colorbar = bool(show_colorbar)

    def values_array(self) -> np.ndarray:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError(f"BaseValueTrack values must be 1D, but got shape {values.shape}.")
        return values

    def validate(self, length: int) -> None:
        values = self.values_array()
        if values.shape[0] != length:
            raise ValueError(f"BaseValueTrack length {values.shape[0]} != structure length {length}.")


class BaseCategoryTrack:
    """
    Per-position discrete category overlay (bpRNA loop type, custom annotation, ...).

    Renders as: bead fill on planar, line markers on arc, outer-ring wedges on circular,
    sidebar on contact_map — using a discrete palette and a categorical legend instead of a colorbar.

    Args:
        categories: Length-L sequence of category labels (str or int).
        palette: Optional mapping from category to color. ``None`` falls back to a default
            palette informed by [LOOP_TYPE_PALETTE][] and a categorical cycle.
        name: Display name used in the legend.
        show_legend: Whether the renderer attaches a legend.

    Raises:
        ValueError: From [validate][] when ``categories`` length disagrees with the
            structure length.
    """

    categories: Sequence[Union[str, int]]
    palette: Mapping[Union[str, int], str] | None
    name: str
    show_legend: bool

    def __init__(
        self,
        categories: Sequence[Union[str, int]],
        *,
        palette: Mapping[Union[str, int], str] | None = None,
        name: str = "category",
        show_legend: bool = True,
    ) -> None:
        self.categories = list(categories)
        self.palette = None if palette is None else dict(palette)
        self.name = name
        self.show_legend = bool(show_legend)

    def validate(self, length: int) -> None:
        if len(self.categories) != length:
            raise ValueError(f"BaseCategoryTrack length {len(self.categories)} != structure length {length}.")

    @classmethod
    def from_topology(
        cls,
        dot_bracket: str,
        *,
        mode: str = "loop_type",
        name: str | None = None,
    ) -> BaseCategoryTrack:
        """
        Build a track from bpRNA annotation of the given dot-bracket.

        Args:
            dot_bracket: Dot-bracket notation.
            mode: "loop_type" maps per-position bpRNA codes to friendly names
                (stem / hairpin / bulge / internal / multiloop / external / end).
                "structural_class" keeps the raw bpRNA characters (S / H / B / I / M / X / E).
            name: Optional track name for the legend.
        """
        categories = _bpRNA_categories(dot_bracket, mode=mode)
        palette: dict[Union[str, int], str] | None = None
        if mode == "structural_class":
            palette = {}
            for key, color in STRUCTURAL_CLASS_PALETTE.items():
                palette[key] = color
        return cls(categories=categories, palette=palette, name=name or mode)

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        *,
        palette: str = "nature",
        name: str | None = "sequence",
    ) -> BaseCategoryTrack:
        """
        Build a track of nucleotide categories from a sequence using a nucleotide palette.

        Produces the CS²BP² outer-ring look: one colored block per nucleotide, with the palette
        drawn from [NUCLEOTIDE_PALETTES][] (so ``A`` / ``C`` / ``G`` / ``U`` each get a
        distinct color). Unknown bases are normalized to the ``N`` category and use the ``N`` color.

        Args:
            sequence: RNA (or DNA) sequence. Case-insensitive.
            palette: Name from [NUCLEOTIDE_PALETTES][].
            name: Optional track name shown in the legend.
        """
        from ..palettes import NUCLEOTIDE_PALETTES

        if palette not in NUCLEOTIDE_PALETTES:
            choices = ", ".join(sorted(NUCLEOTIDE_PALETTES))
            raise ValueError(f"Unknown nucleotide palette: {palette}. Available: {choices}.")
        pal = NUCLEOTIDE_PALETTES[palette]
        track_palette: dict[Union[str, int], str] = {}
        for base, color in pal.items():
            track_palette[base] = color
        categories = []
        for base in sequence:
            base = base.upper()
            categories.append(base if base in pal else "N")
        return cls(categories=categories, palette=track_palette, name=name or "sequence")


class RegionTrack:
    """
    Named regions / motifs / domains along the sequence.

    Each region is ``(start, stop, label)`` with both endpoints inclusive (0-indexed).
    Renders as: highlighted backbone span on planar/circular, x-range band on arc,
    diagonal box on contact_map.

    Args:
        regions: Sequence of ``(start, stop, label)`` triples. Both endpoints are inclusive.
        colors: Region colors. May be a single color (broadcast to all regions), a sequence
            cycled across regions, or ``None`` to use a default palette.
        name: Track name (currently unused in legends but kept for symmetry with other tracks).
        show_labels: Whether to render the per-region label.
        alpha: Region fill alpha, in ``[0, 1]``.

    Raises:
        ValueError: If any region has ``stop < start``, or — from [validate][] — any
            region falls outside ``[0, length)``.
    """

    regions: Sequence[Tuple[int, int, str]]
    colors: Union[Sequence[str], str, None]
    name: str
    show_labels: bool
    alpha: float

    def __init__(
        self,
        regions: Sequence[Tuple[int, int, str]] = (),
        *,
        colors: Union[Sequence[str], str, None] = None,
        name: str = "regions",
        show_labels: bool = True,
        alpha: float = 0.25,
    ) -> None:
        self.regions = [(int(start), int(stop), str(label)) for start, stop, label in regions]
        self.colors = colors
        self.name = name
        self.show_labels = bool(show_labels)
        self.alpha = float(alpha)
        for start, stop, _ in self.regions:
            if stop < start:
                raise ValueError(f"RegionTrack region stop {stop} < start {start}.")

    def validate(self, length: int) -> None:
        for start, stop, _ in self.regions:
            if start < 0 or stop >= length:
                raise ValueError(f"RegionTrack region ({start}, {stop}) is outside structure length {length}.")

    @classmethod
    def from_topology(
        cls,
        dot_bracket: str,
        *,
        loop_kinds: Sequence[str] = ("hairpin", "multiloop"),
        sequence: str | None = None,
        name: str | None = None,
        colors: Union[Sequence[str], str, None] = None,
    ) -> RegionTrack:
        """
        Build a track from the topology of the given dot-bracket, with one region per matching loop span.

        Args:
            dot_bracket: Dot-bracket notation.
            loop_kinds: Loop kinds to include — any subset of
                {"hairpin", "bulge", "internal", "multiloop", "external"}.
            sequence: Optional RNA sequence; defaults to ``N`` * len(dot_bracket).
            name: Optional track name.
            colors: Forwarded to [RegionTrack][]. ``None`` uses the loop-type palette.
        """
        regions = _topology_regions(dot_bracket, loop_kinds=loop_kinds, sequence=sequence)
        if colors is None:
            colors = _topology_region_colors(regions)
        return cls(regions=regions, colors=colors, name=name or "loops")


class PairAnnotationTrack:
    """
    Per-pair Leontis-Westhof annotation.

    Each entry is ``{(i, j): LeontisWesthof(edge_5p, edge_3p, orientation)}`` with
    ``i < j``. Reversed keys are rejected because the edge annotation is
    5'/3' direction-sensitive. Renders as small glyphs placed near the pair endpoints
    on [plot_planar_graph][] and [plot_circular_diagram][]. No-op on
    [plot_contact_map][] and [plot_arc_diagram][] (the natural targets for LW glyphs
    are 2D layouts).

    Args:
        annotations: Mapping from ``(i, j)`` (with ``i < j``) to a [LeontisWesthof][] entry.
        glyph_size: Matplotlib scatter size for each glyph.
        glyph_color: Edge color for the glyph; cis pairs fill with the same color, trans
            pairs render with a white fill.
        name: Track name (kept for symmetry with other tracks).

    Raises:
        ValueError: If any key is a self-pair, reversed (``i > j``), duplicated, or — from
            [validate][] — outside ``[0, length)`` or absent from the plotted structure.
        TypeError: If any value is not a [LeontisWesthof][] instance.
    """

    annotations: Mapping[Tuple[int, int], LeontisWesthof]
    glyph_size: float
    glyph_color: str
    name: str

    def __init__(
        self,
        annotations: Mapping[Tuple[int, int], LeontisWesthof],
        *,
        glyph_size: float = 22.0,
        glyph_color: str = "#222222",
        name: str = "Leontis-Westhof",
    ) -> None:
        normalized: dict[Tuple[int, int], LeontisWesthof] = {}
        for key, value in annotations.items():
            a, b = int(key[0]), int(key[1])
            if a == b:
                raise ValueError(f"PairAnnotationTrack: pair ({a}, {b}) is a self-pair.")
            if a > b:
                raise ValueError(
                    f"PairAnnotationTrack pair ({a}, {b}) must be ordered as (5' index, 3' index) with i < j."
                )
            if not isinstance(value, LeontisWesthof):
                raise TypeError(f"PairAnnotationTrack annotation for {(a, b)} must be LeontisWesthof.")
            normalized_key = (a, b)
            if normalized_key in normalized:
                raise ValueError(f"Duplicate PairAnnotationTrack annotation for pair {normalized_key}.")
            normalized[normalized_key] = value
        self.annotations = normalized
        self.glyph_size = float(glyph_size)
        self.glyph_color = glyph_color
        self.name = name

    def validate(self, length: int, pair_set: set[Tuple[int, int]] | None = None) -> None:
        for i, j in self.annotations:
            if i < 0 or j >= length:
                raise ValueError(f"PairAnnotationTrack pair ({i}, {j}) outside structure length {length}.")
            if pair_set is not None and (i, j) not in pair_set:
                raise ValueError(f"PairAnnotationTrack pair ({i}, {j}) is not present in the plotted structure.")


ColorTrack: TypeAlias = BaseValueTrack | BaseCategoryTrack
Track = Union[ColorTrack, RegionTrack, PairProbabilityTrack, SequenceDiffTrack, PairAnnotationTrack]


def _collect_color_track(
    tracks: Sequence[Track] | None,
    length: int,
) -> ColorTrack | None:
    """Return the last per-position color-bearing track in ``tracks`` (later overrides earlier)."""
    if not tracks:
        return None
    chosen: ColorTrack | None = None
    for track in tracks:
        if isinstance(track, (BaseValueTrack, BaseCategoryTrack)):
            track.validate(length)
            chosen = track
    return chosen


def _collect_color_tracks(
    tracks: Sequence[Track] | None,
    length: int,
) -> list[ColorTrack]:
    """Return all per-position color tracks, validating length."""
    if not tracks:
        return []
    result: list[ColorTrack] = []
    for track in tracks:
        if isinstance(track, (BaseValueTrack, BaseCategoryTrack)):
            track.validate(length)
            result.append(track)
    return result


def _collect_region_tracks(tracks: Sequence[Track] | None, length: int) -> list[RegionTrack]:
    if not tracks:
        return []
    result: list[RegionTrack] = []
    for track in tracks:
        if isinstance(track, RegionTrack):
            track.validate(length)
            result.append(track)
    return result


def _collect_sequence_diff_track(
    tracks: Sequence[Track] | None,
) -> SequenceDiffTrack | None:
    """Return the last [SequenceDiffTrack][] in ``tracks`` (later overrides earlier)."""
    if not tracks:
        return None
    chosen: SequenceDiffTrack | None = None
    for track in tracks:
        if isinstance(track, SequenceDiffTrack):
            chosen = track
    return chosen


def _collect_pair_annotation_track(
    tracks: Sequence[Track] | None,
    length: int,
    pairs: np.ndarray | None = None,
) -> PairAnnotationTrack | None:
    """Return the last [PairAnnotationTrack][] in ``tracks`` (later overrides earlier)."""
    if not tracks:
        return None
    chosen: PairAnnotationTrack | None = None
    for track in tracks:
        if isinstance(track, PairAnnotationTrack):
            chosen = track
    if chosen is None:
        return None
    pair_set: set[Tuple[int, int]] | None = None
    if pairs is not None:
        pair_set = {(min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs.tolist()}
    chosen.validate(length, pair_set=pair_set)
    return chosen


def _sequence_diff_mask(track: SequenceDiffTrack | None, sequence: str | None, length: int) -> np.ndarray:
    """Return a length-L boolean mask: True where sequence differs from track.reference_sequence."""
    if track is None:
        return np.zeros(length, dtype=bool)
    return track.diff_mask(sequence, length)


def _check_probability_value(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be a finite number between 0.0 and 1.0.")


def _bpRNA_categories(dot_bracket: str, *, mode: str) -> list[str]:
    if mode not in {"loop_type", "structural_class"}:
        raise ValueError(f"Unknown mode {mode!r}; expected 'loop_type' or 'structural_class'.")
    sequence = "N" * len(dot_bracket)
    annotation = _annotate_structural_classes(sequence, dot_bracket)
    if mode == "structural_class":
        return list(annotation)
    return [_STRUCTURAL_CLASS_TO_LOOP_TYPE.get(c, c) for c in annotation]


def _topology_regions(
    dot_bracket: str,
    *,
    loop_kinds: Sequence[str],
    sequence: str | None = None,
) -> list[Tuple[int, int, str]]:
    from multimolecule.utils.rna.secondary_structure import RnaSecondaryStructureTopology

    requested = {k.lower() for k in loop_kinds}
    unknown = requested - _TOPOLOGY_REGION_KINDS
    if unknown:
        choices = ", ".join(sorted(_TOPOLOGY_REGION_KINDS))
        bad = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown loop_kinds: {bad}. Available loop kinds: {choices}.")
    seq = sequence if sequence is not None else "N" * len(dot_bracket)
    topology = RnaSecondaryStructureTopology(sequence=seq, secondary_structure=dot_bracket)
    out: list[Tuple[int, int, str]] = []
    for loop in topology.loops():
        kind_name = loop.kind.name.lower()
        if kind_name not in requested:
            continue
        label = kind_name.capitalize()
        for span in loop.spans:
            if span.stop < span.start:
                continue
            out.append((int(span.start), int(span.stop), label))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def _topology_region_colors(regions: Sequence[Tuple[int, int, str]]) -> list[str]:
    from ..palettes import LOOP_TYPE_PALETTE

    return [LOOP_TYPE_PALETTE.get(label.lower(), "#999999") for *_, label in regions]


def _annotate_structural_classes(sequence: str, dot_bracket: str) -> str:
    from multimolecule.utils.rna.secondary_structure import RnaSecondaryStructureTopology, annotate_structure

    topology = RnaSecondaryStructureTopology(sequence=sequence, secondary_structure=dot_bracket)
    return annotate_structure(topology)
