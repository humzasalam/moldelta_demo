"""Molecule rendering utilities using RDKit (fast path with PNG LRU caching)."""

import io
import base64
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS, rdDepictor

# ── Dark background color (Nord bg_dark) ──
_BG_COLOR = (0.180, 0.204, 0.251, 1.0)  # #2E3440
_HIGHLIGHT_GREEN = (0.639, 0.745, 0.549)  # #A3BE8C
_HIGHLIGHT_RED = (0.749, 0.380, 0.416)    # #BF616A
_BOND_COLOR = (0.847, 0.871, 0.914)       # #D8DEE9

_DEFAULT_SIZE = (600, 450)  # ↓ smaller than before for speed

# Prefer RDKit's faster coord generator without kekulization pitfalls
rdDepictor.SetPreferCoordGen(True)


def _error_image_bytes(size=_DEFAULT_SIZE):
    """Generate a placeholder error image (PNG bytes)."""
    img = Image.new("RGBA", size, (46, 52, 64, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=8192)
def _mol_block_with_coords(smiles: str) -> Optional[str]:
    """Cache a MOL block with computed 2D coords (text, so it's cacheable)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        # Fallback without kekulization issues
        rdDepictor.Compute2DCoords(mol)
    return Chem.MolToMolBlock(mol)


def _mol_from_block(block: Optional[str]) -> Optional[Chem.Mol]:
    if not block:
        return None
    try:
        return Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
    except Exception:
        return None


def _prepare_drawer(width: int, height: int) -> Draw.rdMolDraw2D.MolDraw2DCairo:
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.setBackgroundColour(_BG_COLOR)
    # Keep these modest for speed
    opts.bondLineWidth = 2.2
    opts.multipleBondOffset = 0.14
    opts.scalingFactor = 32
    opts.fixedBondLength = 32
    opts.minFontSize = 14
    opts.maxFontSize = 22
    opts.additionalAtomLabelPadding = 0.10

    # Atom colors suited for dark background
    opts.updateAtomPalette({
        0: _BOND_COLOR,
        1: _BOND_COLOR,
        6: _BOND_COLOR,                  # C
        7: (0.533, 0.753, 0.816),        # N
        8: _HIGHLIGHT_RED,               # O
        9: _HIGHLIGHT_GREEN,             # F
        16: (0.816, 0.529, 0.439),       # S
        17: (0.922, 0.796, 0.545),       # Cl
        35: (0.706, 0.557, 0.678),       # Br
    })
    return drawer


@lru_cache(maxsize=8192)
def _png_for(smiles: str,
             size: Tuple[int, int],
             highlight_atoms: Tuple[int, ...] = (),
             color: Tuple[float, float, float] = _HIGHLIGHT_GREEN) -> bytes:
    """Render a molecule to PNG bytes. Cached by smiles/size/highlights."""
    block = _mol_block_with_coords(smiles)
    mol = _mol_from_block(block)
    if mol is None:
        return _error_image_bytes(size)

    w, h = size
    drawer = _prepare_drawer(w, h)

    # Highlight maps (atoms only; bonds computed automatically by RDKit)
    hatoms = list(highlight_atoms) if highlight_atoms else []
    h_atom_colors = {idx: color for idx in hatoms}
    h_radii = {idx: 0.42 for idx in hatoms}

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=hatoms,
        highlightAtomColors=h_atom_colors,
        highlightAtomRadii=h_radii,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def smiles_to_png(smiles: str, size: Tuple[int, int] = _DEFAULT_SIZE) -> bytes:
    """Convert SMILES to PNG bytes (fast, cached)."""
    return _png_for(smiles, size, ())


def smiles_to_image(smiles: str, size: Tuple[int, int] = _DEFAULT_SIZE) -> Image.Image:
    """Convert SMILES to PIL Image (uses cached PNG)."""
    return Image.open(io.BytesIO(smiles_to_png(smiles, size)))


def smiles_to_base64(smiles: str, size: Tuple[int, int] = _DEFAULT_SIZE) -> str:
    """Convert SMILES to a base64 data URI (uses cached PNG)."""
    b = smiles_to_png(smiles, size)
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"


@lru_cache(maxsize=8192)
def _diff_atom_indices(parent_smiles: str, child_smiles: str) -> Tuple[int, ...]:
    """Compute child atom indices that are NOT in parent; cached.

    Strategy:
      1) Try a direct substructure match (fast).
      2) If no match, run MCS with a *tiny* timeout (0.2s).
      3) If still nothing usable, return empty (no highlight).
    """
    p_block = _mol_block_with_coords(parent_smiles)
    c_block = _mol_block_with_coords(child_smiles)
    parent = _mol_from_block(p_block)
    child = _mol_from_block(c_block)
    if child is None:
        return tuple()
    if parent is None:
        # no parent → highlight nothing (just render child)
        return tuple()

    match = child.GetSubstructMatch(parent)
    if match:
        all_atoms = set(range(child.GetNumAtoms()))
        return tuple(sorted(all_atoms - set(match)))

    # Tiny-timeout MCS; keep relaxed settings for speed
    try:
        mcs = rdFMCS.FindMCS(
            [parent, child],
            timeout=1,                  # was 0.2
            threshold=0.8,
            ringMatchesRingOnly=True,   # matches your earlier behavior
            completeRingsOnly=True,     # ↑ this helps stable highlights on ring edits
            matchValences=False,
        )
        if mcs.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol is not None:
                m = child.GetSubstructMatch(mcs_mol)
                if m:
                    all_atoms = set(range(child.GetNumAtoms()))
                    return tuple(sorted(all_atoms - set(m)))
    except Exception:
        pass

    return tuple()


def highlight_diff_png(parent_smiles: str, child_smiles: str,
                       size: Tuple[int, int] = _DEFAULT_SIZE) -> bytes:
    """Render child with diff atoms highlighted; cached via _png_for."""
    diff_atoms = _diff_atom_indices(parent_smiles, child_smiles)
    return _png_for(child_smiles, size, diff_atoms, _HIGHLIGHT_GREEN)


def highlight_diff_image(parent_smiles: str, child_smiles: str,
                         size: Tuple[int, int] = _DEFAULT_SIZE) -> Image.Image:
    return Image.open(io.BytesIO(highlight_diff_png(parent_smiles, child_smiles, size)))
