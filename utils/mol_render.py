"""Molecule rendering utilities using RDKit."""

import io
import base64

import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS


# ── Dark background color (Nord bg_dark) ──
_BG_COLOR = (0.180, 0.204, 0.251, 1.0)  # #2E3440
_HIGHLIGHT_GREEN = (0.639, 0.745, 0.549)  # #A3BE8C (aurora_green)
_HIGHLIGHT_RED = (0.749, 0.380, 0.416)    # #BF616A (aurora_red)
_BOND_COLOR = (0.847, 0.871, 0.914)       # #D8DEE9 (snow_0)


def _mol_to_png_bytes(mol, size=(400, 300), highlight_atoms=None,
                       highlight_color=None):
    """Render an RDKit mol to PNG bytes with dark background."""
    if mol is None:
        return _error_image_bytes(size)

    AllChem.Compute2DCoords(mol)

    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.setBackgroundColour(_BG_COLOR)
     
    # Significantly improve rendering quality
    opts.bondLineWidth = 3.0  # Increased from 2.5
    opts.multipleBondOffset = 0.18  # Increased from 0.15
    opts.scalingFactor = 40  # Increased from 30
    opts.fixedBondLength = 35  # Increased from 30
    opts.minFontSize = 18  # Larger font for atom labels
    opts.maxFontSize = 28
    
    # Anti-aliasing and quality settings
    opts.additionalAtomLabelPadding = 0.15

    # Make atoms and bonds visible on dark background
    opts.updateAtomPalette({
        0: (0.847, 0.871, 0.914),  # default (snow)
        6: (0.847, 0.871, 0.914),  # C
        7: (0.533, 0.753, 0.816),  # N (frost_1)
        8: (0.749, 0.380, 0.416),  # O (aurora_red)
        9: (0.639, 0.745, 0.549),  # F (aurora_green)
        16: (0.816, 0.529, 0.439), # S (aurora_orange)
        17: (0.922, 0.796, 0.545), # Cl (aurora_yellow)
        35: (0.706, 0.557, 0.678), # Br (aurora_purple)
    })

    highlight_atom_colors = {}
    highlight_bond_colors = {}
    highlight_radii = {}

    if highlight_atoms:
        color = highlight_color or _HIGHLIGHT_GREEN
        for atom_idx in highlight_atoms:
            highlight_atom_colors[atom_idx] = color
            highlight_radii[atom_idx] = 0.45  # Slightly larger highlight

    Draw.rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms or [],
        highlightAtomColors=highlight_atom_colors if highlight_atoms else {},
        highlightBonds=[],
        highlightBondColors={},
        highlightAtomRadii=highlight_radii if highlight_atoms else {},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _error_image_bytes(size=(400, 300)):
    """Generate a placeholder error image."""
    img = Image.new("RGBA", size, (46, 52, 64, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data
def smiles_to_png(smiles, size=(1200, 900)):
    """Convert SMILES to PNG bytes (cached). Returns bytes."""
    mol = Chem.MolFromSmiles(smiles)
    return _mol_to_png_bytes(mol, size)


@st.cache_data
def smiles_to_image(smiles, size=(1200, 900)):
    """Convert SMILES to PIL Image (cached)."""
    png_bytes = smiles_to_png(smiles, size)
    return Image.open(io.BytesIO(png_bytes))


@st.cache_data
def smiles_to_base64(smiles, size=(1200, 900)):
    """Convert SMILES to a base64 data URI string."""
    png_bytes = smiles_to_png(smiles, size)
    b64 = base64.b64encode(png_bytes).decode()
    return f"data:image/png;base64,{b64}"


@st.cache_data
def highlight_diff_png(parent_smiles, child_smiles, size=(1200, 900)):
    """Render the child molecule with diff atoms highlighted. Returns PNG bytes."""
    parent = Chem.MolFromSmiles(parent_smiles)
    child = Chem.MolFromSmiles(child_smiles)

    if child is None:
        return _error_image_bytes(size)
    if parent is None:
        return _mol_to_png_bytes(child, size)

    # Try direct substructure match first
    match = child.GetSubstructMatch(parent)
    if match:
        all_atoms = set(range(child.GetNumAtoms()))
        diff_atoms = list(all_atoms - set(match))
        return _mol_to_png_bytes(child, size, highlight_atoms=diff_atoms,
                                  highlight_color=_HIGHLIGHT_GREEN)

    # Fallback: Maximum Common Substructure
    try:
        mcs_result = rdFMCS.FindMCS(
            [parent, child],
            threshold=0.8,
            timeout=2,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        if mcs_result.smartsString:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            if mcs_mol:
                mcs_match = child.GetSubstructMatch(mcs_mol)
                if mcs_match:
                    all_atoms = set(range(child.GetNumAtoms()))
                    diff_atoms = list(all_atoms - set(mcs_match))
                    return _mol_to_png_bytes(child, size,
                                              highlight_atoms=diff_atoms,
                                              highlight_color=_HIGHLIGHT_GREEN)
    except Exception:
        pass

    # Final fallback: render without highlighting
    return _mol_to_png_bytes(child, size)


def highlight_diff_image(parent_smiles, child_smiles, size=(1200, 900)):
    """Render the child molecule with diff atoms highlighted. Returns PIL Image."""
    png_bytes = highlight_diff_png(parent_smiles, child_smiles, size)
    return Image.open(io.BytesIO(png_bytes))
