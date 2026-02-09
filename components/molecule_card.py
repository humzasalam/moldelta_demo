"""Molecule detail card component."""

import streamlit as st

from utils.theme import NORD
from utils.mol_render import highlight_diff_image, smiles_to_image, smiles_to_png, highlight_diff_png


def render_molecule_card(molecule, parent):
    """Render a simplified molecule card with image + property deltas side-by-side.

    Args:
        molecule: Dict with the child molecule data.
        parent: Dict with the parent molecule data.
    """
    parent_smiles = parent["smiles"]
    child_smiles = molecule["smiles"]

    # Two-column layout: image on left, property deltas on right
    img_col, deltas_col = st.columns([1, 2])

    with img_col:
        # Parent molecule (plain render)
        st.caption("Parent")
        parent_png = smiles_to_png(parent_smiles, size=(520, 390))
        st.image(parent_png, use_container_width=True)

        # Child molecule (with diff highlighting)
        st.caption("Child")
        child_png = highlight_diff_png(parent_smiles, child_smiles, size=(520, 390))
        st.image(child_png, use_container_width=True)

        st.markdown(
            f'<div class="smiles-display" style="font-size: 0.75rem;">{child_smiles}</div>',
            unsafe_allow_html=True,
        )

    with deltas_col:
        st.markdown(f"### {molecule['name']}")
        mod = molecule.get('modification') or molecule.get('reaction_type') or ''
        if mod:
            st.markdown(f"**{mod}**")
        st.markdown("")

        st.markdown("**Property Deltas**")

        mol_props = molecule.get("properties", {})
        deltas = molecule.get("delta_properties", {})

        lower_is_better = {
            "Hepatotoxicity probability", "hERG (nM)",
            "MolLogP_unitless", "MolWt (g/mol)", "TPSA (Ang^2)",
        }

        property_list = [
            ("binding_probability", "Binding Probability"),
            ("Hepatotoxicity probability", "Hepatotoxicity"),
            ("Caco2", "Caco-2 Permeability"),
            ("Half_Life (h)", "Half-Life (h)"),
            ("LD50 (nM)", "LD50 (nM)"),
            ("hERG (nM)", "hERG (nM)"),
            ("MolLogP_unitless", "LogP"),
            ("MolWt (g/mol)", "Mol. Weight"),
            ("TPSA (Ang^2)", "TPSA (A\u00b2)"),
        ]

        # Create grid of property delta cards (4 rows x 2 cols)
        for i in range(0, len(property_list), 2):
            cols = st.columns(2)
            for j, (prop_key, prop_label) in enumerate(property_list[i:i+2]):
                with cols[j]:
                    # Check top-level first (after DataFrame extraction), then nested properties
                    val = molecule.get(prop_key, mol_props.get(prop_key, 0))
                    # Delta from top-level or nested delta_properties
                    delta = molecule.get(f"delta_{prop_key}", deltas.get(prop_key, 0))

                    delta_color = "inverse" if prop_key in lower_is_better else "normal"

                    if prop_key == "MolWt (g/mol)":
                        st.metric(prop_label, f"{val:.1f}",
                                  delta=f"{delta:+.1f}",
                                  delta_color=delta_color)
                    elif prop_key == "TPSA (Ang^2)":
                        st.metric(prop_label, f"{val:.1f}",
                                  delta=f"{delta:+.1f}",
                                  delta_color=delta_color)
                    else:
                        st.metric(prop_label, f"{val:.2f}",
                                  delta=f"{delta:+.2f}",
                                  delta_color=delta_color)


@st.dialog("Molecule Details", width="large")
def show_molecule_dialog(molecule, parent):
    """Show molecule details in a native Streamlit dialog.

    Args:
        molecule: Dict with the child molecule data.
        parent: Dict with the parent molecule data.
    """
    render_molecule_card(molecule, parent)
    if st.button("Close", use_container_width=True):
        st.session_state.selected_molecule_id = None
        st.rerun()
