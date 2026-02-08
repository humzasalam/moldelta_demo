"""
One-time script to generate 50 validated Celecoxib analog children with
correlated fake scores. Run once to produce data/children.json.

Usage:
    cd moldelta-demo
    python scripts/generate_children.py
"""

import json
import random
import math
from rdkit import Chem
from rdkit.Chem import Descriptors

random.seed(42)

PARENT_SMILES = "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F"
PARENT_PROPS = {
    "binding_probability": 0.72,
    "binding_affinity_log10": -6.8,
    "hERG_risk": 0.45,
    "hepatotoxicity": 0.30,
    "solubility": 0.55,
    "lipophilicity": 3.2,
    "general_toxicity_score": 0.38,
}

# ── Celecoxib analogs: hand-curated SMILES modifications ──
# Each tuple: (smiles, reaction_type, description_hint)
CANDIDATES = [
    # Category 1: Para-substituent variations on the tolyl ring (replace CH3)
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "halogenation", "Para-chloro on tolyl ring"),
    ("FC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "halogenation", "Para-fluoro on tolyl ring"),
    ("BrC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "halogenation", "Para-bromo on tolyl ring"),
    ("COC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "O-alkylation", "Para-methoxy on tolyl ring"),
    ("OC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "hydroxylation", "Para-hydroxy on tolyl ring"),
    ("NC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "amination", "Para-amino on tolyl ring"),
    ("N#CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "cyanation", "Para-cyano on tolyl ring"),
    ("CCC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "alkylation", "Para-ethyl on tolyl ring"),
    ("CC(C)C1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "alkylation", "Para-isopropyl on tolyl ring"),
    ("FC(F)(F)C1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "fluorination", "Para-trifluoromethyl on tolyl ring"),

    # Category 2: Sulfonamide modifications
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "N-methylation", "N-methyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N(C)C)(=O)=O)C(F)(F)F", "N-alkylation", "N,N-dimethyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NCC)(=O)=O)C(F)(F)F", "N-alkylation", "N-ethyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(O)(=O)=O)C(F)(F)F", "hydrolysis", "Sulfonic acid (sulfonamide to SO3H)"),

    # Category 3: Pyrazole C3 substituent variations (replace CF3)
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)F", "defluorination", "Difluoromethyl at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C", "defluorination", "Methyl at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)CC", "alkylation", "Ethyl at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C#N", "cyanation", "Cyano at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(=O)O", "carboxylation", "Carboxylic acid at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(=O)N", "amidation", "Carboxamide at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)Cl", "chlorination", "Chloro at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(Cl)(Cl)Cl", "halogenation", "Trichloromethyl at C3"),

    # Category 4: Ring modifications on the sulfonamide phenyl
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1F)S(N)(=O)=O)C(F)(F)F", "fluorination", "3-fluoro on sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C2=CC(F)=CC=C2S(N)(=O)=O)C(F)(F)F", "fluorination", "2-fluoro on sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C(C)=C1)S(N)(=O)=O)C(F)(F)F", "methylation", "3-methyl on sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C2=CC(Cl)=CC=C2S(N)(=O)=O)C(F)(F)F", "chlorination", "2-chloro on sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1Cl)S(N)(=O)=O)C(F)(F)F", "chlorination", "3-chloro on sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1OC)S(N)(=O)=O)C(F)(F)F", "O-alkylation", "3-methoxy on sulfonamide phenyl"),

    # Category 5: Multi-site / hybrid modifications
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-Cl tolyl + N-methyl sulfonamide"),
    ("FC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-F tolyl + N-methyl sulfonamide"),
    ("COC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-OMe tolyl + N-methyl sulfonamide"),
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)F", "multi-site", "Para-Cl tolyl + CHF2 at C3"),
    ("FC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)F", "multi-site", "Para-F tolyl + CHF2 at C3"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C(C)=C1)S(N)(=O)=O)C(F)F", "multi-site", "3-Me sulfonamide phenyl + CHF2 at C3"),
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1F)S(N)(=O)=O)C(F)(F)F", "multi-site", "Para-Cl tolyl + 3-F sulfonamide phenyl"),
    ("FC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1F)S(N)(=O)=O)C(F)(F)F", "multi-site", "Para-F tolyl + 3-F sulfonamide phenyl"),
    ("CC1=CC(C)=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F", "methylation", "2,4-dimethyl on tolyl ring"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)Cl", "halogenation", "CClF2 at C3"),
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C", "multi-site", "Para-Cl tolyl + methyl at C3"),
    ("FC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C", "multi-site", "Para-F tolyl + methyl at C3"),
    ("COC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)F", "multi-site", "Para-OMe tolyl + CHF2 at C3"),
    ("NC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-NH2 tolyl + N-methyl sulfonamide"),
    ("OC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-OH tolyl + N-methyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1F)S(NC)(=O)=O)C(F)(F)F", "multi-site", "3-F sulfonamide phenyl + N-methyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1Cl)S(NC)(=O)=O)C(F)(F)F", "multi-site", "3-Cl sulfonamide phenyl + N-methyl sulfonamide"),
    ("CCC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-Et tolyl + N-methyl sulfonamide"),
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C2=CC(F)=CC=C2S(N)(=O)=O)C(F)(F)F", "multi-site", "Para-Cl tolyl + 2-F sulfonamide phenyl"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1OC)S(NC)(=O)=O)C(F)(F)F", "multi-site", "3-OMe sulfonamide phenyl + N-methyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NCC)(=O)=O)C(F)F", "multi-site", "N-ethyl sulfonamide + CHF2 at C3"),
    ("ClC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1Cl)S(N)(=O)=O)C(F)(F)F", "multi-site", "Para-Cl tolyl + 3-Cl sulfonamide phenyl"),
    ("FC(F)(F)C1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(NC)(=O)=O)C(F)(F)F", "multi-site", "Para-CF3 tolyl + N-methyl sulfonamide"),
    ("CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)CBr", "halogenation", "Bromomethyl at C3"),
    ("COC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1F)S(N)(=O)=O)C(F)(F)F", "multi-site", "Para-OMe tolyl + 3-F sulfonamide phenyl"),
]

# Literature sources for "known binders"
LIT_SOURCES = [
    "J. Med. Chem. 2023, 66, 4521-4538",
    "Bioorg. Med. Chem. 2024, 32, 115742",
    "Eur. J. Med. Chem. 2023, 256, 115463",
    "ACS Med. Chem. Lett. 2024, 15, 312-318",
    "J. Med. Chem. 2024, 67, 1123-1140",
]

# Mark these indices as known binders (0-indexed)
KNOWN_BINDER_INDICES = {0, 2, 10, 14, 22, 28, 35}


def compute_correlated_scores(smiles, idx, parent_props):
    """Generate fake but chemically correlated scores."""
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = Descriptors.NumHAcceptors(mol)
    hbd = Descriptors.NumHDonors(mol)
    n_rotatable = Descriptors.NumRotatableBonds(mol)
    n_heavy = mol.GetNumHeavyAtoms()

    # Parent reference MW and LogP
    parent_mol = Chem.MolFromSmiles(PARENT_SMILES)
    parent_mw = Descriptors.ExactMolWt(parent_mol)
    parent_logp = Descriptors.MolLogP(parent_mol)

    # Binding probability: loosely correlated with logP (sweet spot around 3-4)
    # and molecular size (bigger can be better up to a point)
    logp_factor = 1.0 - abs(logp - 3.5) * 0.08
    size_factor = min(1.0, n_heavy / 25.0)
    base_binding = 0.72 + random.gauss(0.06, 0.12) * logp_factor * size_factor
    binding_probability = max(0.35, min(0.95, base_binding))

    # Binding affinity: correlated with binding probability
    binding_affinity = -6.8 + (binding_probability - 0.72) * 8.0 + random.gauss(0, 0.3)
    binding_affinity = max(-10.5, min(-5.5, binding_affinity))

    # hERG risk: higher for more lipophilic, basic amines increase risk
    herg_base = 0.45 + (logp - parent_logp) * 0.06
    has_basic_N = "N" in smiles and "S(N" not in smiles.replace("S(NC", "")
    if has_basic_N:
        herg_base += 0.08
    herg_risk = max(0.05, min(0.85, herg_base + random.gauss(0, 0.08)))

    # Hepatotoxicity: somewhat random but correlated with MW
    hepato_base = 0.30 + (mw - parent_mw) * 0.001 + random.gauss(0, 0.08)
    hepatotoxicity = max(0.05, min(0.75, hepato_base))

    # Solubility: inversely correlated with logP
    sol_base = 0.55 - (logp - parent_logp) * 0.12 + random.gauss(0, 0.06)
    solubility = max(0.10, min(0.90, sol_base))

    # General toxicity: composite of hERG and hepato
    general_tox = 0.4 * herg_risk + 0.4 * hepatotoxicity + 0.2 * random.gauss(0.35, 0.05)
    general_toxicity_score = max(0.05, min(0.80, general_tox))

    # Known binders get a boost
    if idx in KNOWN_BINDER_INDICES:
        binding_probability = min(0.95, binding_probability + 0.10)
        binding_affinity = min(-5.5, binding_affinity - 0.8)

    props = {
        "binding_probability": round(binding_probability, 2),
        "binding_affinity_log10": round(binding_affinity, 1),
        "hERG_risk": round(herg_risk, 2),
        "hepatotoxicity": round(hepatotoxicity, 2),
        "solubility": round(solubility, 2),
        "lipophilicity": round(logp, 2),
        "general_toxicity_score": round(general_toxicity_score, 2),
    }

    delta = {}
    for key in parent_props:
        if key in props:
            delta[key] = round(props[key] - parent_props[key], 2)

    return props, delta


def main():
    parent_mol = Chem.MolFromSmiles(PARENT_SMILES)
    assert parent_mol is not None, "Parent SMILES invalid!"

    children = []
    valid_count = 0

    for idx, (smiles, reaction_type, description) in enumerate(CANDIDATES):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  WARNING: Invalid SMILES at index {idx}: {smiles}")
            continue

        # Canonicalize
        canonical = Chem.MolToSmiles(mol)

        props, delta = compute_correlated_scores(canonical, idx, PARENT_PROPS)

        child_id = f"child_{valid_count + 1:03d}"
        name = f"MolDelta-{valid_count + 1:03d}"

        is_known = idx in KNOWN_BINDER_INDICES
        lit_source = random.choice(LIT_SOURCES) if is_known else None

        child = {
            "id": child_id,
            "smiles": canonical,
            "name": name,
            "generation": 1,
            "reaction_type": reaction_type,
            "modification": description,
            "binding_probability": props["binding_probability"],
            "binding_affinity_log10": props["binding_affinity_log10"],
            "delta_properties": delta,
            "is_known_binder": is_known,
            "literature_source": lit_source,
            "properties": props,
        }

        children.append(child)
        valid_count += 1

    print(f"\nGenerated {valid_count} valid children out of {len(CANDIDATES)} candidates")

    out_path = "data/children.json"
    with open(out_path, "w") as f:
        json.dump(children, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
