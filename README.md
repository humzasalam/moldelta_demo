# MolDelta - AI-Powered Lead Optimization

An interactive molecular optimization platform for drug discovery, featuring real-time visualization and multi-property optimization.

## Features

- 🧬 **Interactive Molecule Visualization** - View parent and child molecules with structural highlighting
- 📊 **Multi-Property Optimization** - Optimize across 9 ADME properties + binding probability
- 🎯 **Smart Ranking** - Automatic optimization scoring based on property improvements
- 📈 **Dynamic Scatter Plots** - Explore molecular property space interactively
- 🔍 **Detailed Molecule Cards** - View comprehensive property deltas and predictions

## Live Demo

🚀 **[Try it live on Streamlit Cloud](#)** *(coming soon)*

## Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/moldelta-demo.git
cd moldelta-demo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Technologies

- **Streamlit** - Interactive web framework
- **RDKit** - Molecular visualization and cheminformatics
- **Plotly** - Interactive data visualization
- **Pandas** - Data manipulation
- **Nord Theme** - Modern UI design

## Properties Optimized

1. Binding Probability (higher is better)
2. Hepatotoxicity (lower is better)
3. Caco-2 Permeability (higher is better)
4. Half-Life (higher is better)
5. LD50 Toxicity (higher is better)
6. hERG Cardiotoxicity (lower is better)
7. LogP Lipophilicity (lower is better)
8. Molecular Weight (lower is better)
9. TPSA Polar Surface Area (lower is better)

## Project Structure

```
moldelta-demo/
├── app.py                 # Main Streamlit application
├── components/            # UI components
│   ├── viz_panel.py      # Visualization and controls
│   └── molecule_card.py  # Molecule detail cards
├── utils/                 # Utility functions
│   ├── plotting.py       # Plotly chart builders
│   ├── mol_render.py     # RDKit molecule rendering
│   └── theme.py          # Nord theme configuration
└── data/                  # Molecular data
    ├── parent_1.json     # Pimavanserin parent
    ├── parent_2.json     # Brexpiprazole parent
    ├── children_1.json   # Pimavanserin children (56)
    └── children_2.json   # Brexpiprazole children (50)
```

## Built With

Made with ❤️ for drug discovery
