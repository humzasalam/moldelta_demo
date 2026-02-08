# MolDelta Demo - Setup & Run Instructions

## Quick Start (Recommended)

Run the automated setup script:

```bash
cd "/Users/humzasalam/Desktop/MolDelta YC Demo/moldelta-demo"
./start.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Launch the Streamlit app at http://localhost:8501

---

## Manual Setup (Alternative)

If you prefer manual setup or the script doesn't work:

### Step 1: Create Virtual Environment

```bash
cd "/Users/humzasalam/Desktop/MolDelta YC Demo/moldelta-demo"
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install streamlit plotly rdkit pandas Pillow
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open automatically at http://localhost:8501

---

## Troubleshooting

### "Module not found" errors
Make sure your virtual environment is activated:
```bash
source venv/bin/activate
```

### Port already in use
If port 8501 is busy, use a different port:
```bash
streamlit run app.py --server.port 8502
```

### RDKit installation issues
RDKit can be tricky. If pip fails, try:
```bash
conda install -c conda-forge rdkit
```

Or use a pre-built wheel from:
https://github.com/rdkit/rdkit/releases

---

## What Changed in This Version

✅ Fixed viewport layout with internal chat scrolling
✅ Removed parent molecule preview, subtitle, and Enamine button
✅ Added "Ready for Analysis" placeholder state
✅ Loading animations now appear in the analysis box
✅ Command-only chat (no chatty error messages)
✅ Natural language commands auto-set axes
✅ Simplified molecule detail view with side-by-side layout
✅ Enhanced intent parser for robust command understanding
