Uranium Dataset Package
This package provides tools for loading, exploring and summarising a curated subset of the U.S. Geological Survey’s Mineral Resources Data System (MRDS) focusing on uranium occurrences. It mirrors the modular design of the Colorado River Streamlit app described in the colorado_river directory of the original data‑mine project
GitHub
. Instead of a web UI, this package emphasises reproducible scripts and notebooks that work in both Python and R.
Goals
Reproducibility. All data paths and API keys are stored in config.json, so you can rebuild the dataset or point to a different CSV without changing any code. The included uranium_dataset_pipeline.py demonstrates how to download the full MRDS archive, filter it for uranium deposits, and optionally clean names using ChatGPT via an API key.
Separation of concerns. Functionality is broken into small modules:
data_loader.py reads config.json and returns a pandas DataFrame for analysis.
eda.py contains simple exploratory functions to count records by state, deposit type, and to summarise numeric columns.
meta.py exposes a lightweight command‑line interface that prints basic dataset metadata (number of records, unique states, etc.).
uranium_eda.ipynb is an R notebook that loads the dataset, summarises key variables and produces a few plots using dplyr and ggplot2.
Transparency. Each module is heavily commented to aid students learning how to work with data. Docstrings explain what each function does, its parameters and return values, and notes on performance and design choices.
Directory structure
uranium_dataset/
├── data/                # the filtered uranium dataset (`uranium_mines.csv`)
├── config.json          # configuration file with dataset path and API keys
├── data_loader.py       # functions to load configuration and dataset
├── eda.py               # exploratory data analysis helpers
├── meta.py              # command‑line interface for summary statistics
├── uranium_eda.ipynb    # R notebook demonstrating basic analysis
└── README.md            # this file
Usage
Loading data in Python
from uranium_dataset import load_dataset, count_by_state

# Load full dataset using default configuration
df = load_dataset()

# Get a table of record counts per state
counts = count_by_state(df)
print(counts.head())
Command‑line overview
To see a brief overview of the dataset or counts by specific fields, run:
python -m uranium_dataset.meta            # prints overview
python -m uranium_dataset.meta state      # counts by state
python -m uranium_dataset.meta type       # counts by deposit type
R notebook exploration
Open uranium_eda.ipynb in JupyterLab or VS Code with the R kernel installed. The notebook reads config.json, loads the CSV using read.csv, groups records by state and deposit type, and draws bar charts. Each cell is commented to explain its purpose.
Extending the package
You can extend the package by adding new functions to eda.py (e.g. clustering by location or temporal analysis) or by incorporating additional datasets into the data directory. To rebuild the CSV from scratch, run uranium_dataset_pipeline.py with your own API key in config.json.
Acknowledgements
The underlying uranium occurrence data comes from the U.S. Geological Survey’s MRDS, which compiles reports on metallic and non‑metallic mineral resources worldwide
datacommons.psu.edu
. Many of the column definitions used in DATA.md are adapted from USGS metadata for the MRDS dataset
portal1-geo.sabu.mtu.edu
.