## Colorado & Gunnison River Flow — Streamlit App

[![Colorado River CI](https://github.com/wmacevoy/data-mine-wmacevoy/actions/workflows/colorado_river_ci.yaml/badge.svg)](https://github.com/wmacevoy/data-mine-wmacevoy/actions/workflows/colorado_river_ci.yaml)

A small, teaching-oriented Streamlit app that fetches and visualizes USGS river data (Colorado & Gunnison Rivers near Grand Junction). It shows instantaneous values (IV), daily values (DV), basic feature engineering, sampling gap diagnostics, and simple anomaly scores.

### Data sources
- **USGS Water Services API**: `nwis/iv` (instantaneous values) and `nwis/dv` (daily values)
  - 00060 = discharge (cubic feet per second)
  - 00065 = gage height (feet)

### Repository layout
- `app.py`: Streamlit UI and charts
- `usgs.py`: USGS API client, caching to `./data/` as Parquet
- `eda.py`: Helpers for timezone handling, resampling, feature engineering, and anomalies
- `requirements.txt`: Python deps
- `setup.sh`: Creates/updates the local env at `./.venv`, resets caches, installs deps
- `context.sh`: Shared helpers (e.g., `conda_exe`, `conda_venv`, `python_exe`, Streamlit env exports)
- `python.sh`: Runs `python` inside the local env
- `run.sh`: Runs Streamlit via `python.sh -m streamlit`
- `px.py`: Simple Parquet data explorer (CLI)
- `px.sh`: Runs `px.py` inside the env
- `meta.py`: Lists configured USGS sources from `config.json`
- `meta.sh`: Runs `meta.py` inside the env
- `.streamlit/config.toml`: Auto-created by `setup.sh` based on `config.json`
- `config.json`: App configuration (USGS sources, debug flag, Streamlit defaults)
- `data/`: Cached Parquet files per site and window

### Quick start
Prerequisites:
- macOS/Linux with either `mamba` or `conda` on PATH, or Windows WSL 
Steps:
1) Make scripts executable once:
   - `chmod +x setup.sh run.sh px.sh meta.sh python.sh`
2) Setup/reset the environment
   - `./setup.sh --help` shows utility flags:
     - `--debug`: implies `--reset`, clears `./data/` and `./debug/`, sets `config.json.debug=true`
     - `--reset`: clears `./data/` and `./debug/`; reinstalls deps if needed
     - `--restart`: deletes and recreates `./.venv` (fresh env)
3) Run `./run.sh` and open the URL printed by Streamlit (default `http://localhost:8501`).
   - Extra Streamlit args can be passed through, e.g. `./run.sh --server.port 8502`


Notes:
- The scripts use a prefix-based env at `./.venv` (no `conda init` or manual activation needed).
- If you prefer, you can still `conda activate /abs/path/to/.venv` and run `python -m streamlit run app.py` by hand.

### How it works (short version)
- IV data are fetched in UTC, converted to America/Denver for display, then made
  timezone-naive before display to ensure Arrow/Streamlit compatibility.
- DV data are date-indexed (no timezone).
- Data are cached to `./data/` as Parquet for speed. If you change windows (e.g., IV days, DV years), new cache files are written.

Notes on serialization (Arrow):
- Streamlit converts DataFrames to Arrow tables. Mixed-type object columns (e.g.,
  floats + timestamps) can cause errors. The app sanitizes DataFrames before
  display so datetime columns are timezone-naive and numeric columns remain
  numeric.
- The IV gap summary is rendered as JSON to avoid mixed-type table issues.

### JSON debug views
For easier troubleshooting of timestamp and schema, the app includes expanders showing small slices of the raw JSON from USGS:
- “Raw IV JSON (sample)” under the IV section
- “Raw DV JSON (sample)” under the DV section

### Troubleshooting
- **“bad interpreter” or temp-dir shebang errors when starting Streamlit**
  - We invoke Streamlit via `python -m streamlit` inside the env. Always use `./run.sh`. If you still see issues, try `./setup.sh --restart`

- **ArrowInvalid / tz-aware timestamp errors in Streamlit tables**
  - This app normalizes datetimes before display. If you still see Arrow errors:
    - Ensure deps match `requirements.txt` (notably `pandas>=2.2` and `pyarrow`).
    - Delete stale cache files in `./data/` and refresh.
    - Use the Debug snapshots (below) and open the saved `*_dtypes.txt` files to
      spot columns with unexpected types.

- **Port already in use**
  - Stop the other process, or run on a different port: `python -m streamlit run app.py --server.port 8502`


### USGS site catalog
Defined in `config.json` under `usgs_sources`. You can add more site codes as needed, e.g.:
```
"Colorado River near Cameo (09095500)": "09095500",
"Gunnison River near Grand Junction (09152500)": "09152500",
"Colorado River at CO–UT State Line (09163500)": "09163500",
```

The app and CLI will load `SITE_CATALOG` from `config.json` with a safe fallback baked into `usgs.py`.

### Meta utilities
- To list currently available USGS sources:
  - `./meta.sh`

### Development
- Python formatting and style: keep code readable and explicit
- Cached data: safe to delete `./data/*.parquet` when schemas change

Debugging aid (optional):
- In the app sidebar, enable “Save debug snapshots” to write JSON and CSV
  snippets into `./debug/`. These include:
  - `iv_json_*`, `dv_json_*`: raw USGS JSON slices
  - `iv_raw_*`, `iv_local_*`, `iv_display_*`: IV data at key steps
  - `dv_*`: DV data
  - `feats_*`, `anoms_*`, `anoms_display_*`: engineered features/anomalies
  Each snapshot writes a `*_head.csv` (first rows) and `*_dtypes.txt` (column
  dtypes) to help diagnose schema issues.

### License
### Parquet Explorer (CLI)
Use `px.py` to browse cached Parquet files from the command line.

Examples (use the `./px.sh` wrapper so the env is used automatically):
- List columns and dtypes: `./px.sh data/*.parquet --columns`
- Show head and basic info: `./px.sh data/09163500_dv_5y.parquet --info --head 10`
- Select columns and filter rows: `./px.sh data/09095500_iv_7d.parquet --select time,discharge_cfs --where "discharge_cfs > 1000" --head 10`
- Time-window filter: `./px.sh data/09095500_iv_7d.parquet --time-col time --start 2025-08-10 --end 2025-08-12 --head 20`

### License
Licensed under the MIT License. See `LICENSE`.


