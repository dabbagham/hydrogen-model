# Hydrogen Cost Model

A hydrogen production and transport cost model with both a legacy PyQt desktop GUI and a Streamlit web UI. The tool
combines renewable generation cost calculations (solar and wind) with transport and conversion pathways to estimate
minimum delivered cost of hydrogen and identify the cheapest source/location for a given demand and destination.

## Summary

The application estimates the **lowest delivered cost of hydrogen** for a chosen demand location by:

- Evaluating many candidate renewable sites worldwide.
- Computing the cost to generate hydrogen at each site (solar vs. wind + electrolyzer system).
- Computing the cost to move hydrogen to the demand location using multiple carrier media and transport modes.
- Selecting the minimum-cost pathway and reporting the cheapest source, medium, and total cost per kg H₂.
- Optionally running Monte Carlo simulations to quantify uncertainty in cost drivers.

## What this tool does

At a high level, the model:

1. Loads renewable resource data for candidate generation locations.
2. Calculates generation costs for hydrogen based on electrolyzer technology, cost trajectories, and site-specific
   renewable output.
3. Calculates transport costs to move hydrogen to a target demand location using shipping, trucking, and pipelines,
   with multiple carrier media (NH₃, LOHC, liquid H₂, gaseous H₂).
4. Selects the lowest-cost generation + transport pathway for the specified demand and destination.
5. Optionally runs a Monte Carlo simulation across parameter distributions to quantify cost uncertainty.

## Methodology

### End-to-end flow

1. **Input capture**: User provides demand location (lat/long), annual demand, year, electrolyzer type, and transport
   options in the UI.
2. **Generation cost computation**: For each candidate renewable site, the model sizes the required renewable capacity
   and electrolyzer plant, annualizes CapEx, and adds OpEx to derive cost per kg H₂.
3. **Transport cost computation**: For each site, the model estimates multi-leg transport costs (site → port → ship →
   port → demand location), compares possible carrier media, and includes optional pipeline legs.
4. **Selection**: The minimum total cost (generation + transport) is selected across all candidate sites and media.
5. **Output**: The tool saves and displays the cheapest pathway and supporting cost metrics.

### Generation cost model
Generation costs are computed per candidate renewable location based on:

- **Electrolyzer technology** (alkaline, SOEC, PEM) with year-based CapEx/efficiency trajectories.
- **Renewable plant sizing** to meet the annual hydrogen demand given the selected full-load hours.
- **Annualized capital costs** and operating costs for solar, wind, and electrolyzer systems.
- **Energy consumption** for hydrogen production and compression.

Key computations performed in `generation_costs.py` include:

- **Sizing**: Electricity demand is derived from hydrogen demand and electrolyzer efficiency. This is translated into
  required solar array size and number of wind turbines using site-specific resource data.
- **Annualization**: CapEx is annualized using the discount rate and lifetime assumptions for each asset.
- **Cost comparison**: The model computes the annualized cost for solar and wind and selects the cheaper source per
  site.
- **Electrolyzer cost**: CapEx + OpEx for the electrolyzer system is added to yield a total generation cost per kg H₂.

These computations return yearly cost and cost per kg of H₂ for each candidate site. The model uses deterministic
parameter defaults that can be overridden in the Streamlit UI.

### Transport cost model
Transport costs combine the following steps:

1. **Distance estimation** between generation sites, ports, and the demand location (driving distances use OSRM).
2. **Shipping network routing** for maritime segments (using a pre-built shipping network graph).
3. **Carrier pathways** evaluated for NH₃, LOHC, liquid H₂, and gaseous H₂ with optional pipeline segments.
4. **Minimum cost selection** across available media and routing choices, including centralized or decentralized
   conversion.

In more detail (`geo_path.py` and transport cost functions):

- **Port selection**: The nearest port to the demand location is chosen using a KD-tree lookup; precomputed shipping
  distances are reused when available to reduce runtime.
- **Multi-leg routing**:
  - **Generation → port**: trucking or pipeline (if allowed).
  - **Port → port**: shipping network route distance.
  - **Port → demand**: trucking or pipeline (if allowed).
- **Carrier evaluation**: For each carrier (NH₃, LOHC, liquid H₂, gaseous H₂), the model evaluates conversion and
  transport costs across legs and chooses the cheapest feasible pathway.
- **Total transport cost**: The minimum transport cost per kg H₂ is added to the generation cost.

### Monte Carlo simulation
The Monte Carlo engine (`mc_main.py`) samples distributions for key parameters (CapEx, OpEx, efficiencies, etc.),
recomputes generation and transport costs across iterations, and returns distributions for total cost per kg H₂. The
Streamlit UI exposes both deterministic overrides and distribution bounds for each sampled parameter.

At a high level:

- Each iteration draws a random value for each uncertain parameter from its distribution.
- Generation and transport costs are recomputed for all sites.
- The resulting cost distributions are aggregated and surfaced in the UI and output files.

## Inputs

### Core model inputs
Provided via the GUI (Streamlit or PyQt):

- **Destination latitude & longitude** (decimal degrees).
- **Annual hydrogen demand** (kilotons per year).
- **Year** (2020, 2030, 2040, 2050).
- **Electrolyzer type** (alkaline, SOEC, PEM).
- **Allow central conversion** (boolean).
- **Allow pipelines** (boolean).
- **Maximum pipeline length** (km).

### Deterministic parameters
The Streamlit UI provides a detailed editable list of deterministic parameter defaults (e.g., discount rate, full load
hours, electrolyzer lifetimes). These are defined in `streamlit_parameters.py` under `DETERMINISTIC_PARAMETERS`.

### Monte Carlo parameters
The Monte Carlo simulation draws from distribution bounds and types defined in `streamlit_parameters.py` under
`MC_PARAMETERS` (triangular distributions for CapEx/OpEx, efficiencies, etc.). The Streamlit UI allows overriding
these bounds before simulation.

## Outputs

### Deterministic run
When running the deterministic model in Streamlit, the tool:

- Writes the full results dataframe to `Results/final_df.csv`.
- Displays the minimum cost and cheapest pathway details in the UI.

### Monte Carlo run
The Monte Carlo run produces:

- A dataframe of per-location cost metrics (shown in the UI).
- Arrays of total cost per kg H₂, generation cost per kg H₂, and solar/wind cost samples.
- When executed via the PyQt workflow (`mc_main.py`), output CSV files are saved under `Results/mc/`.

## How to run

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

### 3) Run the legacy PyQt desktop app

```bash
python MainWindow.py
```

> **Note:** The transport model uses the OSRM public routing API for driving distances. Internet access is required
> for these queries.

## Parameters and configuration

- **Deterministic parameters** live in `streamlit_parameters.py` and are used for single-run deterministic analysis
  and as fixed values in the Monte Carlo simulation.
- **Monte Carlo parameter distributions** are also defined in `streamlit_parameters.py` and sampled in the MC engine
  (`mc_main.py`, `mc_generation_costs.py`).
- **Data inputs** are primarily under `Data/` (renewables, shipping, port indices, etc.).

## Repository layout (key files)

- `streamlit_app.py` — Streamlit UI and orchestration for deterministic & Monte Carlo runs.
- `MainWindow.py` — legacy PyQt desktop application entrypoint.
- `generation_costs.py` — deterministic generation cost calculation.
- `geo_path.py` — transport routing and cost aggregation.
- `mc_main.py` — Monte Carlo engine.
- `streamlit_parameters.py` — parameter definitions for UI and MC simulation.
- `Results/` — output directory for model runs.

## Example workflow

1. Start the Streamlit UI.
2. Enter destination coordinates and demand.
3. Choose the year and electrolyzer type.
4. Decide whether pipelines and centralized conversion are allowed.
5. Optionally enable Monte Carlo, adjust distribution bounds, and run.
6. Review the minimum cost result and exported CSV outputs.
