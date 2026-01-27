import html
from pathlib import Path

import pandas as pd
import streamlit as st

from generation_costs import generation_costs
from streamlit_parameters import DETERMINISTIC_PARAMETERS, MC_PARAMETERS


DATA_PATH = Path("Data/renewables.csv")


def info_icon(reference: str) -> None:
    safe_reference = html.escape(reference, quote=True)
    st.markdown(f"<span title='{safe_reference}'>ℹ️</span>", unsafe_allow_html=True)


def init_state(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_data
def load_renewables_data():
    return pd.read_csv(DATA_PATH, index_col=0)


def render_deterministic_parameters():
    st.subheader("Deterministic parameters")
    header_cols = st.columns([5, 2, 1, 1])
    header_cols[0].markdown("**Parameter**")
    header_cols[1].markdown("**Value**")
    header_cols[2].markdown("**Unit**")
    header_cols[3].markdown("**Reference**")

    parameters = {}
    for param in DETERMINISTIC_PARAMETERS:
        state_key = f"det_{param['key']}"
        init_state(state_key, param["default"])
        cols = st.columns([5, 2, 1, 1])
        cols[0].write(param["label"])
        parameters[param["key"]] = cols[1].number_input(
            label=param["label"],
            value=st.session_state[state_key],
            key=state_key,
            label_visibility="collapsed",
        )
        cols[2].write(param["unit"])
        with cols[3]:
            info_icon(param["reference"])
    return parameters


def render_mc_parameters():
    st.subheader("Monte Carlo parameters")
    header_cols = st.columns([4, 1, 2, 2, 2, 2, 1])
    header_cols[0].markdown("**Parameter**")
    header_cols[1].markdown("**Unit**")
    header_cols[2].markdown("**Lower bound**")
    header_cols[3].markdown("**Middle**")
    header_cols[4].markdown("**Upper bound**")
    header_cols[5].markdown("**Distribution**")
    header_cols[6].markdown("**Source**")

    overrides = {}
    for param in MC_PARAMETERS:
        cols = st.columns([4, 1, 2, 2, 2, 2, 1])
        cols[0].write(param["label"])
        cols[1].write(param["unit"])

        lower_key = f"mc_{param['lower_key']}"
        init_state(lower_key, param["lower"])
        overrides[param["lower_key"]] = cols[2].number_input(
            label=f"{param['label']} lower",
            value=st.session_state[lower_key],
            key=lower_key,
            label_visibility="collapsed",
        )

        if "middle_key" in param:
            middle_key = f"mc_{param['middle_key']}"
            init_state(middle_key, param["middle"])
            overrides[param["middle_key"]] = cols[3].number_input(
                label=f"{param['label']} middle",
                value=st.session_state[middle_key],
                key=middle_key,
                label_visibility="collapsed",
            )
        else:
            cols[3].write("–")

        upper_key = f"mc_{param['upper_key']}"
        init_state(upper_key, param["upper"])
        overrides[param["upper_key"]] = cols[4].number_input(
            label=f"{param['label']} upper",
            value=st.session_state[upper_key],
            key=upper_key,
            label_visibility="collapsed",
        )
        cols[5].write(param["distribution"])
        with cols[6]:
            info_icon(param["source"])

    return overrides


def run_single_model(parameters, latitude, longitude, demand, year, electrolyser_type, centralised, pipeline,
                     max_pipeline_dist):
    from geo_path import transport_costs
    from print_results import get_path, print_basic_results
    df = load_renewables_data()
    df = generation_costs(
        df,
        demand,
        year=year,
        type=electrolyser_type,
        interest=parameters["discount_rate"],
        full_load_hours=parameters["full_load_hours"],
        parameters=parameters,
    )
    df = transport_costs(
        df,
        (latitude, longitude),
        demand,
        centralised=centralised,
        pipeline=pipeline,
        max_pipeline_dist=max_pipeline_dist,
    )
    df["Total Yearly Cost"] = df["Yearly gen. cost"] + df["Yearly Transport Cost"]
    df["Total Cost per kg H2"] = df["Gen. cost per kg H2"] + df["Transport Cost per kg H2"]
    df.to_csv("Results/final_df.csv")
    min_cost, mindex, cheapest_source, cheapest_medium, cheapest_elec = print_basic_results(df)
    final_path = get_path(df, (latitude, longitude), centralised, pipeline, max_pipeline_dist)
    return min_cost, mindex, cheapest_source, cheapest_medium, cheapest_elec, final_path


def run_mc_model(parameters, mc_overrides, latitude, longitude, demand, year, electrolyser_type, centralised, pipeline,
                 max_pipeline_dist, iterations):
    from mc_main import MonteCarloComputing
    mc_engine = MonteCarloComputing(parameter_set=None)
    return mc_engine.mc_main(
        (latitude, longitude),
        demand,
        year=year,
        centralised=centralised,
        pipeline=pipeline,
        max_pipeline_dist=max_pipeline_dist,
        iterations=iterations,
        elec_type=electrolyser_type,
        parameter_overrides=mc_overrides,
        deterministic_parameters=parameters,
    )


def main():
    st.set_page_config(page_title="Hydrogen Cost Model", layout="wide")
    st.title("Hydrogen Cost Model (Streamlit UI)")

    with st.sidebar:
        st.header("Model inputs")
        latitude = st.number_input("Latitude", value=0.0)
        longitude = st.number_input("Longitude", value=0.0)
        demand = st.number_input("Yearly hydrogen demand (kilotons)", min_value=0.0, value=0.0, step=10.0)
        year = st.selectbox("Year", [2020, 2030, 2040, 2050])
        electrolyser_choice = st.selectbox(
            "Electrolyzer type",
            ["alkaline", "solid oxide electrolyzer cell", "polymer electrolyte membrane"],
        )
        electrolyser_type = {
            "alkaline": "alkaline",
            "solid oxide electrolyzer cell": "SOEC",
            "polymer electrolyte membrane": "PEM",
        }[electrolyser_choice]
        centralised = st.checkbox("Allow central conversion")
        pipeline = st.checkbox("Allow pipelines")
        max_pipeline_dist = st.number_input("Maximum pipeline length (km)", min_value=0, value=0, step=10)

        st.divider()
        st.subheader("Monte Carlo")
        enable_mc = st.checkbox("Run as Monte Carlo simulation")
        iterations = st.number_input("Iterations", min_value=1, value=1000, step=10)

    with st.expander("Deterministic parameters (editable)", expanded=True):
        parameters = render_deterministic_parameters()

    with st.expander("Monte Carlo parameters (editable)", expanded=False):
        mc_overrides = render_mc_parameters()

    st.divider()
    run_label = "Run Monte Carlo" if enable_mc else "Run Model"
    if st.button(run_label, type="primary"):
        with st.spinner("Running model..."):
            if enable_mc:
                df, total_cost_per_kg_h2, generation_cost_per_kg_h2, solar_cost, wind_cost = run_mc_model(
                    parameters,
                    mc_overrides,
                    latitude,
                    longitude,
                    demand,
                    year,
                    electrolyser_type,
                    centralised,
                    pipeline,
                    max_pipeline_dist,
                    iterations,
                )
                st.success("Monte Carlo simulation completed.")
                st.write(df.head())
                st.write("Total cost per kg H2 (sample)", total_cost_per_kg_h2[:5])
                st.write("Generation cost per kg H2 (sample)", generation_cost_per_kg_h2[:5])
                st.write("Solar cost (sample)", solar_cost[:5])
                st.write("Wind cost (sample)", wind_cost[:5])
            else:
                min_cost, mindex, cheapest_source, cheapest_medium, cheapest_elec, final_path = run_single_model(
                    parameters,
                    latitude,
                    longitude,
                    demand,
                    year,
                    electrolyser_type,
                    centralised,
                    pipeline,
                    max_pipeline_dist,
                )
                st.success("Model run completed.")
                st.markdown("### Results")
                st.write(f"Index: {mindex}")
                st.write(f"Minimum cost: {min_cost:.4f} €/kg H₂")
                st.write(f"Cheapest source: {cheapest_source[0]}, {cheapest_source[1]}")
                st.write(f"Cheapest medium: {cheapest_medium}")
                st.write(f"Cheaper electricity: {cheapest_elec}")
                st.write(f"Final path: {final_path}")


if __name__ == "__main__":
    main()
