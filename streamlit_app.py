from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import time
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

from generation_costs import generation_costs
from streamlit_parameters import DETERMINISTIC_PARAMETERS, MC_PARAMETERS


DATA_PATH = Path("Data/renewables.csv")
LOCATIONS_PATH = Path("Data/locationslist.csv")
LOG_LINE_LIMIT = 200


def info_icon(reference: str) -> None:
    with st.popover("ℹ️"):
        st.write(reference)


@st.cache_data
def load_renewables_data():
    return pd.read_csv(DATA_PATH, index_col=0)


@st.cache_data
def load_distribution_nodes():
    if not LOCATIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LOCATIONS_PATH)


def render_results_table(
    df: pd.DataFrame,
    label: str,
    column_config: dict | None = None,
    na_label: str = "Not available",
) -> None:
    st.subheader(label)
    display_df = df.replace("None", np.nan).where(pd.notnull(df), na_label)
    st.dataframe(display_df, use_container_width=True, column_config=column_config)
    st.download_button(
        label=f"Download {label} (CSV)",
        data=df.to_csv(index=False),
        file_name=f"{label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )


def render_cost_map(
    df: pd.DataFrame,
    metric: str,
    end_point: tuple,
    cheapest_source: tuple | None,
    distribution_nodes: pd.DataFrame,
    show_nodes: bool,
    show_demand: bool,
    show_cheapest: bool,
):
    is_numeric = pd.api.types.is_numeric_dtype(df[metric])
    hover_data = {"Latitude": ":.5f", "Longitude": ":.5f"}
    if is_numeric:
        hover_data[metric] = ":.3f"
    fig = px.scatter_geo(
        df,
        lat="Latitude",
        lon="Longitude",
        color=metric,
        color_continuous_scale="Viridis" if is_numeric else None,
        hover_name="Region" if "Region" in df.columns else None,
        hover_data=hover_data,
        title=f"{metric} (global heatmap)",
        height=520,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.8))
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(229, 229, 229)",
            showcountries=True,
            countrycolor="rgb(255, 255, 255)",
            showlakes=True,
            lakecolor="rgb(255, 255, 255)",
        )
    )
    if is_numeric:
        fig.update_layout(coloraxis_colorbar=dict(title="Cost (€/kg H₂)"))
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=40),
    )

    if show_nodes and not distribution_nodes.empty:
        fig.add_trace(
            go.Scattergeo(
                lat=distribution_nodes["Latitude"],
                lon=distribution_nodes["Longitude"],
                mode="markers",
                marker=dict(size=3, color="rgba(50, 50, 50, 0.35)"),
                name="Distribution nodes",
            )
        )

    if show_demand:
        fig.add_trace(
            go.Scattergeo(
                lat=[end_point[0]],
                lon=[end_point[1]],
                mode="markers",
                marker=dict(size=10, color="#e45756", symbol="star"),
                name="Demand location",
            )
        )

    if show_cheapest and cheapest_source:
        fig.add_trace(
            go.Scattergeo(
                lat=[cheapest_source[0]],
                lon=[cheapest_source[1]],
                mode="markers",
                marker=dict(size=9, color="#54a24b", symbol="diamond"),
                name="Cheapest source",
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def render_top_locations(df: pd.DataFrame) -> None:
    st.subheader("Top locations")
    candidate_metrics = [
        "Total Cost per kg H2",
        "Gen. cost per kg H2",
        "Transport Cost per kg H2",
        "Total Yearly Cost",
        "Yearly gen. cost",
        "Yearly Transport Cost",
    ]
    available_metrics = [metric for metric in candidate_metrics if metric in df.columns]
    if not available_metrics:
        st.info("No numeric cost metrics available for ranking.")
        return

    metric = st.selectbox("Rank by", available_metrics, key="top_locations_metric")
    top_n = st.slider("Number of locations", min_value=5, max_value=200, value=25, step=5)

    top_df = df.nsmallest(top_n, metric)
    display_columns = [
        "Latitude",
        "Longitude",
        "Total Cost per kg H2",
        "Gen. cost per kg H2",
        "Transport Cost per kg H2",
        "Cheapest Medium",
        "Cheaper source",
    ]
    display_columns = [col for col in display_columns if col in top_df.columns]
    column_config = {
        "Latitude": st.column_config.NumberColumn(format="%.6f"),
        "Longitude": st.column_config.NumberColumn(format="%.6f"),
        "Total Cost per kg H2": st.column_config.NumberColumn(format="%.3f"),
        "Gen. cost per kg H2": st.column_config.NumberColumn(format="%.3f"),
        "Transport Cost per kg H2": st.column_config.NumberColumn(format="%.3f"),
    }
    render_results_table(
        top_df[display_columns],
        f"Top {top_n} by {metric}",
        column_config=column_config,
    )


def render_cost_breakdown_charts(df: pd.DataFrame) -> None:
    st.subheader("Cost distributions by component")
    st.caption(
        "Production cost reflects on-site generation (electricity + CAPEX/OPEX), transport cost includes conversion "
        "and delivery to the demand location, and total delivered cost is their sum."
    )
    metrics = [
        ("Production cost (€/kg H₂)", "Gen. cost per kg H2"),
        ("Transport cost (€/kg H₂)", "Transport Cost per kg H2"),
        ("Total delivered cost (€/kg H₂)", "Total Cost per kg H2"),
    ]
    cols = st.columns(3)
    for col, (title, metric) in zip(cols, metrics):
        if metric not in df.columns:
            col.info(f"No data for {title}.")
            continue
        series = pd.Series(df[metric]).dropna()
        if series.empty:
            col.info(f"No data for {title}.")
            continue
        fig = px.histogram(series, nbins=60, title=title)
        fig.update_layout(xaxis_title="Cost (€/kg H₂)", yaxis_title="Locations")
        col.plotly_chart(fig, use_container_width=True)


def render_mc_distribution_charts(
    total_cost_per_kg_h2: np.ndarray,
    generation_cost_per_kg_h2: np.ndarray,
    solar_cost: np.ndarray,
    wind_cost: np.ndarray,
) -> None:
    st.subheader("Monte Carlo cost distributions")
    sample_size = st.slider(
        "Samples per distribution",
        min_value=1000,
        max_value=100000,
        value=20000,
        step=1000,
    )

    def sample_array(array: np.ndarray) -> np.ndarray:
        flat = array.ravel()
        if len(flat) <= sample_size:
            return flat
        indices = np.random.choice(len(flat), size=sample_size, replace=False)
        return flat[indices]

    charts = {
        "Total cost per kg H2": sample_array(total_cost_per_kg_h2),
        "Generation cost per kg H2": sample_array(generation_cost_per_kg_h2),
        "Solar cost": sample_array(solar_cost),
        "Wind cost": sample_array(wind_cost),
    }

    for title, values in charts.items():
        series = pd.Series(values).dropna()
        if series.empty:
            st.info(f"No data available for {title}.")
            continue
        fig = px.histogram(series, nbins=60, title=title)
        fig.update_layout(xaxis_title="Cost (€/kg H₂)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        stats = {
            "mean": series.mean(),
            "p50": series.quantile(0.5),
            "p95": series.quantile(0.95),
        }
        st.caption(
            f"Mean: {stats['mean']:.3f} | Median: {stats['p50']:.3f} | 95th percentile: {stats['p95']:.3f}"
        )


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
        cols = st.columns([5, 2, 1, 1])
        cols[0].write(param["label"])
        parameters[param["key"]] = cols[1].number_input(
            label=param["label"],
            value=param["default"],
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
        overrides[param["lower_key"]] = cols[2].number_input(
            label=f"{param['label']} lower",
            value=param["lower"],
            key=lower_key,
            label_visibility="collapsed",
        )

        if "middle_key" in param:
            middle_key = f"mc_{param['middle_key']}"
            overrides[param["middle_key"]] = cols[3].number_input(
                label=f"{param['label']} middle",
                value=param["middle"],
                key=middle_key,
                label_visibility="collapsed",
            )
        else:
            cols[3].write("–")

        upper_key = f"mc_{param['upper_key']}"
        overrides[param["upper_key"]] = cols[4].number_input(
            label=f"{param['label']} upper",
            value=param["upper"],
            key=upper_key,
            label_visibility="collapsed",
        )
        cols[5].write(param["distribution"])
        with cols[6]:
            info_icon(param["source"])

    return overrides


class StreamlitLogSink:
    def __init__(self, placeholder: "st.delta_generator.DeltaGenerator", line_limit: int = LOG_LINE_LIMIT) -> None:
        self._placeholder = placeholder
        self._line_limit = line_limit
        self._lines: List[str] = []

    def write(self, message: str) -> int:
        if not message:
            return 0
        for line in message.splitlines():
            if line.strip() == "":
                continue
            self._lines.append(line)
        if len(self._lines) > self._line_limit:
            self._lines = self._lines[-self._line_limit :]
        self._placeholder.code("\n".join(self._lines))
        return len(message)

    def flush(self) -> None:
        return None


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
    return df, min_cost, mindex, cheapest_source, cheapest_medium, cheapest_elec, final_path


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
    st.markdown(
        """
        **Methodology overview**
        - This model follows the global hydrogen production and transport cost framework described by Collis & Schomäcker
          (2022) and computes levelized production costs for each renewable resource location, then evaluates transport
          options (H₂ gas, LOHC, NH₃, or liquid H₂ where applicable) to a user-defined demand point.
        - For each candidate location, the model combines **production cost** (electricity + CAPEX/OPEX and efficiency
          assumptions) with **transport cost** (conversion, shipping, and distribution) to estimate the **total
          delivered cost** in €/kg H₂.
        - The model reports the cheapest source, transport medium, and full path for the selected demand point, while
          retaining the full global grid for ranking and visualization.
        - Scenario years begin at 2020 because the underlying techno-economic data and baseline assumptions are anchored
          to 2020 in the reference study; later decades (2030/2040/2050) reflect forward-looking learning curves and
          cost projections. This keeps comparisons consistent with the paper and allows users to examine trajectories
          relative to a common baseline.
        """
    )
    st.markdown(
        """
        Use the inputs in the left sidebar to define the project location and demand, then review or adjust the
        deterministic assumptions below. Enable Monte Carlo to edit uncertainty ranges and run the stochastic model.
        """
    )
    st.caption(
        "Tip: Hover/click the ℹ️ icons in the parameter tables to view data sources and references."
    )

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "selected_latitude" not in st.session_state:
        st.session_state["selected_latitude"] = 0.0
    if "selected_longitude" not in st.session_state:
        st.session_state["selected_longitude"] = 0.0

    with st.sidebar:
        st.header("Model inputs")
        st.markdown(
            """
            **How to use**
            - Enter the project **location** (latitude/longitude), **demand**, and **year**.
            - Choose the **electrolyzer type** and optional **transport assumptions**.
            - Keep the defaults if you don't want to override model assumptions.
            """
        )
        location_mode = st.radio(
            "Location input mode",
            ["Manual entry", "Pick from map"],
            horizontal=True,
        )
        if location_mode == "Manual entry":
            st.session_state["selected_latitude"] = st.number_input(
                "Latitude",
                value=st.session_state["selected_latitude"],
                format="%.6f",
                step=0.0001,
            )
            st.session_state["selected_longitude"] = st.number_input(
                "Longitude",
                value=st.session_state["selected_longitude"],
                format="%.6f",
                step=0.0001,
            )
        else:
            st.caption("Select a grid point from the map below to populate latitude/longitude.")
            st.number_input(
                "Latitude",
                value=st.session_state["selected_latitude"],
                format="%.6f",
                step=0.0001,
                disabled=True,
            )
            st.number_input(
                "Longitude",
                value=st.session_state["selected_longitude"],
                format="%.6f",
                step=0.0001,
                disabled=True,
            )
        latitude = st.session_state["selected_latitude"]
        longitude = st.session_state["selected_longitude"]
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
        enable_mc = st.toggle(
            "Run as Monte Carlo simulation",
            value=False,
            help="Enable to run stochastic simulations using the Monte Carlo parameter ranges.",
        )
        iterations = st.number_input("Iterations", min_value=1, value=1000, step=10)

    with st.expander("Deterministic parameters (editable)", expanded=True):
        st.caption("These inputs are always used. Adjust values to override model defaults.")
        parameters = render_deterministic_parameters()

    with st.expander("Monte Carlo parameters (editable)", expanded=False):
        if enable_mc:
            st.caption("These ranges are applied only when Monte Carlo simulation is enabled.")
            mc_overrides = render_mc_parameters()
        else:
            st.info("Enable Monte Carlo in the sidebar to edit these uncertainty ranges.")
            mc_overrides = {}

    if location_mode == "Pick from map":
        st.subheader("Select location from map")
        st.caption("Click a point to populate the latitude/longitude inputs.")
        map_df = load_renewables_data()[["Latitude", "Longitude"]].dropna()
        if len(map_df) > 5000:
            map_df = map_df.sample(5000, random_state=42)
        select_fig = px.scatter_geo(
            map_df,
            lat="Latitude",
            lon="Longitude",
            opacity=0.6,
            height=400,
        )
        select_fig.update_traces(marker=dict(size=4, color="#4c78a8"))
        select_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            geo=dict(
                showland=True,
                landcolor="rgb(229, 229, 229)",
                showcountries=True,
                countrycolor="rgb(255, 255, 255)",
                showlakes=True,
                lakecolor="rgb(255, 255, 255)",
            ),
        )
        selection = st.plotly_chart(select_fig, use_container_width=True, on_select="rerun")
        if selection and selection.get("points"):
            point = selection["points"][0]
            selected_lat = point.get("lat")
            selected_lon = point.get("lon")
            if selected_lat is not None and selected_lon is not None:
                st.session_state["selected_latitude"] = float(selected_lat)
                st.session_state["selected_longitude"] = float(selected_lon)
                st.success(
                    f"Selected location updated to {selected_lat:.6f}, {selected_lon:.6f}."
                )

    st.divider()
    run_label = "Run Monte Carlo" if enable_mc else "Run Model"
    if st.button(run_label, type="primary"):
        st.markdown("### Model run status")
        st.info(
            "Live logs below show model stages, warnings, and progress updates. "
            "If the run appears stalled, check for repeated warnings or missing paths."
        )
        log_container = st.expander("Live run log", expanded=True)
        log_placeholder = log_container.empty()
        log_sink = StreamlitLogSink(log_placeholder)
        with st.status("Running model...", expanded=True) as status:
            with redirect_stdout(log_sink), redirect_stderr(log_sink):
                start_time = time.perf_counter()
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
                    runtime = time.perf_counter() - start_time
                    st.session_state["results"] = {
                        "mode": "mc",
                        "df": df,
                        "runtime": runtime,
                        "arrays": {
                            "total_cost_per_kg_h2": total_cost_per_kg_h2,
                            "generation_cost_per_kg_h2": generation_cost_per_kg_h2,
                            "solar_cost": solar_cost,
                            "wind_cost": wind_cost,
                        },
                    }
                else:
                    df, min_cost, mindex, cheapest_source, cheapest_medium, cheapest_elec, final_path = run_single_model(
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
                    runtime = time.perf_counter() - start_time
                    st.session_state["results"] = {
                        "mode": "deterministic",
                        "df": df,
                        "runtime": runtime,
                        "min_cost": min_cost,
                        "mindex": mindex,
                        "cheapest_source": cheapest_source,
                        "cheapest_medium": cheapest_medium,
                        "cheapest_elec": cheapest_elec,
                        "final_path": final_path,
                    }
            status.update(label="Model run completed.", state="complete", expanded=False)

    results = st.session_state.get("results")
    if results:
        if results["mode"] == "mc":
            st.success("Monte Carlo simulation completed.")
            st.metric("Run time (seconds)", f"{results['runtime']:.2f}")
            render_results_table(results["df"], "Cheapest location summary")
            with st.expander("Cost distribution samples", expanded=False):
                st.write("Total cost per kg H2 (sample)", results["arrays"]["total_cost_per_kg_h2"][:5])
                st.write(
                    "Generation cost per kg H2 (sample)",
                    results["arrays"]["generation_cost_per_kg_h2"][:5],
                )
                st.write("Solar cost (sample)", results["arrays"]["solar_cost"][:5])
                st.write("Wind cost (sample)", results["arrays"]["wind_cost"][:5])
            render_mc_distribution_charts(
                results["arrays"]["total_cost_per_kg_h2"],
                results["arrays"]["generation_cost_per_kg_h2"],
                results["arrays"]["solar_cost"],
                results["arrays"]["wind_cost"],
            )
        else:
            st.success("Model run completed.")
            st.metric("Run time (seconds)", f"{results['runtime']:.2f}")
            st.markdown("### Results")
            if results["mindex"] is None or results["cheapest_source"] is None:
                st.warning(
                    "No valid minimum cost was found. Check inputs, transport assumptions, or missing data."
                )
            else:
                st.write(f"Index: {results['mindex']}")
                st.write(f"Minimum cost: {results['min_cost']:.4f} €/kg H₂")
                st.write(
                    f"Cheapest source: {results['cheapest_source'][0]}, {results['cheapest_source'][1]}"
                )
                st.write(f"Cheapest medium: {results['cheapest_medium']}")
                st.write(f"Cheaper electricity: {results['cheapest_elec']}")
                st.write(f"Final path: {results['final_path']}")

            st.caption(
                "Cells marked 'Not available' indicate that a transport pathway is infeasible for that location or that "
                "the input data is missing for the selected assumptions."
            )
            render_results_table(results["df"], "Final results")
            render_cost_breakdown_charts(results["df"])
            with st.expander("Top locations", expanded=False):
                render_top_locations(results["df"])

            st.subheader("Interactive cost heatmap")
            metric = st.selectbox(
                "Metric",
                [
                    "Total Cost per kg H2",
                    "Gen. cost per kg H2",
                    "Transport Cost per kg H2",
                    "Cheapest Medium",
                ],
                key="map_metric",
            )
            map_scope = st.radio(
                "Map scope",
                ["All locations", "Top locations"],
                horizontal=True,
                key="map_scope",
            )
            show_nodes = st.checkbox("Show distribution nodes", value=True, key="show_nodes")
            show_demand = st.checkbox("Show demand marker", value=True, key="show_demand")
            show_cheapest = st.checkbox("Show cheapest source marker", value=True, key="show_cheapest")
            map_df = results["df"]
            if map_scope == "Top locations":
                numeric_metrics = [
                    "Total Cost per kg H2",
                    "Gen. cost per kg H2",
                    "Transport Cost per kg H2",
                ]
                map_metric = metric if metric in numeric_metrics else "Total Cost per kg H2"
                map_top_n = st.slider("Map top N", min_value=50, max_value=1000, value=250, step=50)
                map_df = results["df"].nsmallest(map_top_n, map_metric)
            distribution_nodes = load_distribution_nodes()
            render_cost_map(
                map_df,
                metric,
                (latitude, longitude),
                results["cheapest_source"],
                distribution_nodes,
                show_nodes,
                show_demand,
                show_cheapest,
            )


if __name__ == "__main__":
    main()
