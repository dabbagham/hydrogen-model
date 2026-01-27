from pandas import DataFrame

from mc_geo_path import *
from mc_generation_costs import *
from mc_parameter_def import *
import numpy
import timeit
import os


class MonteCarloComputing:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set

    def run_mc_model(self, parameter_overrides=None, deterministic_parameters=None):
        latitude = self.parameter_set.get_lat()
        longitude = self.parameter_set.get_long()
        end_tuple = (latitude, longitude)  # [lat, long]
        h2_demand = self.parameter_set.get_demand()
        year = self.parameter_set.get_year()
        centralised = self.parameter_set.get_allow_centralised()
        pipeline = self.parameter_set.get_allow_pipeline()
        max_pipeline_dist = self.parameter_set.get_max_pipe_dist()
        iterations = self.parameter_set.get_iterations()
        elec = self.parameter_set.get_elec_type()

        et = end_tuple
        h2 = h2_demand
        yr = year

        start = timeit.default_timer()

        df, total_cost_per_kg_h2, generation_cost_per_kg_h2, solar_cost, wind_cost = self.mc_main(
            et,
            h2,
            yr,
            centralised,
            pipeline,
            max_pipeline_dist,
            iterations,
            elec,
            parameter_overrides=parameter_overrides,
            deterministic_parameters=deterministic_parameters,
        )

        newpath = "Results/mc/" + str(round(et[0])) + ',' + str(round(et[1])) + '__' + str(
            yr) + '__' + str(h2) + '__' + elec + '__' + str(pipeline) + '__' + str(iterations)
        if pipeline == True:
            newpath = newpath + '__Pipe'
        if centralised == False:
            newpath = newpath + '__decent'

        if not os.path.exists(newpath):
            os.makedirs(newpath)

        np.savetxt(newpath + '/' + 'total_cost_per_kg_h2.csv', total_cost_per_kg_h2, delimiter=",")
        np.savetxt(newpath + '/' + 'generation_cost_per_kg_h2.csv', generation_cost_per_kg_h2,
                   delimiter=",")
        np.savetxt(newpath + '/' + 'solar_cost.csv', solar_cost, delimiter=",")
        np.savetxt(newpath + '/' + 'wind_cost.csv', wind_cost, delimiter=",")

        # stop timer
        stop = timeit.default_timer()
        print('Total Time: ', stop - start)

        df['Total Cost per kg H2'] = df['Total Cost per kg H2'].astype(float)
        cheapest_location_df = df.nsmallest(1, 'Total Cost per kg H2')

        return cheapest_location_df

    def mc_main(self, end_plant_tuple, h2_demand, year=2021, centralised=True, pipeline=True, max_pipeline_dist=2000,
                iterations=1000, elec_type='alkaline', parameter_overrides=None, deterministic_parameters=None):
        """Runs a monte carlo simulation of the model. Takes the desired end location [lat, long], the H2 demand (
        kt/yr), the year, if redistribution is centralised or not, if pipelines are allowed, and the maximum allowed
        pipeline distance (km) as input. Calculates the minimum of (transport + generation) cost for all possible start
        locations to determine the cheapest source of renewable H2. """

        df = pd.read_csv(filepath_or_buffer="Data/renewables.csv", index_col=0)

        total_cost_per_kg_h2 = np.zeros((iterations, len(df)))
        generation_cost_per_kg = np.zeros((iterations, len(df)))
        solar_cost = np.zeros((iterations, len(df)))
        wind_cost = np.zeros((iterations, len(df)))

        cost_end_nh3 = np.zeros(iterations)
        cost_end_lohc = np.zeros(iterations)
        cost_end_h2_liq = np.zeros(iterations)

        # Define parameters for generation costs
        year_diff, capex_extra, capex_h2, lifetime_hours, electrolyser_efficiency, elec_opex, other_capex_elec, water_cost, \
        capex_wind, opex_wind, capex_solar, opex_factor_solar = define_gen_parameters(
            year, iterations, elec_type, overrides=parameter_overrides)

        interest = 0.08
        full_load_hours = 2000
        if deterministic_parameters:
            interest = deterministic_parameters.get("discount_rate", interest)
            full_load_hours = deterministic_parameters.get("full_load_hours", full_load_hours)

        for i in range(iterations):
            df, cost_end_nh3[i], cost_end_lohc[i], cost_end_h2_liq[i] = initial_geo_calcs(df, end_plant_tuple,
                                                                             centralised=centralised,
                                                                             pipeline=pipeline,
                                                                             max_pipeline_dist=max_pipeline_dist)

        for i in range(iterations):
            df = mc_generation_costs(
                df,
                h2_demand,
                year_diff,
                capex_extra[i],
                capex_h2[i],
                lifetime_hours,
                electrolyser_efficiency[i],
                elec_opex[i],
                other_capex_elec[i],
                water_cost[i],
                capex_wind[i],
                opex_wind[i],
                capex_solar[i],
                opex_factor_solar[i],
                interest=interest,
                full_load_hours=full_load_hours,
                parameters=deterministic_parameters,
            )

            df = mc_transport_costs(df, end_plant_tuple, h2_demand, cost_end_nh3[i], cost_end_lohc[i], cost_end_h2_liq[i],
                                    centralised=centralised, pipeline=pipeline,
                                    max_pipeline_dist=max_pipeline_dist)

            df['Total Yearly Cost'] = df['Yearly gen. cost'] + df['Yearly Transport Cost']
            df['Total Cost per kg H2'] = df['Gen. cost per kg H2'] + df['Transport Cost per kg H2']

            total_cost_per_kg_h2[i, :] = df['Total Cost per kg H2'].to_numpy()
            generation_cost_per_kg[i, :] = df['Gen. cost per kg H2'].to_numpy()
            solar_cost[i, :] = df['Elec Cost Solar'].to_numpy()
            wind_cost[i, :] = df['Elec Cost Wind'].to_numpy()

        return df, total_cost_per_kg_h2, generation_cost_per_kg, solar_cost, wind_cost
