import numpy as np


def _get_override(overrides, key, default):
    if overrides is None:
        return default
    return overrides.get(key, default)


def normalize(min, max, array):
    """Normalizes 'value' between min and max."""

    normalized = (array - min) / (max - min)

    return normalized


def define_gen_parameters(year, iterations, type='alkaline', overrides=None):
    """Defines distributions for the parameters for the monte carlo simulation."""

    if 2020 <= year <= 2050:
        year_diff = year - 2020
    elif year < 2020:
        year_diff = 0
    elif year > 2050:
        year_diff = 30

    # Determination of electrolyser parameters
    if type == 'alkaline':
        capex_extra = np.random.triangular(
            _get_override(overrides, "ae_external_capex_lower", 2.30),
            _get_override(overrides, "ae_external_capex_middle", 2.47),
            _get_override(overrides, "ae_external_capex_upper", 2.65),
            (iterations, 1),
        )  # [Eur/kg h2]
        capex_h2_min = _get_override(overrides, "ae_electrolyser_capex_lower", 477)
        capex_h2_mid = _get_override(overrides, "ae_electrolyser_capex_middle", 830)
        capex_h2_max = _get_override(overrides, "ae_electrolyser_capex_upper", 1060)
        capex_h2 = np.random.triangular(
            capex_h2_min,
            capex_h2_mid,
            capex_h2_max,
            (iterations, 1),
        )  # [Eur/kW]
        capex_growth_low = _get_override(overrides, "electrolyser_capex_growth_lower", 0.975)
        capex_growth_mid = _get_override(overrides, "electrolyser_capex_growth_middle", 0.98)
        capex_growth_high = _get_override(overrides, "electrolyser_capex_growth_upper", 0.995)
        capex_growth = np.where(
            capex_h2 < capex_h2_mid,
            capex_growth_mid + normalize(capex_h2_min, capex_h2_mid, capex_h2) * (capex_growth_high - capex_growth_mid),
            capex_growth_low + normalize(capex_h2_mid, capex_h2_max, capex_h2) * (capex_growth_mid - capex_growth_low),
        )
        capex_h2 = capex_h2 * (capex_growth ** year_diff)  # [Eur/kW]
        lifetime_hours = _get_override(overrides, "ae_lifetime_hours", 75000) + _get_override(
            overrides, "ae_lifetime_growth_per_year", 1667) * year_diff  # [hours]
        electrolyser_efficiency = np.random.uniform(
            _get_override(overrides, "ae_efficiency_lower", 0.67),
            _get_override(overrides, "ae_efficiency_upper", 0.7),
            (iterations, 1),
        ) + _get_override(overrides, "ae_efficiency_growth_per_year", 0.002666) * year_diff  # []
    elif type == 'SOEC':
        capex_extra = np.random.triangular(
            _get_override(overrides, "soe_external_capex_lower", 2.30),
            _get_override(overrides, "soe_external_capex_middle", 2.47),
            _get_override(overrides, "soe_external_capex_upper", 2.65),
            (iterations, 1),
        )  # [Eur/kg h2]
        capex_h2_min = _get_override(overrides, "soe_electrolyser_capex_lower", 566)
        capex_h2_mid = _get_override(overrides, "soe_electrolyser_capex_middle", 1131)
        capex_h2_max = _get_override(overrides, "soe_electrolyser_capex_upper", 1912)
        capex_h2 = np.random.triangular(
            capex_h2_min,
            capex_h2_mid,
            capex_h2_max,
            (iterations, 1),
        )  # [Eur/kW]
        capex_growth_low = _get_override(overrides, "electrolyser_capex_growth_lower", 0.975)
        capex_growth_mid = _get_override(overrides, "electrolyser_capex_growth_middle", 0.98)
        capex_growth_high = _get_override(overrides, "electrolyser_capex_growth_upper", 0.995)
        capex_growth = np.where(
            capex_h2 < capex_h2_mid,
            capex_growth_mid + normalize(capex_h2_min, capex_h2_mid, capex_h2) * (capex_growth_high - capex_growth_mid),
            capex_growth_low + normalize(capex_h2_mid, capex_h2_max, capex_h2) * (capex_growth_mid - capex_growth_low),
        )
        capex_h2 = capex_h2 * (capex_growth ** year_diff)  # [Eur/kW]
        lifetime_hours = _get_override(overrides, "soe_lifetime_hours", 20000) + _get_override(
            overrides, "soe_lifetime_growth_per_year", 2167) * year_diff  # [hours]
        electrolyser_efficiency = np.random.uniform(
            _get_override(overrides, "soe_efficiency_lower", 0.77),
            _get_override(overrides, "soe_efficiency_upper", 0.81),
            (iterations, 1),
        ) + _get_override(overrides, "soe_efficiency_growth_per_year", 0.002666) * year_diff  # []
    else:
        capex_extra = np.random.triangular(
            _get_override(overrides, "pem_external_capex_lower", 0.8),
            _get_override(overrides, "pem_external_capex_middle", 0.91),
            _get_override(overrides, "pem_external_capex_upper", 1),
            (iterations, 1),
        )  # [Eur/kg h2]
        capex_h2_min = _get_override(overrides, "pem_electrolyser_capex_lower", 322)
        capex_h2_mid = _get_override(overrides, "pem_electrolyser_capex_middle", 994)
        capex_h2_max = _get_override(overrides, "pem_electrolyser_capex_upper", 1731)
        capex_h2 = np.random.triangular(
            capex_h2_min,
            capex_h2_mid,
            capex_h2_max,
            (iterations, 1),
        )  # [Eur/kW]
        capex_growth_low = _get_override(overrides, "electrolyser_capex_growth_lower", 0.975)
        capex_growth_mid = _get_override(overrides, "electrolyser_capex_growth_middle", 0.98)
        capex_growth_high = _get_override(overrides, "electrolyser_capex_growth_upper", 0.995)
        capex_growth = np.where(
            capex_h2 < capex_h2_mid,
            capex_growth_mid + normalize(capex_h2_min, capex_h2_mid, capex_h2) * (capex_growth_high - capex_growth_mid),
            capex_growth_low + normalize(capex_h2_mid, capex_h2_max, capex_h2) * (capex_growth_mid - capex_growth_low),
        )
        capex_h2 = capex_h2 * (capex_growth ** year_diff)  # [Eur/kW]
        lifetime_hours = _get_override(overrides, "pem_lifetime_hours", 60000) + _get_override(
            overrides, "pem_lifetime_growth_per_year", 2250) * year_diff  # [hours]
        electrolyser_efficiency = np.random.uniform(
            _get_override(overrides, "pem_efficiency_lower", 0.58),
            _get_override(overrides, "pem_efficiency_upper", 0.6),
            (iterations, 1),
        ) + _get_override(overrides, "pem_efficiency_growth_per_year", 0.004) * year_diff  # []

    elec_opex = np.random.triangular(
        _get_override(overrides, "electrolyser_opex_lower", 0.01),
        _get_override(overrides, "electrolyser_opex_middle", 0.015),
        _get_override(overrides, "electrolyser_opex_upper", 0.02),
        (iterations, 1),
    )  # [% of elec CapEx]
    other_capex_elec = np.random.triangular(
        _get_override(overrides, "electrolyser_other_capex_lower", 30),
        _get_override(overrides, "electrolyser_other_capex_middle", 41.6),
        _get_override(overrides, "electrolyser_other_capex_upper", 50),
        (iterations, 1),
    )  # [Eur/kW]
    water_cost = np.random.triangular(
        _get_override(overrides, "electrolyser_water_lower", 0.05),
        _get_override(overrides, "electrolyser_water_middle", 0.07),
        _get_override(overrides, "electrolyser_water_upper", 0.09),
        (iterations, 1),
    )  # [Eur/kg H2]

    # Determination of wind power parameters
    if year <= 2030:
        capex_wind = np.random.triangular(
            _get_override(overrides, "wind_capex_lower", 1200),
            _get_override(overrides, "wind_capex_middle", 1260),
            _get_override(overrides, "wind_capex_upper", 1500),
            (iterations, 1),
        ) * (0.9775 ** year_diff)   #[Eur/kW]
    else:
        capex_wind = np.random.triangular(
            _get_override(overrides, "wind_capex_lower", 1200),
            _get_override(overrides, "wind_capex_middle", 1260),
            _get_override(overrides, "wind_capex_upper", 1500),
            (iterations, 1),
        ) * (0.9775 ** 10) * (
            0.9985 ** (year - 2030))
    opex_wind = np.random.triangular(
        _get_override(overrides, "wind_opex_lower", 6),
        _get_override(overrides, "wind_opex_middle", 8),
        _get_override(overrides, "wind_opex_upper", 10),
        (iterations, 1),
    )  # [Eur/MWh]

    # Determination of solar parameters
    capex_solar = np.random.triangular(
        _get_override(overrides, "solar_capex_lower", 500),
        _get_override(overrides, "solar_capex_middle", 700),
        _get_override(overrides, "solar_capex_upper", 1100),
        (iterations, 1),
    ) * (0.9986 ** year_diff)  # [Eur/kWp]
    opex_factor_solar = np.random.triangular(
        _get_override(overrides, "solar_opex_lower", 0.01),
        _get_override(overrides, "solar_opex_middle", 0.015),
        _get_override(overrides, "solar_opex_upper", 0.02),
        (iterations, 1),
    )  # []

    return year_diff, capex_extra, capex_h2, lifetime_hours, electrolyser_efficiency, elec_opex, other_capex_elec, water_cost, capex_wind, opex_wind, capex_solar, opex_factor_solar
