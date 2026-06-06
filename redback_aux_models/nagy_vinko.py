import numpy as np
import astropy.units as uu
from astropy.cosmology import Planck18 as cosmo
from collections import namedtuple
from scipy.ndimage import percentile_filter

import redback.photosphere as photosphere
import redback.sed as sed
from redback.constants import day_to_s, km_cgs, solar_mass
from redback.sed import flux_density_to_spectrum
from redback.utils import (
    calc_kcorrected_properties, citation_wrapper, get_optimal_time_array,
    lambda_to_nu,
)
from redback.wrappers import cond_jit

@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_psi(x):
    if x == 1.0:
        return 0.0
    if x > 0.0:
        return np.sin(np.pi * x) / (np.pi * x)
    return 1.0


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_eta(x, exponent):
    if x < 0.4:
        return 1.0
    return np.exp(-exponent * (x - 0.4))


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_theta(x, exponent):
    if x < 0.4:
        return 1.0
    return (x / 0.4) ** (-exponent)


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_series(x):
    pix = np.pi * x
    pix2 = pix * pix
    return 1.0 - pix2 / 6.0 + pix2 * pix2 / 120.0 - pix2 ** 3 / 5040.0 + pix2 ** 4 / 362880.0


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_series_derivative(x):
    pix2 = np.pi * np.pi
    return (
        -pix2 * x / 3.0
        + pix2 * pix2 * x ** 3.0 / 30.0
        - pix2 ** 3.0 * x ** 5.0 / 840.0
        + pix2 ** 4.0 * x ** 7.0 / 45360.0)


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_psi_derivative(x):
    pix = np.pi * x
    return (pix * np.cos(pix) - np.sin(pix)) / (np.pi * x * x)


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_im_integral(b, density_exponent, moment_power):
    if density_exponent == 0.0:
        return b ** (moment_power + 1.0) / (moment_power + 1.0)
    n_steps = 20000
    dx = b / n_steps
    total = 0.0
    x = 0.0
    for _ in range(n_steps):
        xp = x + dx
        total += 0.5 * (
            _lc2_eta(x, density_exponent) * x ** moment_power
            + _lc2_eta(xp, density_exponent) * xp ** moment_power
        ) * dx
        x = xp
    return total


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_smooth_floor(value, floor, width):
    if width <= 0.0:
        if value < floor:
            return floor
        return value
    arg = (value - floor) / width
    if arg > 50.0:
        return value
    if arg < -50.0:
        return floor
    return floor + width * np.log1p(np.exp(arg))


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_smooth_inner_front(value, floor, width):
    if width <= 0.0:
        if value < floor:
            return floor
        return value
    floor_squared = floor * floor
    width_squared = 2.0 * floor * width
    value_squared = value * value
    return np.sqrt(_lc2_smooth_floor(value_squared, floor_squared, width_squared))


@cond_jit(nopython=True, fastmath=True, cache=False)
def _lc2_recombination_front(
        central_temperature, previous_xi, recombination_temperature, reference_mode,
        smoothing_width):
    if recombination_temperature <= 0.0:
        return 1.0
    if central_temperature <= 0.0:
        return 0.4
    if reference_mode:
        y = previous_xi
        front_resolution = 1e-9
        local_temperature = 0.0
        while y >= 0.4 and local_temperature < recombination_temperature:
            local_temperature = central_temperature * _lc2_psi(y) ** 0.25
            y -= front_resolution
        if y < previous_xi:
            return y + 0.5 * front_resolution
        return previous_xi

    target = (recombination_temperature / central_temperature) ** 4
    if not reference_mode and _lc2_psi(previous_xi) >= target:
        return previous_xi

    low = 0.0
    if target >= 1.0:
        raw_front = 0.0
    else:
        high = previous_xi
        for _ in range(60):
            mid = 0.5 * (low + high)
            if _lc2_psi(mid) >= target:
                low = mid
            else:
                high = mid
        raw_front = low

    return min(previous_xi, _lc2_smooth_inner_front(raw_front, 0.4, smoothing_width))


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_component_luminosity_kernel(
        time_days, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
        thermal_energy, exponential_density, powerlaw_density, kappa, magnetar_energy,
        magnetar_timescale, gamma_leakage, integration_timestep, reference_mode,
        recombination_max_delta, minimum_timestep, recombination_smoothing_width,
        recombination_fine_xi):
    day = 86400.0
    xmin = 0.4
    e_ni = 3.89e10
    e_co = 6.8e9
    c_lc2 = 2.99e10
    m_sun = 1.989e33
    radiation_constant_lc2 = 7.57e-15

    output = np.zeros(len(time_days))
    if len(time_days) == 0:
        return output
    if powerlaw_density == 3.0 or powerlaw_density == 5.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    mass = mej * m_sun
    m_ni = nickel_mass * m_sun
    e_kin = kinetic_energy * 1.0e51
    e_th_input = thermal_energy * 1.0e51
    e_mag = magnetar_energy * 1.0e51
    t_mag = magnetar_timescale * day
    gamma_coeff = gamma_leakage * day * day
    t_ni = 8.8 * day
    t_co = 111.3 * day
    dt_seconds = max(integration_timestep, 1.0)

    t_initial = (e_th_input * np.pi / (4.0 * radius ** 3 * radiation_constant_lc2)) ** 0.25
    ith = 1.0 / (np.pi * np.pi)
    im = _lc2_im_integral(1.0, exponential_density, 2.0)
    if powerlaw_density == 0.0:
        f_mass = im
        g_kinetic = _lc2_im_integral(1.0, exponential_density, 4.0)
    else:
        f_mass = (3.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 3.0) / (3.0 * (3.0 - powerlaw_density))
        g_kinetic = (5.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 5.0) / (5.0 * (5.0 - powerlaw_density))

    velocity = np.sqrt(2.0 * e_kin * f_mass / (g_kinetic * mass))
    rho0 = mass / (4.0 * np.pi * radius ** 3 * f_mass)
    diffusion_time = 3.0 * kappa * rho0 * radius * radius / (np.pi * np.pi * c_lc2)
    hydro_time = radius / velocity
    e_th = 4.0 * np.pi * radius ** 3 * radiation_constant_lc2 * t_initial ** 4.0 * ith
    if e_th <= 0.0 or diffusion_time <= 0.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    p1 = e_ni * m_ni * t_ni / e_th
    p2 = t_ni / diffusion_time
    p3 = t_ni / e_th
    p4 = t_ni / t_co
    p5 = e_co / e_ni

    t = 0.0
    energy_factor = 1.0
    xh = 1.0
    xi = 1.0
    x_ni = 1.0
    x_co = 0.0
    ionization_energy = 1.6e13
    target_index = 0
    max_time_seconds = time_days[-1] * day
    previous_time = 0.0
    previous_luminosity = 0.0

    while target_index < len(time_days) and time_days[target_index] < 0.0:
        output[target_index] = 0.0
        target_index += 1

    while t <= max_time_seconds + dt_seconds and target_index < len(time_days):
        step_seconds = dt_seconds
        if not reference_mode and recombination_temperature > 0.0 and xi > xmin:
            if xi <= recombination_fine_xi:
                step_seconds = min(step_seconds, minimum_timestep)
            sigma_now = (radius + velocity * t) / radius
            central_temperature_now = t_initial * energy_factor ** 0.25 / sigma_now
            if central_temperature_now > recombination_temperature:
                xi_now = _lc2_recombination_front(
                    central_temperature_now, xh, recombination_temperature, False,
                    recombination_smoothing_width)
                sigma_next = (radius + velocity * (t + step_seconds)) / radius
                central_temperature_next = t_initial * energy_factor ** 0.25 / sigma_next
                xi_next = _lc2_recombination_front(
                    central_temperature_next, xi_now, recombination_temperature, False,
                    recombination_smoothing_width)
                front_delta = abs(xi_next - xi_now)
                if front_delta > recombination_max_delta and recombination_max_delta > 0.0:
                    step_seconds = max(
                        minimum_timestep,
                        step_seconds * recombination_max_delta / front_delta)

        if t == 0.0:
            opacity_factor = 1.0
        else:
            opacity_factor = 1.0 - np.exp(-gamma_coeff / (t * t))

        if t_mag == 0.0:
            magnetar_luminosity = 0.0
            magnetar_luminosity_half = 0.0
        else:
            magnetar_luminosity = e_mag / (t_mag * (1.0 + t / t_mag) ** 2.0)
            magnetar_luminosity_half = e_mag / (t_mag * (1.0 + (t + 0.5 * step_seconds) / t_mag) ** 2.0)

        photosphere_radius = radius + velocity * t
        sigma = photosphere_radius / radius
        sigma_half = (radius + velocity * (t + 0.5 * step_seconds)) / radius
        central_temperature = t_initial * energy_factor ** 0.25 / sigma

        if xi >= xmin and (
                central_temperature > recombination_temperature
                or (not reference_mode and recombination_temperature > 0.0)):
            xi = _lc2_recombination_front(
                central_temperature, xh, recombination_temperature, reference_mode,
                recombination_smoothing_width)
        else:
            xi = xmin

        dxi = (_lc2_series(xh) - _lc2_series(xi)) * xi
        heating = x_ni + p5 * x_co
        d_x_ni = -x_ni * step_seconds / t_ni
        d_x_co = (x_ni - p4 * x_co) * step_seconds / t_ni

        recombination_radius = xi * photosphere_radius
        recombination_dr = dxi * photosphere_radius
        if recombination_dr > 0.0:
            recombination_dr = 0.0
        if powerlaw_density == 0.0:
            recombination_luminosity = (
                -4.0 * np.pi * rho0 * _lc2_eta(xi, exponential_density)
                * sigma ** -3.0 * ionization_energy * recombination_radius ** 2.0
                * recombination_dr / step_seconds
            )
        else:
            recombination_luminosity = (
                -4.0 * np.pi * rho0 * _lc2_theta(xi, powerlaw_density)
                * sigma ** -3.0 * ionization_energy * recombination_radius ** 2.0
                * recombination_dr / step_seconds
            )
        if recombination_luminosity < 0.0:
            recombination_luminosity = 0.0

        diffusion_luminosity = xi * energy_factor * e_th * opacity_factor / diffusion_time
        luminosity = diffusion_luminosity + recombination_luminosity

        while target_index < len(time_days) and time_days[target_index] * day <= t:
            target_time = time_days[target_index] * day
            if t == previous_time:
                output[target_index] = luminosity
            else:
                weight = (target_time - previous_time) / (t - previous_time)
                output[target_index] = previous_luminosity + weight * (luminosity - previous_luminosity)
            target_index += 1

        d_f_1 = sigma / xi ** 3.0 * (
            p1 * heating - p2 * energy_factor * xi
            - 2.0 * energy_factor * xi ** 2.0 * dxi * t_ni / (sigma * step_seconds)
            + p3 * magnetar_luminosity
        )
        d_f_2 = sigma_half / xi ** 3.0 * (
            p1 * heating - p2 * (energy_factor + d_f_1 * step_seconds * 0.5 / t_ni) * xi
            - 2.0 * (energy_factor + d_f_1 * step_seconds * 0.5 / t_ni) * xi ** 2.0
            * dxi * t_ni / (sigma_half * step_seconds) + p3 * magnetar_luminosity_half
        )
        d_f_3 = sigma_half / xi ** 3.0 * (
            p1 * heating - p2 * (energy_factor + d_f_2 * step_seconds * 0.5 / t_ni) * xi
            - 2.0 * (energy_factor + d_f_2 * step_seconds * 0.5 / t_ni) * xi ** 2.0
            * dxi * t_ni / (sigma_half * step_seconds) + p3 * magnetar_luminosity_half
        )
        d_f_4 = sigma_half / xi ** 3.0 * (
            p1 * heating - p2 * (energy_factor + d_f_3 * step_seconds / t_ni) * xi
            - 2.0 * (energy_factor + d_f_3 * step_seconds / t_ni) * xi ** 2.0
            * dxi * t_ni / (sigma_half * step_seconds) + p3 * magnetar_luminosity_half
        )
        energy_factor += (d_f_1 + 2.0 * d_f_2 + 2.0 * d_f_3 + d_f_4) * step_seconds / (6.0 * t_ni)
        if (not reference_mode) and energy_factor < 0.0:
            energy_factor = 0.0

        x_ni += d_x_ni
        x_co += d_x_co
        xh = xi
        previous_time = t
        previous_luminosity = luminosity
        t += step_seconds

    return output


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_reduced_xi_and_xidot(
        t, energy_factor, x_ni, x_co, radius, velocity, t_initial,
        recombination_temperature, p1, p2, p3, p5, t_ni, magnetar_energy,
        magnetar_timescale):
    xmin = 0.4
    if recombination_temperature <= 0.0:
        return 1.0, 0.0
    if energy_factor <= 0.0:
        return xmin, 0.0

    sigma = (radius + velocity * t) / radius
    central_temperature = t_initial * energy_factor ** 0.25 / sigma
    if central_temperature <= recombination_temperature:
        return xmin, 0.0

    xi = _lc2_recombination_front(
        central_temperature, 1.0, recombination_temperature, False, 0.0)
    if xi <= xmin:
        return xmin, 0.0

    psi_value = _lc2_psi(xi)
    psi_derivative = _lc2_psi_derivative(xi)
    if psi_value <= 0.0 or psi_derivative == 0.0:
        return xi, 0.0
    series_derivative = _lc2_series_derivative(xi)

    if magnetar_timescale == 0.0:
        magnetar_luminosity = 0.0
    else:
        magnetar_luminosity = (
            magnetar_energy / (magnetar_timescale * (1.0 + t / magnetar_timescale) ** 2.0))

    heating = x_ni + p5 * x_co
    heating_term = p1 * heating - p2 * energy_factor * xi + p3 * magnetar_luminosity
    sigma_log_derivative = velocity / (radius + velocity * t)
    unconstrained_f_log_derivative = sigma * heating_term / (xi ** 3.0 * t_ni * energy_factor)
    denominator = 2.0 * series_derivative + psi_derivative / psi_value
    if denominator == 0.0:
        return xi, 0.0
    xi_dot = (4.0 * sigma_log_derivative - unconstrained_f_log_derivative) / denominator
    if xi_dot > 0.0:
        xi_dot = 0.0
    return xi, xi_dot


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_reduced_derivatives(
        t, energy_factor, x_ni, x_co, radius, velocity, t_initial,
        recombination_temperature, p1, p2, p3, p4, p5, t_ni,
        magnetar_energy, magnetar_timescale):
    xi, xi_dot = _nagy_vinko_reduced_xi_and_xidot(
        t, energy_factor, x_ni, x_co, radius, velocity, t_initial,
        recombination_temperature, p1, p2, p3, p5, t_ni, magnetar_energy,
        magnetar_timescale)
    sigma = (radius + velocity * t) / radius
    if magnetar_timescale == 0.0:
        magnetar_luminosity = 0.0
    else:
        magnetar_luminosity = (
            magnetar_energy / (magnetar_timescale * (1.0 + t / magnetar_timescale) ** 2.0))
    heating = x_ni + p5 * x_co
    heating_term = p1 * heating - p2 * energy_factor * xi + p3 * magnetar_luminosity
    front_series_derivative = -_lc2_series_derivative(xi) * xi * xi_dot
    d_energy_factor = (
        sigma * heating_term / (xi ** 3.0 * t_ni)
        - 2.0 * energy_factor * front_series_derivative / xi)
    d_x_ni = -x_ni / t_ni
    d_x_co = (x_ni - p4 * x_co) / t_ni
    return d_energy_factor, d_x_ni, d_x_co


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_reduced_luminosity(
        t, energy_factor, x_ni, x_co, radius, velocity, rho0, e_th,
        diffusion_time, gamma_coeff, exponential_density, powerlaw_density,
        recombination_temperature, ionization_energy, t_initial, p1, p2, p3, p5,
        t_ni, magnetar_energy, magnetar_timescale):
    xi, xi_dot = _nagy_vinko_reduced_xi_and_xidot(
        t, energy_factor, x_ni, x_co, radius, velocity, t_initial,
        recombination_temperature, p1, p2, p3, p5, t_ni, magnetar_energy,
        magnetar_timescale)
    sigma = (radius + velocity * t) / radius
    if t == 0.0:
        opacity_factor = 1.0
    else:
        opacity_factor = 1.0 - np.exp(-gamma_coeff / (t * t))
    diffusion_luminosity = xi * energy_factor * e_th * opacity_factor / diffusion_time
    photosphere_radius = radius + velocity * t
    recombination_radius = xi * photosphere_radius
    if powerlaw_density == 0.0:
        density_profile = _lc2_eta(xi, exponential_density)
    else:
        density_profile = _lc2_theta(xi, powerlaw_density)
    recombination_luminosity = (
        -4.0 * np.pi * rho0 * density_profile * sigma ** -3.0
        * ionization_energy * recombination_radius ** 2.0
        * photosphere_radius * (-_lc2_series_derivative(xi) * xi * xi_dot))
    if recombination_luminosity < 0.0:
        recombination_luminosity = 0.0
    return diffusion_luminosity + recombination_luminosity


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_component_luminosity_reduced_kernel(
        time_days, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
        thermal_energy, exponential_density, powerlaw_density, kappa, magnetar_energy,
        magnetar_timescale, gamma_leakage, integration_timestep):
    day = 86400.0
    xmin = 0.4
    e_ni = 3.89e10
    e_co = 6.8e9
    c_lc2 = 2.99e10
    m_sun = 1.989e33
    radiation_constant_lc2 = 7.57e-15

    output = np.zeros(len(time_days))
    if len(time_days) == 0:
        return output
    if powerlaw_density == 3.0 or powerlaw_density == 5.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    mass = mej * m_sun
    m_ni = nickel_mass * m_sun
    e_kin = kinetic_energy * 1.0e51
    e_th_input = thermal_energy * 1.0e51
    e_mag = magnetar_energy * 1.0e51
    t_mag = magnetar_timescale * day
    gamma_coeff = gamma_leakage * day * day
    t_ni = 8.8 * day
    t_co = 111.3 * day
    dt_seconds = max(integration_timestep, 1.0)

    t_initial = (e_th_input * np.pi / (4.0 * radius ** 3 * radiation_constant_lc2)) ** 0.25
    if powerlaw_density == 0.0:
        f_mass = _lc2_im_integral(1.0, exponential_density, 2.0)
        g_kinetic = _lc2_im_integral(1.0, exponential_density, 4.0)
    else:
        f_mass = (3.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 3.0) / (
            3.0 * (3.0 - powerlaw_density))
        g_kinetic = (5.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 5.0) / (
            5.0 * (5.0 - powerlaw_density))

    velocity = np.sqrt(2.0 * e_kin * f_mass / (g_kinetic * mass))
    rho0 = mass / (4.0 * np.pi * radius ** 3 * f_mass)
    diffusion_time = 3.0 * kappa * rho0 * radius * radius / (np.pi * np.pi * c_lc2)
    e_th = 4.0 * np.pi * radius ** 3 * radiation_constant_lc2 * t_initial ** 4.0 / (np.pi * np.pi)
    if e_th <= 0.0 or diffusion_time <= 0.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    p1 = e_ni * m_ni * t_ni / e_th
    p2 = t_ni / diffusion_time
    p3 = t_ni / e_th
    p4 = t_ni / t_co
    p5 = e_co / e_ni
    ionization_energy = 1.6e13

    t = 0.0
    energy_factor = 1.0
    x_ni = 1.0
    x_co = 0.0
    target_index = 0
    max_time_seconds = time_days[-1] * day
    previous_time = 0.0
    previous_luminosity = _nagy_vinko_reduced_luminosity(
        t, energy_factor, x_ni, x_co, radius, velocity, rho0, e_th,
        diffusion_time, gamma_coeff, exponential_density, powerlaw_density,
        recombination_temperature, ionization_energy, t_initial, p1, p2, p3, p5,
        t_ni, e_mag, t_mag)

    while target_index < len(time_days) and time_days[target_index] < 0.0:
        output[target_index] = 0.0
        target_index += 1

    while t <= max_time_seconds + dt_seconds and target_index < len(time_days):
        luminosity = _nagy_vinko_reduced_luminosity(
            t, energy_factor, x_ni, x_co, radius, velocity, rho0, e_th,
            diffusion_time, gamma_coeff, exponential_density, powerlaw_density,
            recombination_temperature, ionization_energy, t_initial, p1, p2, p3, p5,
            t_ni, e_mag, t_mag)

        while target_index < len(time_days) and time_days[target_index] * day <= t:
            target_time = time_days[target_index] * day
            if t == previous_time:
                output[target_index] = luminosity
            else:
                weight = (target_time - previous_time) / (t - previous_time)
                output[target_index] = previous_luminosity + weight * (luminosity - previous_luminosity)
            target_index += 1

        k1_f, k1_ni, k1_co = _nagy_vinko_reduced_derivatives(
            t, energy_factor, x_ni, x_co, radius, velocity, t_initial,
            recombination_temperature, p1, p2, p3, p4, p5, t_ni, e_mag, t_mag)
        k2_f, k2_ni, k2_co = _nagy_vinko_reduced_derivatives(
            t + 0.5 * dt_seconds, energy_factor + 0.5 * dt_seconds * k1_f,
            x_ni + 0.5 * dt_seconds * k1_ni, x_co + 0.5 * dt_seconds * k1_co,
            radius, velocity, t_initial, recombination_temperature, p1, p2, p3,
            p4, p5, t_ni, e_mag, t_mag)
        k3_f, k3_ni, k3_co = _nagy_vinko_reduced_derivatives(
            t + 0.5 * dt_seconds, energy_factor + 0.5 * dt_seconds * k2_f,
            x_ni + 0.5 * dt_seconds * k2_ni, x_co + 0.5 * dt_seconds * k2_co,
            radius, velocity, t_initial, recombination_temperature, p1, p2, p3,
            p4, p5, t_ni, e_mag, t_mag)
        k4_f, k4_ni, k4_co = _nagy_vinko_reduced_derivatives(
            t + dt_seconds, energy_factor + dt_seconds * k3_f,
            x_ni + dt_seconds * k3_ni, x_co + dt_seconds * k3_co,
            radius, velocity, t_initial, recombination_temperature, p1, p2, p3,
            p4, p5, t_ni, e_mag, t_mag)

        previous_time = t
        previous_luminosity = luminosity
        energy_factor += dt_seconds * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f) / 6.0
        if energy_factor < 0.0:
            energy_factor = 0.0
        x_ni += dt_seconds * (k1_ni + 2.0 * k2_ni + 2.0 * k3_ni + k4_ni) / 6.0
        x_co += dt_seconds * (k1_co + 2.0 * k2_co + 2.0 * k3_co + k4_co) / 6.0
        t += dt_seconds

    return output


def _lc2_smooth_log_luminosity(luminosity, smoothing_timescale, time_step, percentile):
    if smoothing_timescale <= 0.0 or len(luminosity) < 3:
        return luminosity
    window = int(np.round(smoothing_timescale / time_step))
    if window < 3:
        return luminosity
    if window % 2 == 0:
        window += 1
    percentile = min(100.0, max(0.0, percentile))
    safe_luminosity = np.maximum(luminosity, 1e-300)
    return np.exp(percentile_filter(np.log(safe_luminosity), percentile, size=window, mode="nearest"))


def _nagy_vinko_component_luminosity(
        time, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
        thermal_energy, exponential_density, powerlaw_density, kappa, magnetar_energy,
        magnetar_timescale, gamma_leakage, integration_timestep, reference_mode,
        recombination_max_delta, minimum_timestep, recombination_smoothing_width,
        recombination_fine_xi, output_smoothing_timescale, output_filter_percentile,
        output_timestep, reduced_mode, reduced_timestep):
    time = np.atleast_1d(np.asarray(time, dtype=float))
    output = np.zeros_like(time, dtype=float)
    valid = time >= 0
    if not np.any(valid):
        return output
    if recombination_temperature <= 0.0 and nickel_mass <= 0.0 and magnetar_energy <= 0.0:
        output[valid] = _nagy_vinko_cooling_component_luminosity(
            time[valid], radius, mej, thermal_energy, exponential_density,
            powerlaw_density, kappa, kinetic_energy, gamma_leakage)
        return output
    order = np.argsort(time[valid])
    valid_indices = np.where(valid)[0]
    sorted_indices = valid_indices[order]
    sorted_times = time[sorted_indices]
    if reduced_mode:
        sorted_luminosity = _nagy_vinko_component_luminosity_reduced_kernel(
            sorted_times, radius, mej, recombination_temperature, nickel_mass,
            kinetic_energy, thermal_energy, exponential_density, powerlaw_density,
            kappa, magnetar_energy, magnetar_timescale, gamma_leakage, reduced_timestep)
        output[sorted_indices] = sorted_luminosity
        return output
    if (not reference_mode) and output_smoothing_timescale > 0.0 and sorted_times[-1] > 0.0:
        dense_step = max(output_timestep / day_to_s, 1e-6)
        dense_times = np.arange(0.0, sorted_times[-1] + dense_step, dense_step)
        if dense_times[-1] < sorted_times[-1]:
            dense_times = np.append(dense_times, sorted_times[-1])
        dense_luminosity = _nagy_vinko_component_luminosity_kernel(
            dense_times, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
            thermal_energy, exponential_density, powerlaw_density, kappa, magnetar_energy,
            magnetar_timescale, gamma_leakage, integration_timestep, reference_mode,
            recombination_max_delta, minimum_timestep, recombination_smoothing_width,
            recombination_fine_xi)
        sorted_luminosity = np.interp(
            sorted_times, dense_times,
            _lc2_smooth_log_luminosity(
                dense_luminosity, output_smoothing_timescale / day_to_s, dense_step,
                output_filter_percentile))
    else:
        sorted_luminosity = _nagy_vinko_component_luminosity_kernel(
            sorted_times, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
            thermal_energy, exponential_density, powerlaw_density, kappa, magnetar_energy,
            magnetar_timescale, gamma_leakage, integration_timestep, reference_mode,
            recombination_max_delta, minimum_timestep, recombination_smoothing_width,
            recombination_fine_xi)
    output[sorted_indices] = sorted_luminosity
    return output


@cond_jit(nopython=True, fastmath=True, cache=False)
def _nagy_vinko_cooling_component_luminosity(
        time_days, radius, mej, thermal_energy, exponential_density,
        powerlaw_density, kappa, kinetic_energy, gamma_leakage):
    day = 86400.0
    xmin = 0.4
    c_lc2 = 2.99e10
    m_sun = 1.989e33
    radiation_constant_lc2 = 7.57e-15
    output = np.zeros(len(time_days))
    if powerlaw_density == 3.0 or powerlaw_density == 5.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    mass = mej * m_sun
    e_kin = kinetic_energy * 1.0e51
    e_th_input = thermal_energy * 1.0e51
    if powerlaw_density == 0.0:
        f_mass = _lc2_im_integral(1.0, exponential_density, 2.0)
        g_kinetic = _lc2_im_integral(1.0, exponential_density, 4.0)
    else:
        f_mass = (3.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 3.0) / (
            3.0 * (3.0 - powerlaw_density))
        g_kinetic = (5.0 * xmin ** powerlaw_density - powerlaw_density * xmin ** 5.0) / (
            5.0 * (5.0 - powerlaw_density))

    velocity = np.sqrt(2.0 * e_kin * f_mass / (g_kinetic * mass))
    rho0 = mass / (4.0 * np.pi * radius ** 3 * f_mass)
    diffusion_time = 3.0 * kappa * rho0 * radius * radius / (np.pi * np.pi * c_lc2)
    hydro_time = radius / velocity
    t_initial = (e_th_input * np.pi / (4.0 * radius ** 3 * radiation_constant_lc2)) ** 0.25
    e_th = 4.0 * np.pi * radius ** 3 * radiation_constant_lc2 * t_initial ** 4.0 / (np.pi * np.pi)
    gamma_coeff = gamma_leakage * day * day
    if e_th <= 0.0 or diffusion_time <= 0.0 or hydro_time <= 0.0:
        for ii in range(len(output)):
            output[ii] = np.nan
        return output

    for ii in range(len(time_days)):
        t = time_days[ii] * day
        if t < 0.0:
            output[ii] = 0.0
        else:
            if t == 0.0:
                opacity_factor = 1.0
            else:
                opacity_factor = 1.0 - np.exp(-gamma_coeff / (t * t))
            energy_factor = np.exp(-t / diffusion_time - 0.5 * t * t / (hydro_time * diffusion_time))
            output[ii] = energy_factor * e_th * opacity_factor / diffusion_time
    return output


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1989ApJ...340..396A/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..77N/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2016A%26A...589A..53N/abstract')
def nagy_vinko_component_bolometric(
        time, radius, mej, recombination_temperature, nickel_mass, kinetic_energy,
        thermal_energy, exponential_density, powerlaw_density, kappa, **kwargs):
    """
    Bolometric luminosity for one LC2/Nagy-Vinko supernova component.

    This is a Python/Numba port of the public LC2 C implementation by Nagy & Vinko,
    based on Arnett & Fu (1989). It evolves the dimensionless internal energy and
    recombination front for a single homologously-expanding ejecta component.

    :param time: source-frame time in days
    :param radius: initial radius in cm
    :param mej: ejecta mass in solar masses
    :param recombination_temperature: recombination temperature in K; set to 0 to disable recombination
    :param nickel_mass: initial nickel mass in solar masses
    :param kinetic_energy: kinetic energy in units of 1e51 erg
    :param thermal_energy: initial thermal energy in units of 1e51 erg
    :param exponential_density: exponential density profile exponent; used when powerlaw_density is 0
    :param powerlaw_density: power-law density profile exponent; 0 uses the exponential profile
    :param kappa: Thomson/electron-scattering opacity in cm^2/g
    :param kwargs: magnetar_energy in 1e51 erg, magnetar_timescale in days,
        gamma_leakage in day^2, lc2_timestep in seconds, and
        lc2_reference_mode to reproduce the original C recombination-front stepping.
        Set lc2_mode='reduced' to use the continuous-front reduced solver.
        In the default continuous mode, lc2_timestep is the coarse internal
        step, lc2_minimum_timestep is used once the recombination front reaches
        lc2_recombination_fine_xi, and lc2_recombination_max_delta limits
        recombination-front motion per internal step.
        lc2_output_smoothing_timescale applies a running lower-percentile output
        filter in days to the fast mode only; lc2_reference_mode remains
        unsmoothed.
    :return: bolometric luminosity in erg/s
    """
    return _nagy_vinko_component_luminosity(
        time=time, radius=radius, mej=mej, recombination_temperature=recombination_temperature,
        nickel_mass=nickel_mass, kinetic_energy=kinetic_energy, thermal_energy=thermal_energy,
        exponential_density=exponential_density, powerlaw_density=powerlaw_density, kappa=kappa,
        magnetar_energy=kwargs.get("magnetar_energy", 0.0),
        magnetar_timescale=kwargs.get("magnetar_timescale", 0.0),
        gamma_leakage=kwargs.get("gamma_leakage", 1e10),
        integration_timestep=kwargs.get("lc2_timestep", 300.0),
        reference_mode=kwargs.get("lc2_reference_mode", False),
        recombination_max_delta=kwargs.get("lc2_recombination_max_delta", 1e-4),
        minimum_timestep=kwargs.get("lc2_minimum_timestep", 240.0),
        recombination_smoothing_width=kwargs.get("lc2_recombination_smoothing_width", 0.0),
        recombination_fine_xi=kwargs.get("lc2_recombination_fine_xi", 0.6),
        output_smoothing_timescale=kwargs.get("lc2_output_smoothing_timescale", 0.25) * day_to_s,
        output_filter_percentile=kwargs.get("lc2_output_filter_percentile", 35.0),
        output_timestep=kwargs.get("lc2_output_timestep", 240.0),
        reduced_mode=kwargs.get("lc2_mode", "direct") == "reduced",
        reduced_timestep=kwargs.get("lc2_reduced_timestep", 1800.0))


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1989ApJ...340..396A/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..77N/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2016A%26A...589A..53N/abstract')
def nagy_vinko_bolometric(
        time, core_radius, core_mass, core_recombination_temperature, nickel_mass,
        core_kinetic_energy, core_thermal_energy, core_exponential_density,
        core_powerlaw_density, core_kappa, shell_radius, shell_mass,
        shell_recombination_temperature, shell_kinetic_energy, shell_thermal_energy,
        shell_exponential_density, shell_powerlaw_density, shell_kappa, **kwargs):
    """
    Two-component Nagy & Vinko/LC2-style bolometric light curve.

    The model is the sum of a dense inner core and an extended low-mass shell.
    Radioactive nickel and optional magnetar input are deposited in the core,
    while the shell is evolved as a cooling/recombining component with no nickel
    unless ``shell_nickel_mass`` is passed through kwargs.

    :param time: source-frame time in days
    :param core_radius: initial core radius in cm
    :param core_mass: core ejecta mass in solar masses
    :param core_recombination_temperature: core recombination temperature in K
    :param nickel_mass: core nickel mass in solar masses
    :param core_kinetic_energy: core kinetic energy in units of 1e51 erg
    :param core_thermal_energy: core initial thermal energy in units of 1e51 erg
    :param core_exponential_density: core exponential density exponent
    :param core_powerlaw_density: core power-law density exponent; 0 uses exponential profile
    :param core_kappa: core opacity in cm^2/g
    :param shell_radius: initial shell radius in cm
    :param shell_mass: shell ejecta mass in solar masses
    :param shell_recombination_temperature: shell recombination temperature in K
    :param shell_kinetic_energy: shell kinetic energy in units of 1e51 erg
    :param shell_thermal_energy: shell initial thermal energy in units of 1e51 erg
    :param shell_exponential_density: shell exponential density exponent
    :param shell_powerlaw_density: shell power-law density exponent; 0 uses exponential profile
    :param shell_kappa: shell opacity in cm^2/g
    :param kwargs: core_gamma_leakage, shell_gamma_leakage, magnetar_energy,
        magnetar_timescale, shell_nickel_mass, and lc2_timestep
    :return: bolometric luminosity in erg/s
    """
    timestep = kwargs.get("lc2_timestep", 300.0)
    core_lbol = _nagy_vinko_component_luminosity(
        time=time, radius=core_radius, mej=core_mass,
        recombination_temperature=core_recombination_temperature, nickel_mass=nickel_mass,
        kinetic_energy=core_kinetic_energy, thermal_energy=core_thermal_energy,
        exponential_density=core_exponential_density, powerlaw_density=core_powerlaw_density,
        kappa=core_kappa, magnetar_energy=kwargs.get("magnetar_energy", 0.0),
        magnetar_timescale=kwargs.get("magnetar_timescale", 0.0),
        gamma_leakage=kwargs.get("core_gamma_leakage", kwargs.get("gamma_leakage", 3e5)),
        integration_timestep=timestep, reference_mode=kwargs.get("lc2_reference_mode", False),
        recombination_max_delta=kwargs.get("lc2_recombination_max_delta", 1e-4),
        minimum_timestep=kwargs.get("lc2_minimum_timestep", 240.0),
        recombination_smoothing_width=kwargs.get("lc2_recombination_smoothing_width", 0.0),
        recombination_fine_xi=kwargs.get("lc2_recombination_fine_xi", 0.6),
        output_smoothing_timescale=kwargs.get("lc2_output_smoothing_timescale", 0.25) * day_to_s,
        output_filter_percentile=kwargs.get("lc2_output_filter_percentile", 35.0),
        output_timestep=kwargs.get("lc2_output_timestep", 240.0),
        reduced_mode=kwargs.get("lc2_mode", "direct") == "reduced",
        reduced_timestep=kwargs.get("lc2_reduced_timestep", 1800.0))
    shell_lbol = _nagy_vinko_component_luminosity(
        time=time, radius=shell_radius, mej=shell_mass,
        recombination_temperature=shell_recombination_temperature,
        nickel_mass=kwargs.get("shell_nickel_mass", 0.0),
        kinetic_energy=shell_kinetic_energy, thermal_energy=shell_thermal_energy,
        exponential_density=shell_exponential_density, powerlaw_density=shell_powerlaw_density,
        kappa=shell_kappa, magnetar_energy=0.0, magnetar_timescale=0.0,
        gamma_leakage=kwargs.get("shell_gamma_leakage", 1e10), integration_timestep=timestep,
        reference_mode=kwargs.get("lc2_reference_mode", False),
        recombination_max_delta=kwargs.get("lc2_recombination_max_delta", 1e-4),
        minimum_timestep=kwargs.get("lc2_minimum_timestep", 240.0),
        recombination_smoothing_width=kwargs.get("lc2_recombination_smoothing_width", 0.0),
        recombination_fine_xi=kwargs.get("lc2_recombination_fine_xi", 0.6),
        output_smoothing_timescale=kwargs.get("lc2_output_smoothing_timescale", 0.25) * day_to_s,
        output_filter_percentile=kwargs.get("lc2_output_filter_percentile", 35.0),
        output_timestep=kwargs.get("lc2_output_timestep", 240.0),
        reduced_mode=kwargs.get("lc2_mode", "direct") == "reduced",
        reduced_timestep=kwargs.get("lc2_reduced_timestep", 1800.0))
    return core_lbol + shell_lbol


@citation_wrapper('https://ui.adsabs.harvard.edu/abs/1989ApJ...340..396A/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..77N/abstract, '
                  'https://ui.adsabs.harvard.edu/abs/2016A%26A...589A..53N/abstract')
def nagy_vinko(
        time, redshift, core_radius, core_mass, core_recombination_temperature, nickel_mass,
        core_kinetic_energy, core_thermal_energy, core_exponential_density,
        core_powerlaw_density, core_kappa, shell_radius, shell_mass,
        shell_recombination_temperature, shell_kinetic_energy, shell_thermal_energy,
        shell_exponential_density, shell_powerlaw_density, shell_kappa, temperature_floor, **kwargs):
    """
    Photometric wrapper for the two-component Nagy & Vinko/LC2-style model.

    The LC2 model itself is bolometric. This wrapper converts the bolometric
    luminosity to flux/magnitude/spectra with redback's standard temperature-floor
    blackbody prescription.
    """
    kwargs['photosphere'] = kwargs.get("photosphere", photosphere.TemperatureFloor)
    kwargs['sed'] = kwargs.get("sed", sed.Blackbody)
    cosmology = kwargs.get('cosmology', cosmo)
    dl = cosmology.luminosity_distance(redshift).cgs.value
    shell_velocity = _nagy_vinko_velocity(
        radius=shell_radius, mej=shell_mass, kinetic_energy=shell_kinetic_energy,
        exponential_density=shell_exponential_density, powerlaw_density=shell_powerlaw_density)
    photosphere_velocity = kwargs.get("vej", shell_velocity / km_cgs)

    if kwargs['output_format'] == 'flux_density':
        frequency = kwargs['frequency']
        frequency, time = calc_kcorrected_properties(frequency=frequency, redshift=redshift, time=time)
        lbol = nagy_vinko_bolometric(
            time=time, core_radius=core_radius, core_mass=core_mass,
            core_recombination_temperature=core_recombination_temperature, nickel_mass=nickel_mass,
            core_kinetic_energy=core_kinetic_energy, core_thermal_energy=core_thermal_energy,
            core_exponential_density=core_exponential_density, core_powerlaw_density=core_powerlaw_density,
            core_kappa=core_kappa, shell_radius=shell_radius, shell_mass=shell_mass,
            shell_recombination_temperature=shell_recombination_temperature,
            shell_kinetic_energy=shell_kinetic_energy, shell_thermal_energy=shell_thermal_energy,
            shell_exponential_density=shell_exponential_density, shell_powerlaw_density=shell_powerlaw_density,
            shell_kappa=shell_kappa, **kwargs)
        photo = kwargs['photosphere'](
            time=time, luminosity=lbol, vej=photosphere_velocity, temperature_floor=temperature_floor)
        sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                              frequency=frequency, luminosity_distance=dl)
        return sed_1.flux_density.to(uu.mJy).value * (1 + redshift)

    time_obs = time
    lambda_observer_frame = kwargs.get('lambda_array', np.geomspace(100, 60000, 100))
    max_source_time = np.max(np.atleast_1d(time_obs)) / (1. + redshift)
    stop_time = kwargs.get("stop_time", max(300.0, 1.2 * max_source_time))
    time_temp = get_optimal_time_array(0.1, stop_time, kwargs.get("dense_resolution", 300))
    time_observer_frame = time_temp * (1. + redshift)
    frequency, time = calc_kcorrected_properties(frequency=lambda_to_nu(lambda_observer_frame),
                                                 redshift=redshift, time=time_observer_frame)
    lbol = nagy_vinko_bolometric(
        time=time, core_radius=core_radius, core_mass=core_mass,
        core_recombination_temperature=core_recombination_temperature, nickel_mass=nickel_mass,
        core_kinetic_energy=core_kinetic_energy, core_thermal_energy=core_thermal_energy,
        core_exponential_density=core_exponential_density, core_powerlaw_density=core_powerlaw_density,
        core_kappa=core_kappa, shell_radius=shell_radius, shell_mass=shell_mass,
        shell_recombination_temperature=shell_recombination_temperature,
        shell_kinetic_energy=shell_kinetic_energy, shell_thermal_energy=shell_thermal_energy,
        shell_exponential_density=shell_exponential_density, shell_powerlaw_density=shell_powerlaw_density,
        shell_kappa=shell_kappa, **kwargs)
    photo = kwargs['photosphere'](
        time=time, luminosity=lbol, vej=photosphere_velocity, temperature_floor=temperature_floor)
    sed_1 = kwargs['sed'](temperature=photo.photosphere_temperature, r_photosphere=photo.r_photosphere,
                          frequency=frequency[:, None], luminosity_distance=dl)
    fmjy = sed_1.flux_density.T
    spectra = flux_density_to_spectrum(fmjy, redshift, lambda_observer_frame)
    if kwargs['output_format'] == 'spectra':
        return namedtuple('output', ['time', 'lambdas', 'spectra'])(time=time_observer_frame,
                                                                    lambdas=lambda_observer_frame,
                                                                    spectra=spectra)
    sed_kwargs = kwargs.copy()
    sed_kwargs.pop("lambda_array", None)
    return sed.get_correct_output_format_from_spectra(time=time_obs, time_eval=time_observer_frame,
                                                      spectra=spectra, lambda_array=lambda_observer_frame,
                                                      **sed_kwargs)


def _nagy_vinko_velocity(radius, mej, kinetic_energy, exponential_density, powerlaw_density):
    if powerlaw_density == 0:
        f_mass = _lc2_im_integral(1.0, exponential_density, 2.0)
        g_kinetic = _lc2_im_integral(1.0, exponential_density, 4.0)
    else:
        f_mass = (3.0 * 0.4 ** powerlaw_density - powerlaw_density * 0.4 ** 3.0) / (
            3.0 * (3.0 - powerlaw_density))
        g_kinetic = (5.0 * 0.4 ** powerlaw_density - powerlaw_density * 0.4 ** 5.0) / (
            5.0 * (5.0 - powerlaw_density))
    return np.sqrt(2.0 * kinetic_energy * 1e51 * f_mass / (g_kinetic * mej * solar_mass))
