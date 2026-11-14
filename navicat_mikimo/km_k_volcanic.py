#!/usr/bin/env python
import itertools
import multiprocessing
import sys
import warnings
from typing import List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk
from joblib import Parallel, delayed
from navicat_volcanic.helpers import bround, group_data_points
from navicat_volcanic.plotting2d import (calc_ci, get_reg_targets, plot_2d,
                                         plot_2d_lsfer)
from navicat_volcanic.plotting3d import (get_bases, plot_3d_contour,
                                         plot_3d_contour_regions)
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import run_record
from .helper import (call_imputter, choose_descriptors, column_masks,
                     curate_profiles, energy_columns)
from .kinetic_solver_runner import calc_km
from .plot_function import (plot_2d_combo, plot_3d_contour_regions_np,
                            plot_3d_m, plot_3d_np, plot_evo)

warnings.filterwarnings("ignore")


def process_n_calc_2d(
    profile: np.ndarray,
    n_target: int,
    t_span: Tuple[float, float],
    rxn_network_all: np.ndarray,
    initial_conc: np.ndarray,
    states: List[str],
    report_as_yield: bool,
    quality: int,
    verb: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process the mkm input and perform the simulation with kinetic profile as the input
    in a case of a single descriptor.

    Parameters:
        profile (np.ndarray): Kinetic profile data.
        n_target (int): Number of target products.
        t_span (Tuple[float, float]): Time span for the simulation.
        rxn_network_all (np.ndarray): Array containing the reaction network information.
        initial_conc (np.ndarray): Initial concentrations.
        states (List[str]): Reaction network column names.
        report_as_yield (bool): Flag indicating whether to report the results as yield or concentration.
        quality (int): Quality level of the simulation.
        verb (int): Verbosity level.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Result of the calculation and its uncertainty.
    """
    try:
        if np.any(np.isnan(profile)):
            return np.array([np.nan] * n_target)
        else:
            result, _ = calc_km(
                None,
                None,
                None,
                rxn_network_all,
                None,
                t_span,
                initial_conc,
                states,
                report_as_yield,
                quality,
                profile,
            )
            return result
    except Exception as e:
        if verb > 1:
            print(f"Fail to compute at point {profile} in the volcano line due to {e}.")
        return np.array([np.nan] * n_target)


def process_n_calc_3d(
    coord: Tuple[int, int],
    grids: Tuple[np.ndarray, np.ndarray],
    n_target: int,
    t_span: Tuple[float, float],
    rxn_network_all: np.ndarray,
    initial_conc: np.ndarray,
    states: List[str],
    report_as_yield: bool,
    quality: int,
    verb: int,
) -> np.ndarray:
    """
    Process the mkm input and perform the simulation with kinetic profile as the input
    in a case of a two descriptor.

    Parameters:
        coord (Tuple[int, int]): Coordinate.
        dgs (List[List[float]]): List of kinetic profiles for all coordinates.
        n_target (int): Number of products.
        t_span (Tuple[float, float]): Time span for the simulation.
        rxn_network_all (np.ndarray): Array containing the reaction network information.
        initial_conc (np.ndarray): Initial concentrations.
        states (List[str]): Reaction network column names.
        report_as_yield (bool): Flag indicating whether to report the results as yield or concentration.
        quality (int): Quality of the integration.
        verb (int): Verbosity level.

    Returns:
        np.ndarray: Array of target concentrations or yields.
    """

    try:
        # NOTE note sure how the grid is organized
        profile = np.array([gridj[coord] for gridj in grids])

        result, _ = calc_km(
            None,
            None,
            None,
            rxn_network_all,
            None,
            t_span,
            initial_conc,
            states,
            report_as_yield,
            quality,
            profile,
        )
        return result
    except Exception as e:
        if verb > 1:
            print(f"Fail to compute at point {profile} in the volcano line due to {e}.")
        return np.array([np.nan] * n_target)


def process_n_calc_3d_ps(
    coord: Tuple[int, int],
    dgs: np.ndarray,
    t_points: np.ndarray,
    n_target: int,
    rxn_network_all: np.ndarray,
    initial_conc: np.ndarray,
    states: List[str],
    report_as_yield: bool,
    quality: int,
    verb: str,
) -> np.ndarray:
    """
    Process and calculate the MKM for a single descriptor and one physical variable
    (time or temperature) in the kinetic mode.

    Parameters:
        coord (Tuple[int, int]): Coordinate of the point in the descriptor/physical variable space.
        dgs (np.ndarray): Array of free energy profiles.
        t_points (np.ndarray): Array of temperature points.
        n_target (int): Number of products.
        df_network (pd.DataFrame): Reaction network DataFrame.
        tags (List[str]): Reaction data column names.
        states (List[str]): Reaction network column names.
        report_as_yield (bool): Report results as yield if True.
        quality (int): Integration quality level.
        mode (str): Calculation mode ('vtime' or 'vtemp').
        verb (int): Verbosity level.

    Returns:
        result (np.ndarray): np.ndarray: Array of target concentrations or yields.
    """
    try:
        profile = dgs[coord[0], :]
        t_span = (0, t_points[coord[1]])

        result, _ = calc_km(
            None,
            None,
            None,
            rxn_network_all,
            None,
            t_span,
            initial_conc,
            states,
            report_as_yield,
            quality,
            profile,
        )

        return result

    except Exception as e:
        if verb > 1:
            print(f"Fail to compute at point {profile} in the volcano line due to {e}.")
        return np.array([np.nan] * n_target)


def evol_mode(
    d: np.ndarray,
    df_network: pd.DataFrame,
    names: List[str],
    states: List[str],
    t_span: Tuple[float, float],
    report_as_yield: bool,
    quality: float,
    verb: int,
    n_target: int,
    x_scale: str,
    more_species_mkm: List[str],
) -> None:
    """
    Execute the evolution mode: plotting evolution for all profiles in the reaction data.

    Parameters:
        d (np.ndarray): NumPy array of kinetic profiles.
        df_network (pd.DataFrame): DataFrame containing the reaction network.
        names (List[str]): List of names for the profiles.
        states (List[str]): List of states.
        t_span (Tuple[float, float]): Time span for the simulation.
        report_as_yield (bool): Boolean indicating whether to report as yield.
        quality (float): Quality level of the integration.
        verb (int): Verbosity level.
        n_target (int): Number of targets.
        x_scale (float): Time scale for the x-axis.
        more_species_mkm (List[str]): Additional species to be included in the evolution plot.

    Returns:
        None

    Raises:
        None
    """

    if verb > 0:
        print(
            "\n------------Evol mode: plotting evolution for all profiles------------------\n"
        )

    prod_conc_pt = []
    result_solve_ivp_all = []

    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        if last_row_index.lower() in ["initial_conc", "c0", "initial conc"]:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])
    rxn_network_all = df_network.to_numpy()[:, :]

    per_profile = []
    prod_names = [i.replace("*", "") for i in states if "*" in i]
    for i, profile in enumerate(tqdm(d, total=len(d), ncols=80)):
        status, error = "ok", None
        try:
            result, result_solve_ivp = calc_km(
                None,
                None,
                None,
                rxn_network_all,
                None,
                t_span,
                initial_conc,
                states,
                report_as_yield,
                quality,
                profile,
            )
            if len(result) != n_target:
                prod_conc_pt.append(np.array([np.nan] * n_target))
                status, error = "failed", f"{len(result)} target values for {n_target} targets"
            else:
                prod_conc_pt.append(result)
                if not (result_solve_ivp.success and np.all(np.isfinite(result))):
                    status = "failed"
                    error = (
                        f"non-finite target concentrations ({result_solve_ivp.message})"
                        if not np.all(np.isfinite(result))
                        else f"solver reported failure ({result_solve_ivp.message})"
                    )

            result_solve_ivp_all.append(result_solve_ivp)

            states_ = [s.replace("*", "") for s in states]
            plot_evo(result_solve_ivp, names[i], states_, x_scale, more_species_mkm)

        except Exception as e:
            if len(prod_conc_pt) == i:
                prod_conc_pt.append(np.array([np.nan] * n_target))
                result_solve_ivp_all.append(None)
            status, error = "failed", repr(e)
        if status != "ok":
            print(f"Cannot perform mkm for {names[i]}: {error}")
        per_profile.append(
            {
                "id": str(names[i]),
                "x": None,
                "target_conc": dict(zip(prod_names, prod_conc_pt[i])),
                "status": status,
                "error": error,
            }
        )

    run_record.update("vp", per_profile=per_profile)
    if any(p["status"] != "ok" for p in per_profile):
        run_record.set_status("partial")
        run_record.warn(
            f"MKM failed for {[p['id'] for p in per_profile if p['status'] != 'ok']}"
        )
    prod_conc_pt = np.array(prod_conc_pt).T
    if verb > 1:
        prod_names = [i.replace("*", "") for i in states if "*" in i]
        data_dict = dict()
        data_dict["entry"] = names
        for i in range(prod_conc_pt.shape[0]):
            data_dict[prod_names[i]] = prod_conc_pt[i]

        df_ev = pd.DataFrame(data_dict)
        df_ev.to_csv("prod_conc.csv", index=False)
        print(df_ev.to_string(index=False))

    print("\n")


def get_srps_1d(
    d: np.ndarray,
    tags: List[str],
    coeff: np.ndarray,
    regress: bool,
    lfesrs_idx: Optional[List[int]],
    cb: float,
    ms: float,
    xbase: float,
    lmargin: float,
    rmargin: float,
    npoints: int,
    plotmode: str,
    lfesr: bool,
    verb: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, int]:
    """
    Get the simulated reaction (kinetic) profiles (SRP) in case of a single descriptor
    ,kinetic mode.

    Parameters:
        nd (int): Number of dimensions.
        d (np.ndarray): Kinetic Input data.
        tags (List[str]): List of tags for the data.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        regress (bool): Whether to perform regression.
        lfesrs_idx (Optional[List[int]]): Indices of manually chosen descriptors for LFESR.
        cb (np.ndarray): Array of color values.
        ms (np.ndarray): Array of marker styles.
        lmargin (float): Left margin for descriptor range.
        rmargin (float): Right margin for descriptor range.
        npoints (int): Number of points to compute.
        plotmode (str): Plot mode for LFESRs.
        lfesr (bool): Whether to plot LFESRs.
        verb (int): Verbosity level.

    Returns:
        Tuple containing:
        - dgs (np.ndarray): Log rate constant scaling relationships.
        - d (np.ndarray): Curated Kinetic Input data.
        - sigma_dgs (np.ndarray): Sigma values.
        - xint (np.ndarray): Array of descriptor values.
        - tags (List[str]): Updated list of tags.
        - coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        - idx (int): Index of chosen descriptor.
    """

    idx = choose_descriptors(
        1, d, tags, coeff, regress, lfesrs_idx, offset=1, units="log10 k", verb=verb
    )["idx"]

    if lfesr:
        plot_2d_lsfer(
            idx,
            d,
            tags,
            coeff,
            regress,
            cb,
            ms,
            lmargin,
            rmargin,
            npoints,
            plotmode,
            verb,
        )
        all_lfsers = [s + ".png" for s in tags[1:]]
        lfesr_csv = [s + ".csv" for s in tags[1:]]
        all_lfsers.extend(lfesr_csv)

    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="k")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)

    if verb > 1:
        print(f"Range of descriptor set to [{xmin}, {xmax}].")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci

    return dgs, d, sigma_dgs, X, xint, xmax, xmin, tag, tags, coeff, idx


# NOTE okay for kinetic mode?
def get_srps_2d(
    d: np.ndarray,
    tags: List[str],
    coeff: np.ndarray,
    regress: bool,
    lfesrs_idx: Optional[List[int]],
    lmargin: float,
    rmargin: float,
    npoints: int,
    verb: int,
):
    """
    Get the simulated reaction profiles (SRP) in case of 2 descriptors.

    Parameters:
        d (np.ndarray): Array of the free energy profiles.
        tags (List[str]): Reaction data column names.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        regress (bool): Flag indicating whether to perform regression.
        lfesrs_idx (Optional[List[int]]): List of indices for LFE/SRS.
        lmargin (float): Left margin value.
        rmargin (float): Right margin value.
        npoints (int): Number of points to compute.
        verb (int): Verbosity level.

    Returns:
        d (np.ndarray): Input data array.
        grids (List[np.ndarray]): List of 2D grids.
        xint (np.ndarray): Array of x-axis values (1st descriptor).
        yint (np.ndarray): Array of y-axis values (2nd descriptor).
        X1 (float): X1 value (1st descriptor points).
        X2 (float): X2 value (1st descriptor points).
        x1max (float): Maximum value of X1.
        x2max (float): Maximum value of X2.
        x1min (float): Minimum value of X1.
        x2min (float): Minimum value of X2.
        tag1 (str): Tag for X1 (name of the 1st descriptor).
        tag2 (str): Tag for X2 (name of the 2nd descriptor.
        tags (List[str]): Reaction data column names.
        coeff (np.ndarray): One-hot encoding of state (0=INT, 1=TS).
        idx1 (int): Index for the 1st descriptor.
        idx2 (int): Index for the 2nd descriptor.

    """
    from navicat_volcanic.plotting3d import bround, get_reg_targets

    idx1, idx2 = choose_descriptors(
        2, d, tags, coeff, regress, lfesrs_idx, offset=1, units="log10 k", verb=verb
    )["idx"]
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, regress, mode="k"
    )
    x1base, x2base = get_bases(X1, X2)
    lnsteps = range(d.shape[1])
    x1max = bround(X1.max() + rmargin, x1base, "max")
    x1min = bround(X1.min() - lmargin, x1base, "min")
    x2max = bround(X2.max() + rmargin, x2base, "max")
    x2min = bround(X2.min() - lmargin, x2base, "min")
    if verb > 1:
        print(f"Range of descriptors set to [{x1min}, {x1max}] and [{x2min}, {x2max}].")
    xint = np.linspace(x1min, x1max, npoints)
    yint = np.linspace(x2min, x2max, npoints)
    grids = []
    for i, j in enumerate(lnsteps):
        XY = np.vstack([X1, X2, d[:, j]]).T
        X = XY[:, :2]
        Y = XY[:, 2]
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        gridj = np.zeros((npoints, npoints))
        for k, x1 in enumerate(xint):
            for l, x2 in enumerate(yint):
                x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                gridj[k, l] = reg.predict(x1x2)
        grids.append(gridj)

    return (
        d,
        grids,
        xint,
        yint,
        X1,
        X2,
        x1max,
        x2max,
        x1min,
        x2min,
        tag1,
        tag2,
        tags,
        coeff,
        idx1,
        idx2,
    )


def main(
    df,
    df_network,
    tags,
    states,
    n_target,
    xbase,
    lmargin,
    rmargin,
    verb,
    imputer_strat,
    report_as_yield,
    quality,
    p_quality,
    plotmode,
    more_species_mkm,
    lfesr,
    x_scale,
    comp_ci,
    ncore,
    nd,
    lfesrs_idx,
    times,
    temperatures,
):
    if p_quality == 0:
        interpolate = True
        n_point_calc = 100
        npoints = 150
    elif p_quality == 1:
        interpolate = True
        n_point_calc = 100
        npoints = 200
    elif p_quality == 2:
        interpolate = True
        n_point_calc = 150
        npoints = 250
    elif p_quality == 3:
        interpolate = False
        npoints = 250
    elif p_quality > 3:
        interpolate = False
        npoints = 300

    screen_cond = None
    if times is None:
        t_span = (0, 86400)
    elif len(times) == 1:
        t_span = (0, times[0])
    else:
        screen_cond = "vtime"
        nd = 1
        t_finals_log = np.log10(times)
        x2base = np.round((t_finals_log[1] - t_finals_log[0]) / 10, 1)
        x2min = bround(t_finals_log[0], x2base, "min")
        x2max = bround(t_finals_log[1], x2base, "max")
        t_points = np.logspace(x2min, x2max, npoints)
        if verb > 1:
            print(
                """Building actvity/selectivity map with time as the second variable."""
            )

    if temperatures is None:
        temperature = 298.15
    elif len(temperatures) == 1:
        temperature = temperatures[0]
    else:
        sys.exit("Cannot screen over a range of temperature in the kinetic mode.")

    if ncore == -1:
        ncore = multiprocessing.cpu_count()
    if verb > 2:
        print(f"Use {ncore} cores for parallel computing.")

    if plotmode == 0 and comp_ci:
        plotmode = 1

    names = df[df.columns[0]].values
    cb, ms = group_data_points(0, 2, names)
    # Rate constants are log10 k, read by name (kf_i / k_i forward, kr_i / k_-i reverse, i = network
    # step) by the reaction_profile/v1 check. They are put in step order, forward then reverse, before
    # any other column (a Descriptor column is never a rate constant).
    kin = df.attrs["kinetic"]
    rate_cols = list(kin["forward"]) + list(kin["reverse"])
    order = rate_cols + [c for c in df.columns[1:] if c not in rate_cols]
    if list(df.columns[1:]) != order and verb > 0:
        print("Rate-constant columns put in step order (forward, then reverse); other columns after them.")
    n_rate = len(rate_cols)
    tags, roles = energy_columns(order, states)
    d = df[order].to_numpy().astype(np.float64)
    # volcanic's descriptor search takes the reference state as the first column. A kinetic table has
    # none, so a zero placeholder column leads, and a descriptor index (-id) counts the table's own columns.
    tags = np.insert(tags, 0, "reference")
    roles = ["REFERENCE"] + roles
    d = np.insert(d, 0, 0.0, axis=1)
    coeff, regress = column_masks(roles)
    if verb > 0:
        for tag, role in zip(tags[1:], roles[1:]):
            print(f"Assuming field {tag} is {role} (rate constants are log10 k).")

    d, cb, ms, names, rows = curate_profiles(
        d, regress, cb, ms, tags, names, imputer_strat, nstds=3, verb=verb
    )
    run_record.update(
        "vp",
        nd=nd,
        rows={k: rows[k] for k in ("n_in", "n_used", "dropped")},
        imputer=imputer_strat,
        imputed_cells=rows["imputed_cells"],
        descriptor=None,
        grid_file=None,
        target_units="% yield" if report_as_yield else "M",
        kinetic_columns={"forward": [str(c) for c in kin["forward"]],
                         "reverse": [str(c) for c in kin["reverse"]],
                         "descriptors": [str(c) for c in kin.get("descriptors", [])]},
    )
    # %% selecting modes----------------------------------------------------------#
    if nd == 0:
        d_actual = 10 ** d[:, 1 : 1 + n_rate]
        evol_mode(
            d_actual,
            df_network,
            names,
            states,
            t_span,
            report_as_yield,
            quality,
            verb,
            n_target,
            x_scale,
            more_species_mkm,
        )

    elif nd == 1:
        dgs, d, sigma_dgs, X, xint, xmax, xmin, tag, tags, coeff, idx = get_srps_1d(
            d,
            tags,
            coeff,
            regress,
            lfesrs_idx,
            cb,
            ms,
            xbase,
            lmargin,
            rmargin,
            npoints,
            plotmode,
            lfesr,
            verb,
        )

        run_record.update(
            "vp",
            per_profile=[
                {"id": str(names[i]), "x": float(X[i]), "target_conc": None, "status": None, "error": None}
                for i in range(len(X))
            ],
            grid_file="mkm_descr_phys.h5" if screen_cond else ("mkm_vp.h5" if verb > 1 else None),
        )
        initial_conc = np.array([])
        last_row_index = df_network.index[-1]
        if isinstance(last_row_index, str):
            if last_row_index.lower() in ["initial_conc", "c0", "initial conc"]:
                initial_conc = df_network.iloc[-1:].to_numpy()[0]
                df_network = df_network.drop(df_network.index[-1])
        rxn_network_all = df_network.to_numpy()[:, :]

        if screen_cond:
            dgs = 10**dgs
            if verb > 0:
                print(
                    "\n------------Constructing physical-catalyst activity/selectivity map------------------\n"
                )
            gridj = np.zeros((npoints, npoints))
            grid = np.zeros_like(gridj)
            grid_d = np.array([grid] * n_target)
            total_combinations = len(xint) * len(t_points)
            combinations = list(
                itertools.product(range(len(xint)), range(len(t_points)))
            )
            num_chunks = total_combinations // ncore + (total_combinations % ncore > 0)

            # MKM
            for chunk_index in tqdm(range(num_chunks)):
                start_index = chunk_index * ncore
                end_index = min(start_index + ncore, total_combinations)
                chunk = combinations[start_index:end_index]

                results = Parallel(n_jobs=ncore)(
                    delayed(process_n_calc_3d_ps)(
                        coord,
                        dgs,
                        t_points,
                        n_target,
                        rxn_network_all,
                        initial_conc,
                        states,
                        report_as_yield,
                        quality,
                        verb,
                    )
                    for coord in chunk
                )
                i = 0
                for k, l in chunk:
                    for j in range(n_target):
                        grid_d[j][k, l] = results[i][j]
                    i += 1

            if np.any(np.isnan(grid_d)):
                grid_d_fill = np.zeros_like(grid_d)
                for i, gridi in enumerate(grid_d):
                    impuuter = call_imputter(imputer_strat)
                    impuuter.fit(gridi)
                    filled_data = impuuter.transform(gridi)
                    grid_d_fill[i] = filled_data
            else:
                grid_d_fill = grid_d

            x1base = np.round((xint.max() - xint.min()) / 10)
            if x1base == 0:
                x1base = 1
            x1label = "log$_{10}$" + f"({tag})"
            t_points = np.log10(t_points)
            x2label = "log$_{10}$(time) [s]"
            x2base = np.round((t_points[-1] - t_points[0]) / 10, 1)
            if x2base == 0:
                x2base = 0.5

            with h5py.File("mkm_descr_phys.h5", "w") as f:
                group = f.create_group("data")
                # save each numpy array as a dataset in the group
                group.create_dataset("xint", data=xint)
                group.create_dataset("t_points", data=t_points)
                group.create_dataset("grid", data=grid_d)
                group.create_dataset("tag", data=[tag.encode()])
                group.create_dataset("x1label", data=[x1label.encode()])
                group.create_dataset("x2label", data=[x2label.encode()])

            alabel = "Total product concentration [M]"
            afilename = f"activity_{tag}_{screen_cond}.png"
            activity_grid = np.sum(grid_d_fill, axis=0)
            amin = activity_grid.min()
            amax = activity_grid.max()
            if verb > 2:
                with h5py.File("mkm_descr_phys_activity.h5", "w") as f:
                    group = f.create_group("data")
                    # save each numpy array as a dataset in the group
                    group.create_dataset("xint", data=xint)
                    group.create_dataset("yint", data=t_points)
                    group.create_dataset("agrid", data=activity_grid)
                    group.create_dataset("tag", data=[tag.encode()])
                    group.create_dataset("x1label", data=[x1label.encode()])
                    group.create_dataset("x2label", data=[x2label.encode()])
            plot_3d_np(
                xint,
                t_points,
                activity_grid.T,
                amin,
                amax,
                xint.min(),
                xint.max(),
                t_points.min(),
                t_points.max(),
                x1base,
                x2base,
                x1label,
                x2label,
                ylabel=alabel,
                filename=afilename,
            )
            prod = [p for p in states if "*" in p]
            prod = [s.replace("*", "") for s in prod]
            if n_target == 2:
                sfilename = f"selectivity_{tag}_{screen_cond}.png"
                slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
                min_ratio = -3
                max_ratio = 3
                selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
                selectivity_ratio_ = np.clip(selectivity_ratio, min_ratio, max_ratio)
                selectivity_ratio_ = np.nan_to_num(
                    selectivity_ratio_, nan=-3, posinf=3, neginf=-3
                )
                smin = selectivity_ratio_.min()
                smax = selectivity_ratio_.max()
                if verb > 2:
                    with h5py.File("mkm_descr_phys_selectivity.h5", "w") as f:
                        group = f.create_group("data")
                        # save each numpy array as a dataset in the group
                        group.create_dataset("xint", data=xint)
                        group.create_dataset("yint", data=t_points)
                        group.create_dataset("sgrid", data=selectivity_ratio_)
                        group.create_dataset("tag", data=[tag.encode()])
                        group.create_dataset("x1label", data=[x1label.encode()])
                        group.create_dataset("x2label", data=[x2label.encode()])
                plot_3d_np(
                    xint,
                    t_points,
                    selectivity_ratio_.T,
                    smin,
                    smax,
                    xint.min(),
                    xint.max(),
                    t_points.min(),
                    t_points.max(),
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel=slabel,
                    filename=sfilename,
                )

            elif n_target > 2:
                sfilename = f"selectivity_{tag}_{screen_cond}.png"
                dominant_indices = np.argmax(grid_d_fill, axis=0)
                if verb > 2:
                    with h5py.File("mkm_descr_phys_activity.h5", "w") as f:
                        group = f.create_group("data")
                        # save each numpy array as a dataset in the group
                        group.create_dataset("xint", data=xint)
                        group.create_dataset("yint", data=t_points)
                        group.create_dataset("sgrid", data=dominant_indices)
                        group.create_dataset("tag", data=[tag.encode()])
                        group.create_dataset("x1label", data=[x1label.encode()])
                        group.create_dataset("x2label", data=[x2label.encode()])
                plot_3d_contour_regions_np(
                    xint,
                    t_points,
                    dominant_indices.T,
                    xint.min(),
                    xint.max(),
                    t_points.min(),
                    t_points.max(),
                    x1base,
                    x2base,
                    x1label=x1label,
                    x2label=x2label,
                    ylabel="Dominant product",
                    filename=sfilename,
                    id_labels=prod,
                    nunique=len(prod),
                )

        else:
            dgs = 10**dgs
            d = 10**d

            if verb > 0:
                print("\n------------Constructing MKM volcano plot------------------\n")

            # Volcano line
            if interpolate:
                if verb > 0:
                    print(
                        f"Performing microkinetics modelling for the volcano line ({n_point_calc} points)."
                    )
                selected_indices = np.round(
                    np.linspace(0, len(dgs) - 1, n_point_calc)
                ).astype(int)
                trun_dgs = []

                for i, dg in enumerate(dgs):
                    if i not in selected_indices:
                        trun_dgs.append([np.nan] * len(dgs[0]))
                    else:
                        trun_dgs.append(dg)

            else:
                trun_dgs = dgs
                if verb > 0:
                    print(
                        f"Performing microkinetics modelling for the volcano line ({npoints})."
                    )
            prod_conc = np.zeros((len(dgs), n_target))

            dgs_g = np.array_split(trun_dgs, len(trun_dgs) // ncore + 1)

            i = 0
            for batch_dgs in tqdm(dgs_g, total=len(dgs_g), ncols=80):
                results = Parallel(n_jobs=ncore)(
                    delayed(process_n_calc_2d)(
                        profile,
                        n_target,
                        t_span,
                        rxn_network_all,
                        initial_conc,
                        states,
                        report_as_yield,
                        quality,
                        verb,
                    )
                    for profile in batch_dgs
                )
                for j, res in enumerate(results):
                    prod_conc[i, :] = res
                    i += 1
            # interpolation
            prod_conc_ = prod_conc.copy()
            missing_indices = np.isnan(prod_conc[:, 0])
            for i in range(n_target):
                f = interp1d(
                    xint[~missing_indices],
                    prod_conc[:, i][~missing_indices],
                    kind="cubic",
                    fill_value="extrapolate",
                )
                y_interp = f(xint[missing_indices])
                prod_conc_[:, i][missing_indices] = y_interp

            prod_conc_ = prod_conc_.T
            # Volcano points
            print(
                f"Performing microkinetics modelling for the volcano line ({len(d)})."
            )

            prod_conc_pt = np.zeros((len(d), n_target))

            d_g = np.array_split(d, len(d) // ncore + 1)
            i = 0
            for batch_dgs in tqdm(d_g, total=len(d_g), ncols=80):
                results = Parallel(n_jobs=ncore)(
                    delayed(process_n_calc_2d)(
                        profile,
                        n_target,
                        t_span,
                        rxn_network_all,
                        initial_conc,
                        states,
                        report_as_yield,
                        quality,
                        verb,
                    )
                    for profile in batch_dgs
                )
                for j, res in enumerate(results):
                    prod_conc_pt[i, :] = res
                    i += 1

            # interpolation
            missing_indices = np.isnan(prod_conc_pt[:, 0])
            prod_conc_pt_ = prod_conc_pt.copy()
            for i in range(n_target):
                if np.any(np.isnan(prod_conc_pt)):
                    f = interp1d(
                        X[~missing_indices],
                        prod_conc_pt[:, i][~missing_indices],
                        kind="cubic",
                        fill_value="extrapolate",
                    )
                    y_interp = f(X[missing_indices])
                    prod_conc_pt_[:, i][missing_indices] = y_interp
                else:
                    prod_conc_pt_ = prod_conc_pt.copy()

            prod_conc_pt_ = prod_conc_pt_.T

            # Plotting
            xlabel = "log$_{10}$" + f"({tag})"
            ylabel = "Final product concentraion (M)"

            if report_as_yield:
                ybase = np.round((np.max(prod_conc_pt_) - 0) / 8)
                if ybase == 0:
                    ybase = 5
                ylabel = "%yield"
            else:
                ybase = np.round((np.max(prod_conc_pt_) - 0) / 8, 1)
                if ybase == 0:
                    ybase = 0.05
                ylabel = "Final product concentraion (M)"
            xbase = np.round((np.max(xint) - np.min(xint)) / 8)
            if xbase == 0:
                xbase = 5

            ci_ = np.full(prod_conc_.shape[0], None)
            prod_names = [i.replace("*", "") for i in states if "*" in i]
            if prod_conc_.shape[0] > 1:
                plot_2d_combo(
                    xint,
                    prod_conc_,
                    X,
                    prod_conc_pt_,
                    ci=ci_,
                    ms=ms,
                    xmin=xmin,
                    xmax=xmax,
                    xbase=xbase,
                    ybase=ybase,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    filename=f"mkm_volcano_{tag}_combo.png",
                    plotmode=plotmode,
                    labels=prod_names,
                )
                for i in range(prod_conc_.shape[0]):
                    plot_2d(
                        xint,
                        prod_conc_[i],
                        X,
                        prod_conc_pt_[i],
                        ci=ci_[i],
                        xmin=xmin,
                        xmax=xmax,
                        xbase=xbase,
                        ybase=ybase,
                        cb=cb,
                        ms=ms,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        filename=f"mkm_volcano_{tag}_profile{i}.png",
                        plotmode=plotmode,
                    )
                    plt.clf()
            else:
                plot_2d(
                    xint,
                    prod_conc_[0],
                    X,
                    prod_conc_pt_[0],
                    ci=ci_[0],
                    xmin=xmin,
                    xmax=xmax,
                    xbase=xbase,
                    ybase=ybase,
                    cb=cb,
                    ms=ms,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    filename=f"mkm_volcano_{tag}.png",
                    plotmode=plotmode,
                )

            if verb > 1:
                cb = np.array(cb, dtype="S")
                ms = np.array(ms, dtype="S")
                with h5py.File("mkm_vp.h5", "w") as f:
                    group = f.create_group("data")
                    # save each numpy array as a dataset in the group
                    group.create_dataset("descr_all", data=xint)
                    group.create_dataset("prod_conc_", data=prod_conc_)
                    group.create_dataset("descrp_pt", data=X)
                    group.create_dataset("prod_conc_pt_", data=prod_conc_pt_)
                    group.create_dataset("cb", data=cb)
                    group.create_dataset("ms", data=ms)
                    group.create_dataset("tag", data=[tag.encode()])
                    group.create_dataset("xlabel", data=[xlabel.encode()])
                    group.create_dataset("ylabel", data=[ylabel.encode()])
                    group.create_dataset("labels", data=prod_names)

            print("\n")

    elif nd == 2:
        (
            d,
            grids,
            xint,
            yint,
            X1,
            X2,
            x1max,
            x2max,
            x1min,
            x2min,
            tag1,
            tag2,
            tags,
            coeff,
            idx1,
            idx2,
        ) = get_srps_2d(
            d, tags, coeff, regress, lfesrs_idx, lmargin, rmargin, npoints, verb
        )
        if verb > 0:
            print(
                "\n------------Constructing MKM activity/selectivity map------------------\n"
            )
        grid = np.zeros((npoints, npoints))
        grid_d = np.array([grid] * n_target)
        total_combinations = len(xint) * len(yint)
        combinations = list(itertools.product(range(len(xint)), range(len(yint))))
        num_chunks = total_combinations // ncore + (total_combinations % ncore > 0)
        grids = [10**g for g in grids]

        initial_conc = np.array([])
        last_row_index = df_network.index[-1]
        if isinstance(last_row_index, str):
            if last_row_index.lower() in ["initial_conc", "c0", "initial conc"]:
                initial_conc = df_network.iloc[-1:].to_numpy()[0]
                df_network = df_network.drop(df_network.index[-1])
        rxn_network_all = df_network.to_numpy()[:, :]
        # MKM
        for chunk_index in tqdm(range(num_chunks)):
            start_index = chunk_index * ncore
            end_index = min(start_index + ncore, total_combinations)
            chunk = combinations[start_index:end_index]

            results = Parallel(n_jobs=ncore)(
                delayed(process_n_calc_3d)(
                    coord,
                    grids,
                    n_target,
                    t_span,
                    rxn_network_all,
                    initial_conc,
                    states,
                    report_as_yield,
                    quality,
                    verb,
                )
                for coord in chunk
            )
            i = 0
            for k, l in chunk:
                for j in range(n_target):
                    grid_d[j][k, l] = results[i][j]
                i += 1

        if np.any(np.isnan(grid_d)):
            grid_d_fill = np.zeros_like(grid_d)
            for i, gridi in enumerate(grid_d):
                impuuter = call_imputter(imputer_strat)
                impuuter.fit(gridi)
                filled_data = impuuter.transform(gridi)
                grid_d_fill[i] = filled_data
        else:
            grid_d_fill = grid_d

        px = np.zeros_like(d[:, 0])
        py = np.zeros_like(d[:, 0])
        for i in range(d.shape[0]):
            px[i] = X1[i]
            py[i] = X2[i]

        # Plotting and saving
        # bytes copies for the HDF5 files; the plots need the colour/marker strings
        cb_h5 = np.array(cb, dtype="S")
        ms_h5 = np.array(ms, dtype="S")
        run_record.update(
            "vp",
            grid_file="mkm_vp_3d.h5",
            per_profile=[
                {"id": str(names[i]), "x": [float(px[i]), float(py[i])], "target_conc": None,
                 "status": None, "error": None}
                for i in range(len(px))
            ],
        )
        with h5py.File("mkm_vp_3d.h5", "w") as f:
            group = f.create_group("data")
            # save each numpy array as a dataset in the group
            group.create_dataset("xint", data=xint)
            group.create_dataset("yint", data=yint)
            group.create_dataset("grids", data=grid_d_fill)
            group.create_dataset("px", data=px)
            group.create_dataset("py", data=py)
            group.create_dataset("cb", data=cb_h5)
            group.create_dataset("ms", data=ms_h5)
            group.create_dataset("tag1", data=[tag1.encode()])
            group.create_dataset("tag2", data=[tag2.encode()])

        x1label = "log$_{10}$" + f"({tag1})"
        x2label = "log$_{10}$" + f"({tag2})"
        x1base = np.round((x1max - x1min) / 10, 1)
        if x1base == 0:
            x1base = 0.5
        x2base = np.round((x2max - x2min) / 10, 1)
        if x2base == 0:
            x2base = 0.5

        # activity map
        alabel = "Total product concentration [M]"
        afilename = f"activity_{tag1}_{tag2}.png"
        activity_grid = np.sum(grid_d_fill, axis=0)
        amin = activity_grid.min()
        amax = activity_grid.max()
        plot_3d_m(
            xint,
            yint,
            activity_grid.T,
            px,
            py,
            amin,
            amax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=alabel,
            filename=afilename,
            cb=cb,
            ms=ms,
        )
        if verb > 2:
            with h5py.File("mkm_vp_3d_activity.h5", "w") as f:
                group = f.create_group("data")
                # save each numpy array as a dataset in the group
                group.create_dataset("xint", data=xint)
                group.create_dataset("yint", data=yint)
                group.create_dataset("agrid", data=activity_grid)
                group.create_dataset("px", data=px)
                group.create_dataset("py", data=py)
                group.create_dataset("cb", data=cb_h5)
                group.create_dataset("ms", data=ms_h5)
                group.create_dataset("tag1", data=[tag1.encode()])
                group.create_dataset("tag2", data=[tag2.encode()])
                group.create_dataset("x1label", data=[x1label.encode()])
                group.create_dataset("x2label", data=[x2label.encode()])

        # selectivity map
        prod = [p for p in states if "*" in p]
        prod = [s.replace("*", "") for s in prod]
        if n_target == 2:
            slabel = "$log_{10}$" + f"({prod[0]}/{prod[1]})"
            sfilename = f"selectivity_{tag1}_{tag2}.png"
            min_ratio = -3
            max_ratio = 3
            selectivity_ratio = np.log10(grid_d_fill[0] / grid_d_fill[1])
            selectivity_ratio_ = np.clip(selectivity_ratio, min_ratio, max_ratio)
            selectivity_ratio_ = np.nan_to_num(
                selectivity_ratio_, nan=-3, posinf=3, neginf=-3
            )
            smin = selectivity_ratio.min()
            smax = selectivity_ratio.max()
            if verb > 2:
                with h5py.File("mkm_vp_3d_selectivity.h5", "w") as f:
                    group = f.create_group("data")
                    # save each numpy array as a dataset in the group
                    group.create_dataset("xint", data=xint)
                    group.create_dataset("yint", data=yint)
                    group.create_dataset("sgrid", data=selectivity_ratio_)
                    group.create_dataset("px", data=px)
                    group.create_dataset("py", data=py)
                    group.create_dataset("cb", data=cb_h5)
                    group.create_dataset("ms", data=ms_h5)
                    group.create_dataset("tag1", data=[tag1.encode()])
                    group.create_dataset("tag2", data=[tag2.encode()])
                    group.create_dataset("x1label", data=[x1label.encode()])
                    group.create_dataset("x2label", data=[x2label.encode()])

            plot_3d_contour(
                xint,
                yint,
                selectivity_ratio_.T,
                px,
                py,
                smin,
                smax,
                x1min,
                x1max,
                x2min,
                x2max,
                x1base,
                x2base,
                x1label=x1label,
                x2label=x2label,
                ylabel=slabel,
                filename=sfilename,
                cb=cb,
                ms=ms,
                plotmode=plotmode,
            )

        elif n_target > 2:
            dominant_indices = np.argmax(grid_d_fill, axis=0)
            slabel = "Dominant product"
            sfilename = f"selectivity_{tag1}_{tag2}.png"
            if verb > 2:
                with h5py.File("mkm_vp_3d_selectivity.h5", "w") as f:
                    group = f.create_group("data")
                    # save each numpy array as a dataset in the group
                    group.create_dataset("xint", data=xint)
                    group.create_dataset("yint", data=yint)
                    group.create_dataset("dominant_indices", data=dominant_indices)
                    group.create_dataset("px", data=px)
                    group.create_dataset("py", data=py)
                    group.create_dataset("cb", data=cb_h5)
                    group.create_dataset("ms", data=ms_h5)
                    group.create_dataset("tag1", data=[tag1.encode()])
                    group.create_dataset("tag2", data=[tag2.encode()])
                    group.create_dataset("x1label", data=[x1label.encode()])
                    group.create_dataset("x2label", data=[x2label.encode()])

            plot_3d_contour_regions(
                xint,
                yint,
                dominant_indices.T,
                px,
                py,
                amin,
                amax,
                x1min,
                x1max,
                x2min,
                x2max,
                x1base,
                x2base,
                x1label=x1label,
                x2label=x2label,
                ylabel=slabel,
                filename=sfilename,
                cb=cb,
                ms=ms,
                id_labels=prod,
                plotmode=plotmode,
            )
        print("\n")


# TODO test functions
