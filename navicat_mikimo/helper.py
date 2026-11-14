import argparse
import json
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import autograd.numpy as np
import numpy as onp
import pandas as pd
from navicat_volcanic import profile as vprofile
from navicat_volcanic.helpers import _strip_descriptor

from navicat_mikimo import run_record
from navicat_mikimo.console import configure_console, item, section
from navicat_mikimo.io_config import build_config_from_namespace

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

INITIAL_CONC_LABELS = ["initial_conc", "c0", "initial conc"]


def find_input(wdir, stem, required=True):
    """Path of `<stem>.csv` or `<stem>.xlsx` in wdir. Both present is refused (no silent
    preference); none present exits with a message naming the files looked for."""
    found = [Path(wdir, stem + ext) for ext in (".csv", ".xlsx") if Path(wdir, stem + ext).is_file()]
    if len(found) > 1:
        sys.exit(
            f"Both {found[0].name} and {found[1].name} are in {wdir}; keep only one of them."
        )
    if not found:
        if required:
            sys.exit(f"Neither {stem}.csv nor {stem}.xlsx was found in {wdir}.")
        return None
    return found[0]


def read_table(path, **kwargs):
    """Read a csv or xlsx table (first sheet) with pandas. Headers are trimmed, whitespace-only cells are
    empty cells and fully empty rows are skipped, as the reaction_profile/v1 contract reads the same table."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, **kwargs)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    df.columns = [str(c).strip() for c in df.columns]
    obj = df.select_dtypes(include="object").columns
    for c in obj:
        df[c] = df[c].map(lambda v: np.nan if isinstance(v, str) and not v.strip() else v)
    return df.dropna(how="all")


def read_network(wdir):
    """rxn_network.csv or rxn_network.xlsx, steps as rows, the last row the initial
    concentrations; empty cells become 0."""
    path = find_input(wdir, "rxn_network")
    df_network = read_table(path, index_col=0)
    df_network.fillna(0, inplace=True)
    run_record.add_input_file(path, "rxn_network")
    return df_network


def read_profiles(wdir, kinetic_mode):
    """kinetic_data (with -k) or reaction_data, csv or xlsx."""
    stem = "kinetic_data" if kinetic_mode else "reaction_data"
    path = find_input(wdir, stem)
    df = read_table(path).reset_index(drop=True)
    run_record.add_input_file(path, stem)
    return df


def check_profile_choice(profile_choice, df, what):
    """`-e N` must index an existing row (0 <= N < number of rows; negative counts from the end)."""
    n = df.shape[0]
    if not (-n <= profile_choice < n):
        sys.exit(
            f"The selected {what} profile (-e {profile_choice}) is out of range: the table has {n} row(s)."
        )


class SpeciesList(list):
    """The species names of rxn_network (a plain list for every other use) with the roles that were
    resolved for them and for the reaction table's columns. It travels with the names through the
    solver and plotting calls, so each of them reads the same roles."""

    species_roles = None  # {species: role} as the reaction_profile/v1 check resolved them
    declared_roles = None  # {species: role} declared under "species" in a roles file
    column_roles = None  # {reaction table column: INT, TS, END or DESCRIPTOR}


def _declared_species(report):
    """The species roles a roles file declares (the roles file the reaction_profile/v1 check used)."""
    src = (report or {}).get("roles_source")
    if not isinstance(src, str) or not os.path.isfile(src):
        return {}
    try:
        with open(src, encoding="utf-8-sig") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return {}
    sp = doc.get("species") if isinstance(doc, dict) else None
    if not isinstance(sp, dict):
        return {}
    return {str(k).strip(): str(v).strip().lower() for k, v in sp.items()}


def species_list(names, report=None):
    """names as a SpeciesList carrying the roles of a reaction_profile/v1 report."""
    out = SpeciesList(names)
    net = (report or {}).get("network") or {}
    out.species_roles = {s["name"]: s["role"] for s in net.get("species", []) if s.get("role")}
    out.declared_roles = _declared_species(report)
    out.column_roles = dict((report or {}).get("role_map") or {})
    return out


def column_roles_of(states):
    return getattr(states, "column_roles", None) or {}


def column_role(states, column):
    """Role of a reaction table column: the resolved one, else the shared name rule."""
    return column_roles_of(states).get(str(column).strip()) or vprofile.role_of(column)


def species_is_product(states, name):
    """True for a species the roles mark as a product or target; without roles, the shared end-column name rule."""
    role = (getattr(states, "species_roles", None) or {}).get(name)
    if role is not None:
        return role in ("product", "target")
    return vprofile.role_of(name) == "END"


def energy_columns(columns, states):
    """Tags and roles of the energy table's columns (after the id column). A descriptor column loses
    the word "Descriptor" in its tag, as volcanic names it."""
    tags = onp.array([str(c).strip() for c in columns], dtype=object)
    roles = [column_role(states, t) for t in tags]
    for i, (t, r) in enumerate(zip(tags, roles)):
        if r == "DESCRIPTOR":
            tags[i] = _strip_descriptor(t) or t
            if getattr(states, "column_roles", None) is not None:
                states.column_roles[str(tags[i])] = "DESCRIPTOR"
    return tags, roles


def column_masks(roles):
    """(coeff, regress) of volcanic's descriptor search from column roles: coeff marks transition
    states; regress marks every column that is regressed (everything but the descriptors and
    the placeholder column of a kinetic table)."""
    roles = onp.asarray(roles, dtype=object)
    coeff = roles == "TS"
    regress = onp.isin(roles, ("INT", "TS", "END", "KF", "KR"))
    return coeff, regress


def network_steps(df_network):
    """Number of elementary steps (rows without the initial-concentration row)."""
    last = df_network.index[-1]
    n = df_network.shape[0]
    if isinstance(last, str) and last.lower() in INITIAL_CONC_LABELS:
        n -= 1
    return n


def kinetic_rates(df, profile_choice, kinetic):
    """log10 rate constants of one row of kinetic_data: forward (steps 1..n) then reverse, read by name
    (`kinetic` is the report["kinetic"] section of the reaction_profile/v1 check)."""
    row = df.iloc[profile_choice]
    return row[list(kinetic["forward"]) + list(kinetic["reverse"])].to_numpy().astype(np.float64)


def kinetic_record(kinetic):
    return {
        "forward": [str(c) for c in kinetic["forward"]],
        "reverse": [str(c) for c in kinetic["reverse"]],
        "descriptors": [str(c) for c in kinetic.get("descriptors", [])],
    }


def species_roles(states):
    """Role of each network species as reported in mikimo_result.json (index 0 is the catalyst)."""
    known = getattr(states, "species_roles", None) or {}
    return [known.get(str(s).strip(), "intermediate") for s in states]


def record_species(df_network, states, c_final=None):
    """Species list for mikimo_result.json; c_final is filled by the mode that knows it."""
    last = df_network.index[-1]
    c0 = None
    if isinstance(last, str) and last.lower() in INITIAL_CONC_LABELS:
        c0 = df_network.iloc[-1].to_numpy().astype(float)
    roles = species_roles(states)
    out = []
    for i, s in enumerate(states):
        out.append(
            {
                "name": str(s).replace("*", "").strip(),
                "column": str(s),
                "role": roles[i],
                "c0": None if c0 is None else float(c0[i]),
                "c_final": None if c_final is None else float(c_final[i]),
            }
        )
    steps = df_network.iloc[: network_steps(df_network)]
    unused = [str(s) for s in states if not (steps[s] != 0).any()]
    if unused:
        run_record.warn(f"network species {unused} take part in no step")
    run_record.put("species", out)
    return out


def curate_profiles(d, regress, cb, ms, tags, names, imputer_strat, nstds=3, verb=0):
    """volcanic's curate_d (outlier filter, imputation, incomplete rows dropped) plus a report:
    the names of the rows kept (in order), the rows dropped, and every cell whose value the
    curation changed (imputed, or an outlier replaced). The numbers are those of curate_d."""
    from navicat_volcanic.dv1 import curate_d

    before = np.array(d, dtype=float)
    row_id = np.arange(d.shape[0], dtype=d.dtype)
    out, cb, ms = curate_d(
        np.column_stack([d, row_id]),
        np.append(regress, False),
        cb,
        ms,
        np.append(np.asarray(tags, dtype=object), "row"),
        imputer_strat,
        nstds=nstds,
        verb=verb,
    )
    kept = np.rint(np.asarray(out[:, -1], dtype=float)).astype(int)
    d = np.ascontiguousarray(out[:, :-1])
    names = np.asarray(names)
    names_kept = names[kept]
    dropped = [str(names[i]) for i in range(before.shape[0]) if i not in set(kept.tolist())]
    changed = []
    for k, i in enumerate(kept):
        for c in range(d.shape[1]):
            a, b = before[i, c], float(d[k, c])
            if (np.isnan(a) and not np.isnan(b)) or (not np.isnan(a) and a != b):
                changed.append(
                    {"id": str(names[i]), "column": str(tags[c]),
                     "before": None if np.isnan(a) else a, "after": b}
                )
    report = {
        "n_in": int(before.shape[0]),
        "n_used": int(d.shape[0]),
        "dropped": dropped,
        "imputed_cells": changed,
        "imputer": imputer_strat,
        "nstds": nstds,
    }
    if dropped:
        run_record.warn(f"{len(dropped)} profile(s) dropped as incomplete: {dropped}")
    if changed:
        run_record.warn(
            f"{len(changed)} cell(s) imputed or replaced as outliers (imputer {imputer_strat})"
        )
    return d, cb, ms, names_kept, report


def call_imputter(imp_alg):
    """
    Create an instance of the specified imputer type.

    Parameters:
        Imputer_type: Type of imputer. Options: "knn", "iterative", "simple".

    Returns:
        An instance of the specified imputer type.
    """
    # Lazy import to avoid paying sklearn import cost on code paths that do not
    # use missing-value imputation.
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

    if imp_alg == "knn":
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    elif imp_alg == "iterative":
        imputer = IterativeImputer(max_iter=10, random_state=0)
    elif imp_alg == "simple":
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    else:
        # no imputer was chosen (none) or the name is unknown: a grid point whose solve failed is
        # still filled, with the knn imputer, and the record says so
        run_record.warn(
            f"imputer {imp_alg!r}: grid points whose solve failed are filled with the knn imputer"
        )
        imputer = KNNImputer(n_neighbors=5, weights="uniform")
    return imputer


def check_existence(wdir, kinetic_mode, verb):
    """Check for the existence of necessary files."""
    kinetic_mode = False
    k_exist = False

    rows_to_search = ["c0", "initial_conc", "initial conc"]
    columns_to_search = ["k_forward", "k_reverse"]
    nx_path = find_input(wdir, "rxn_network", required=False)
    if nx_path is not None:
        if verb > 2:
            logging.info(f"{nx_path.name} exists")
        df = read_table(nx_path, index_col=0)
        last_row_index = df.index[-1]
        c0_exist = isinstance(last_row_index, str) and last_row_index.lower() in rows_to_search
        k_exist = all([column in df.columns for column in columns_to_search])
        if not c0_exist:
            logging.critical(f"Initial concentration not found in {nx_path.name}.")

    else:
        logging.critical("rxn_network.csv (or rxn_network.xlsx) not found.")

    filename = "reaction_data"
    extensions = [".csv", ".xls", ".xlsx"]
    energy_exist = any(
        Path(wdir, filename + extension).is_file() for extension in extensions
    )

    if energy_exist:
        if verb > 2:
            logging.info("Found reaction data file.")
        if k_exist:
            logging.info("Both energy data and rate constants are provided.")
    else:
        if k_exist:
            logging.info(
                "reaction_data file not found, but rate constants are provided."
            )
        else:
            logging.critical(
                "reaction_data file not found and rate constants are not provided."
            )

    kinetic_exists = os.path.exists(
        os.path.join(wdir, "kinetic_data.csv")
    ) or os.path.exists(os.path.join(wdir, "kinetic_data.xlsx"))
    if kinetic_mode:
        if kinetic_exists:
            if verb > 2:
                logging.info("Found kinetic data file.")
        else:
            logging.critical("kinetic_data file not found.")


_ROW_ISSUES = ("rows_dropped", "cells_imputed", "outliers", "few_rows")


def contract_report(table, network, kinetic_mode, imputer="none", nstds=None):
    """The reaction_profile/v1 report of a reaction (or kinetic) table against the network."""
    try:
        if kinetic_mode:
            return vprofile.validate_kinetic_data(table, network=network, imputer=imputer, nstds=nstds)
        return vprofile.validate_profile(
            table, consumer="mikimo", imputer=imputer, nstds=nstds, network=network
        )
    except vprofile.ArgumentError as e:
        sys.exit(str(e))


def validate_inputs(wdir, kinetic_mode, imputer="none", nstds=None, rows=False):
    """Check the tables of wdir with the reaction_profile/v1 contract and record the outcome.

    A table the contract refuses ends the run with every reason named. The resolved role of each
    column is saved in mikimo_result.json (role_map). Messages about dropped rows and imputed cells
    are kept only when `rows` is True (a screening run uses every row; a single-profile run uses one).
    """
    stem = "kinetic_data" if kinetic_mode else "reaction_data"
    table = find_input(wdir, stem)
    network = find_input(wdir, "rxn_network")
    report = contract_report(table, network, kinetic_mode, imputer, nstds)
    issues = [i for i in report["issues"] if rows or i.get("code") not in _ROW_ISSUES]
    run_record.put("role_map", report.get("role_map", {}))
    run_record.update(
        "validation",
        clear=bool(report["clear"]),
        issues=[f"{i['level']}: {i['message']}" for i in issues],
        roles_source=report.get("roles_source"),
        schema=report.get("schema"),
    )
    if not report["clear"]:
        errors = [i["message"] for i in report["issues"] if i["level"] == "error"]
        sys.exit(
            f"{Path(table).name} was refused by the reaction_profile/v1 check: "
            + "; ".join(errors)
            + ". (python -m navicat_volcanic.profile validate <table> --consumer mikimo --network "
            "<rxn_network> prints the full report.)"
        )
    for i in issues:
        if i["level"] == "warning":
            run_record.warn(i["message"])
    return report


def same_columns(df, report):
    """The table as pandas read it must hold the columns the check resolved roles for."""
    if list(df.columns[1:]) != list(report["role_map"]):
        sys.exit(
            f"The columns read {list(df.columns[1:])} differ from the columns checked "
            f"{list(report['role_map'])}."
        )


def check_network_sanity(df_network, states, issues):
    """Step and species-location checks that the shared contract does not make. True when sane."""
    net = df_network
    last = net.index[-1]
    if isinstance(last, str) and last.lower() in INITIAL_CONC_LABELS:
        net = net.drop(net.index[-1])
    ok = True

    def flag(msg):
        issues.append(" ".join(str(msg).split()))
        logging.warning(msg)

    weird = net.index[(~net.isin([-1, 1])).all(axis=1)].to_list()
    for step in weird:
        ok = False
        flag(f"Likely malformed network step: {step}")
        print(item("Likely malformed network step", step))
    roles = species_roles(states)
    r_idx = [i for i, r in enumerate(roles) if r == "reactant"]
    p_idx = [i for i, r in enumerate(roles) if r in ("product", "target")]
    cols = net.columns.to_numpy()
    mask_r = (~net.iloc[:, r_idx].isin([-1])).all(axis=0).to_numpy()
    if mask_r.any():
        ok = False
        flag(f"The reactant location: {list(cols[r_idx][mask_r])} is likely wrong.")
    mask_p = (~net.iloc[:, p_idx].isin([1])).all(axis=0).to_numpy()
    if mask_p.any():
        ok = False
        flag(f"The product location: {list(cols[p_idx][mask_p])} is likely wrong.")
    return ok


def report_validation(df_network, states, verb):
    """Run the step checks, add them to the validation record and say whether the inputs look valid."""
    extra = []
    sane = check_network_sanity(df_network, states, extra)
    rec = run_record.current().data["validation"] if run_record.current() else {}
    run_record.update(
        "validation",
        clear=bool(rec.get("clear", True)) and sane,
        issues=list(rec.get("issues", [])) + extra,
    )
    if not sane:
        print(section("Input Validation"))
        print(item("Status", "Please recheck reaction network and reaction data."))
    elif verb > 0:
        print(section("Input Validation"))
        print(item("Status", "MKM inputs appear valid."))
    return sane


def _headless():
    """True when nobody can answer a prompt (stdin missing or not a terminal)."""
    try:
        return sys.stdin is None or not sys.stdin.isatty()
    except (AttributeError, ValueError):
        return True


def choose_descriptors(nd, d, tags, coeff, regress, given=None, offset=0, units="kcal/mol", verb=0):
    """The descriptor column (nd = 1) or the pair of columns (nd = 2) of a volcano or map.

    d, tags, coeff and regress are volcanic's: tags[0] is the reference state. `given` is the -id
    option: None asks at the terminal, and ends the run with the candidates listed when stdin is not
    a terminal; ["auto"] takes the candidate with the best mean r2 (ties broken with a fixed seed,
    reported); column indices (0 is the first column after the profile names) or names take
    those columns. `offset` is the number of placeholder columns in front of the table's own
    columns in `tags` (a kinetic table has no reference state, so it carries one). The choice is
    recorded in mikimo_result.json (vp.descriptor). Returns {"idx": index into tags (a pair for
    nd = 2), "chosen": [names], "how": str, "candidates": [...], "warnings": [...]}.
    """
    from navicat_volcanic import helpers as vh
    from navicat_volcanic.dv1 import find_1_dv
    from navicat_volcanic.dv2 import find_2_dv
    from navicat_volcanic.exceptions import DescriptorChoiceError

    find = find_1_dv if nd == 1 else find_2_dv
    candidates = vh.candidates_1_dv if nd == 1 else vh.candidates_2_dv
    choose = vh.choose_1_dv if nd == 1 else vh.choose_2_dv
    ask = vh.user_choose_1_dv if nd == 1 else vh.user_choose_2_dv
    names = [str(t).strip() for t in tags]
    what = "a descriptor" if nd == 1 else "a pair of descriptors"
    try:
        if not given:
            if _headless():
                cands = candidates(d, tags, coeff, regress, verb=-1)
                listing = ", ".join(
                    f"{' + '.join(c['cols'])} (r2 {c['r2']})" for c in cands[:10]
                ) + (f" and {len(cands) - 10} more" if len(cands) > 10 else "")
                raise DescriptorChoiceError(
                    f"vp -nd {nd} needs {what} and stdin is not a terminal, so it cannot ask. Pass "
                    f"-id auto, or -id with {'a column' if nd == 1 else 'two columns'} (an index "
                    f"counted from 0 after the profile names, or a name). Candidates, best mean "
                    f"r2 first: {listing}",
                    cands,
                )
            dvs, r2s = find(d, tags, coeff, regress, verb)
            idx = ask(dvs, r2s, tags)
            if idx is None or (nd == 2 and idx[0] is None):
                sys.exit("No descriptor is suitable for the user. Exiting.")
            pick = {
                "idx": idx,
                "chosen": [names[i] for i in ([idx] if nd == 1 else idx)],
                "how": "chosen at the prompt",
                "candidates": candidates(d, tags, coeff, regress, verb=-1),
                "warnings": [],
            }
        else:
            auto = [g for g in given if isinstance(g, str) and g.strip().lower() == "auto"]
            if auto and len(given) > 1:
                raise DescriptorChoiceError("-id auto cannot be combined with columns")
            choice = "auto" if auto else [g + offset if isinstance(g, int) else g for g in given]
            pick = choose(d, tags, coeff, regress, choice=choice, verb=verb)
    except DescriptorChoiceError as e:
        sys.exit(str(e))
    for w in pick.get("warnings", []):
        run_record.warn(w)
        if verb > 0:
            print(item("Warning", w))
    idx = pick["idx"]
    shown = [idx] if nd == 1 else list(idx)
    run_record.update(
        "vp",
        descriptor={
            "columns": list(pick["chosen"]),
            "index": [int(i) - offset for i in shown],
            "how": pick["how"],
            "units": units,
            "candidates": [
                {"columns": c["cols"], "r2": c["r2"]} for c in pick.get("candidates", [])[:10]
            ],
        },
    )
    if verb > 1:
        print(f"\n**Descriptor: {' and '.join(pick['chosen'])} ({pick['how']})**\n")
    return pick


def _descriptor_arg(value):
    """One -id value: `auto`, a column index or a column name."""
    v = value.strip()
    if v.lower() == "auto":
        return "auto"
    try:
        return int(v)
    except ValueError:
        return v


def preprocess_data_mkm(arguments, mode):
    mode_label = {
        "mkm_solo": "mkm",
        "mkm_screening": "vp",
        "mkm_cond": "cond",
    }.get(mode, mode)

    parser = argparse.ArgumentParser(
        prog="mikimo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "=== MIKIMO CLI ===\n"
            f"Mode: {mode_label}\n\n"
            "Input files expected in --dir:\n"
            "  - rxn_network.csv\n"
            "  - reaction_data.csv/.xlsx  (or kinetic_data.csv/.xlsx with --kinetic)\n\n"
            "Tip: outputs are written to timestamped mode folders by default.\n"
            "Use --outdir to override."
        ),
        epilog=(
            "Examples:\n"
            "  mkm : navicat_mikimo mkm  -d test_cases/test_1 -t 298.15 -Tf 60\n"
            "  vp  : navicat_mikimo vp   -d test_cases/pincer_CO2 -t 323.15 -Tf 7200 -nd 0\n"
            "  cond: navicat_mikimo cond -d test_cases/pincer_CO2_jacs -Tf 7200 7300 -t 298.15\n\n"
            "Please cite mikimo if used in published work: \n" \
            "Worakul, T. et al. ACS Catal. 2024, 14 (13), 9829–9839."
        ),
    )

    parser.add_argument(
        "-version", "--version", action="version", version="%(prog)s 2.0.1"
    )

    parser.add_argument(
        "-d",
        "--d",
        "--dir",
        dest="dir",
        default=".",
        type=str,
        help="Directory containing all required input files (reaction_data, rxn_network in csv or xlsx format)",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        default=".",
        type=str,
        help="Directory for generated outputs (default: outputs/<mode>/<timestamp>).",
    )
    parser.add_argument(
        "-e",
        "--eprofile_choice",
        dest="profile_choice",
        default=0,
        type=int,
        help="Choice of energy profile in the reaction data file. (default: 0 (topmost profile))",
    )

    parser.add_argument(
        "-Tf",
        "--Tf",
        "-Time",
        "--Time",
        dest="time",
        type=float,
        nargs="+",
        help="Total reaction time (s) (default: 1 day (86400 s)))",
    )
    parser.add_argument(
        "-t",
        "--t",
        "-temp",
        "--temp",
        dest="temp",
        type=float,
        nargs="+",
        help="Temperature in K. (default: 298.15 K)",
    )
    parser.add_argument(
        "-xbase",
        "--xbase",
        dest="xbase",
        type=float,
        default=20,
        help="Interval for the x-axis (default: 20)",
    )
    parser.add_argument(
        "-lm",
        "--lm",
        dest="lmargin",
        type=int,
        default=20,
        help="Left margin to pad for visualization, in descriptor variable units. (default: 20)",
    )
    parser.add_argument(
        "-rm",
        "--rm",
        dest="rmargin",
        type=int,
        default=20,
        help="Right margin to pad for visualization, in descriptor variable units. (default: 20)",
    )
    parser.add_argument(
        "-id",
        dest="idx",
        type=_descriptor_arg,
        nargs="+",
        help=(
            "Descriptor(s) for vp -nd 1 (one) and -nd 2 (two): `auto` takes the best mean r2 "
            "(a tie is broken with a fixed seed and reported), or give the columns by index "
            "(0 is the first column after the profile names; the same in energy and kinetic "
            "mode) or by name. Without -id the best candidates are offered at the terminal, and "
            "a run that cannot ask stops with the candidates listed. (default: None)"
        ),
    )
    parser.add_argument(
        "-p",
        "--p",
        "-percent",
        "--percent",
        dest="percent",
        action="store_true",
        help="Flag to report activity as percent yield. (default: False)",
    )
    parser.add_argument(
        "-pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=1,
        help="Plot mode for volcano and activity map plotting. Higher is more detailed, lower is basic. (default: 1)",
    )
    parser.add_argument(
        "-is",
        "--is",
        "--imputer_strat",
        dest="imputer_strat",
        type=str,
        default="none",
        help=(
            "Imputer to refill missing datapoints (simple, knn, iterative, none). With none, a "
            "profile with a missing value is dropped and named. (default: none)"
        ),
    )
    parser.add_argument(
        "-ci",
        "--ci",
        dest="confidence_interval",
        action="store_true",
        help="Toggle to compute confidence interval. (default: False)",
    )
    parser.add_argument(
        "-lfesr",
        "--lfesr",
        dest="lfesr",
        action="store_true",
        help="""Toggle to plot LFESRs. (default: False)""",
    )
    parser.add_argument(
        "-k",
        "--kinetic",
        dest="kinetic_mode",
        action="store_true",
        help="""Toggle to read and use kinetic profiles. (default: False)""",
    )
    parser.add_argument(
        "-iq",
        "--iq",
        dest="int_quality",
        type=int,
        default=1,
        help="""Integration quality (0-2). Higher values will take longer but yield a smoother plot. (default: 1)""",
    )
    parser.add_argument(
        "-pq",
        "--pq",
        dest="plot_quality",
        type=int,
        default=1,
        help="""Plot quality (0-2). (default: 1)""",
    )
    parser.add_argument(
        "-nd",
        "--nd",
        "--run_mode",
        dest="run_mode",
        type=int,
        default=1,
        help="""Run mode (default: 1)
0: Run mkm for every profile in the reaction data input.
1: Construct microkinetic volcano plot.
2: Construct microkinetic activity/selectivity map.
        """,
    )
    parser.add_argument(
        "-x",
        "--x",
        "--xscale",
        dest="xscale",
        type=str,
        default="ls",
        help="Time scale in the evolution plot (ls (log10(s)), s, lmin, min, h, day). (default=ls)",
    )
    parser.add_argument(
        "-a",
        "--a",
        dest="addition",
        type=int,
        nargs="+",
        help="Index of additional species to be included in the evolution plot",
    )
    parser.add_argument(
        "-v",
        "--v",
        "--verb",
        "--verbose",
        dest="verb",
        type=int,
        default=2,
        help="Verbosity level of the code. Higher is more verbose and vice versa. Set to at least 2 to generate csv/h5 output files. (default: 2)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Reduce console output (equivalent to verbosity 0).",
    )
    parser.add_argument(
        "-ncore",
        "--ncore",
        dest="ncore",
        type=int,
        default=1,
        help="Number of cpu cores for the parallel computing. (default: 1)",
    )
    parser.add_argument(
        "-ev",
        "--ev",
        "--plot_evo",
        dest="plot_evo",
        action="store_true",
        help="Toggle to plot evolution plots. (default: False)",
    )
    parser.add_argument(
        "-tt",
        "--map",
        dest="map",
        action="store_true",
        help="""Toggle to construct time-temperature map.
Requires the input of a temperature range (-t temperature_1 temperature_2) and
time range (-T time_1 time_2) in K and s respectively. (default: False)""",
    )

    args = parser.parse_args(arguments)
    cli_cfg = build_config_from_namespace(args, mode)
    shared = cli_cfg.shared

    verb = 0 if shared.quiet else shared.verb
    configure_console()
    logging.getLogger().setLevel(logging.ERROR if verb == 0 else logging.WARNING)
    wdir = str(Path(shared.dir).resolve())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_outdir_by_mode = {
        "mkm_solo": Path("outputs") / "mkm" / stamp,
        "mkm_screening": Path("outputs") / "vp" / stamp,
        "mkm_cond": Path("outputs") / "cond" / stamp,
    }
    raw_outdir = shared.outdir
    if raw_outdir == "." and mode in default_outdir_by_mode:
        raw_outdir = str(default_outdir_by_mode[mode])

    outdir = Path(raw_outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (Path.cwd() / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(outdir)
    if verb > 0:
        print(section("Run Configuration"))
        print(item("Output directory", outdir))
    profile_choice = shared.profile_choice
    x_scale = shared.xscale
    more_species_mkm = shared.addition
    quality = shared.int_quality
    temperatures = shared.temp
    kinetic_mode = shared.kinetic_mode
    t_finals = shared.time
    ncore = shared.ncore
    imputer_strat = str(shared.imputer_strat).strip().lower()
    if imputer_strat not in ("none", "knn", "simple", "iterative"):
        sys.exit(
            f"Unknown imputer {shared.imputer_strat!r}: use none, knn, simple or iterative."
        )

    run_record.start(
        mode_label,
        outdir,
        dir=wdir,
        outdir=str(outdir),
        T_K=temperatures,
        t_s=t_finals,
        kinetic_units="log10_k" if kinetic_mode else None,
        energy_units=None if kinetic_mode else "kcal/mol",
        profile=profile_choice if mode == "mkm_solo" else None,
        imputer=imputer_strat,
        verb=verb,
    )
    if not Path(wdir).is_dir():
        sys.exit(f"Input directory not found: {wdir}")

    if mode == "mkm_solo":
        profile_choice = args.profile_choice

        if t_finals is None:
            if verb > 0:
                print(item("Info", "No time input; using default 86400 s (1 day)."))
            t_finals = 86400
        elif len(t_finals) == 1:
            t_finals = t_finals[0]
        elif len(t_finals) > 1:
            if verb > 0:
                print(item("Info", "Time range provided; using first value."))
            t_finals = t_finals[0]

        if temperatures is None:
            if verb > 0:
                print(item("Info", "No temperature input; using default 298.15 K."))
            temperatures = 298.15
        elif len(temperatures) == 1:
            temperatures = temperatures[0]
        elif len(temperatures) > 1:
            if verb > 0:
                print(item("Info", "Temperature range provided; using first value."))
            temperatures = temperatures[0]
        run_record.update("inputs", T_K=temperatures, t_s=t_finals)

        df_network = read_network(wdir)
        check_existence(wdir, kinetic_mode, verb)
        report = validate_inputs(wdir, kinetic_mode)
        states = species_list(df_network.columns[:].tolist(), report)
        record_species(df_network, states)
        ks = None
        if kinetic_mode:
            df = read_profiles(wdir, kinetic_mode=True)
            check_profile_choice(profile_choice, df, "kinetic")
            ks = kinetic_rates(df, profile_choice, report["kinetic"])
            run_record.update("inputs", kinetic_columns=kinetic_record(report["kinetic"]))
            if pd.isnull(ks).any():
                sys.exit(
                    "The selected kinetic profile is incomplete (due to the presence of NaN)."
                )
            report_validation(df_network, states, verb)
            return (
                None,
                df_network,
                None,
                states,
                t_finals,
                temperatures,
                x_scale,
                more_species_mkm,
                ks,
                quality,
            )

        df = read_profiles(wdir, kinetic_mode=False)
        same_columns(df, report)
        check_profile_choice(profile_choice, df, "energy")
        dg = df.iloc[profile_choice].to_numpy()[1:]
        if pd.isnull(dg).any():
            sys.exit(
                "The selected energy profile is incomplete (due to the presence of NaN)."
            )
        dg = dg.astype(float)
        tags = df.columns.values[1:]
        run_record.put("pathways", report["pathways"])
        report_validation(df_network, states, verb)
        return (
            dg,
            df_network,
            tags,
            states,
            t_finals,
            temperatures,
            x_scale,
            more_species_mkm,
            ks,
            quality,
        )

    elif mode == "mkm_screening":
        lmargin = cli_cfg.lmargin
        rmargin = cli_cfg.rmargin
        xbase = cli_cfg.xbase
        report_as_yield = cli_cfg.percent
        p_quality = cli_cfg.plot_quality
        plotmode = cli_cfg.plotmode
        lfesr = cli_cfg.lfesr
        x_scale = args.xscale
        comp_ci = cli_cfg.confidence_interval
        nd = cli_cfg.run_mode
        lfesrs_idx = cli_cfg.idx
        run_record.update("inputs", nd=nd, descriptor_idx=lfesrs_idx)

        df_network = read_network(wdir)
        check_existence(wdir, kinetic_mode, verb)
        report = validate_inputs(wdir, kinetic_mode, imputer_strat, nstds=3, rows=True)
        states = species_list(df_network.columns[:].tolist(), report)
        n_target = len([states.index(i) for i in states if "*" in i])
        record_species(df_network, states)

        df = read_profiles(wdir, kinetic_mode)
        same_columns(df, report)
        if kinetic_mode:
            df.attrs["kinetic"] = report["kinetic"]
        else:
            run_record.put("pathways", report["pathways"])
        report_validation(df_network, states, verb)

        tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
        if ncore == -1:
            ncore = multiprocessing.cpu_count()
        if verb > 2:
            print(f"Will use {ncore} cores for parallel computing.")

        if plotmode == 0 and comp_ci:
            plotmode = 1

        return (
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
            t_finals,
            temperatures,
            kinetic_mode,
        )

    elif mode == "mkm_cond":
        plot_evo = cli_cfg.plot_evo
        map_tt = cli_cfg.map

        if t_finals is None:
            t_finals = [86400]
        elif len(t_finals) == 1:
            t_finals = [t_finals[0]]
        elif len(t_finals) > 1:
            pass

        if temperatures is None:
            temperatures = [298.15]
        elif len(temperatures) == 1:
            temperatures = [temperatures[0]]
        elif len(temperatures) > 1:
            pass
        run_record.update("inputs", T_K=temperatures, t_s=t_finals, time_temperature_map=map_tt)

        check_existence(wdir, kinetic_mode, verb)
        df_network = read_network(wdir)
        report = validate_inputs(wdir, kinetic_mode)
        states = species_list(df_network.columns[:].tolist(), report)
        record_species(df_network, states)

        ks = None
        if kinetic_mode:
            df = read_profiles(wdir, kinetic_mode=True)
            check_profile_choice(profile_choice, df, "kinetic")
            ks = kinetic_rates(df, profile_choice, report["kinetic"])
            if pd.isnull(ks).any():
                sys.exit(
                    "The selected kinetic profile is incomplete (due to the presence of NaN)."
                )
            report_validation(df_network, states, verb)
            return (
                None,
                df_network,
                None,
                states,
                t_finals,
                [0],
                x_scale,
                more_species_mkm,
                plot_evo,
                map_tt,
                ncore,
                imputer_strat,
                verb,
                ks,
                quality,
            )

        df = read_profiles(wdir, kinetic_mode=False)
        same_columns(df, report)
        check_profile_choice(profile_choice, df, "energy")

        dg = df.iloc[profile_choice].to_numpy()[1:]
        dg = dg.astype(float)
        if pd.isnull(dg).any():
            sys.exit(
                "The selected energy profile is incomplete (due to the presence of NaN)."
            )
        tags = df.columns.values[1:]

        run_record.put("pathways", report["pathways"])
        report_validation(df_network, states, verb)
        return (
            dg,
            df_network,
            tags,
            states,
            t_finals,
            temperatures,
            x_scale,
            more_species_mkm,
            plot_evo,
            map_tt,
            ncore,
            imputer_strat,
            verb,
            ks,
            quality,
        )


def process_data_mkm(
    dg: np.ndarray, df_network: pd.DataFrame, tags: List[str], states: List[str]
) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[np.ndarray], np.ndarray]:
    """
    Processes data for micokinetic modeling.

    Args:
        dg (numpy.array): free energy profile.
        df_network (pandas.DataFrame): Dataframe containing reaction network information.
        c0 (numpy.array): Initial conditions.
        tags (list): Column names for the free energy profile.
        states (list): Column names for the reaction network.

    Returns:
        Tuple: A tuple containing the following:
            initial_conc (numpy.ndarray): Initial concentrations.
            energy_profile_all (list): List of energy profiles.
            dgr_all (list): List of reaction free energies.
            coeff_ts_all (list): List of coefficient arrays.
            rxn_network_all (numpy.ndarray): Reaction networks.
    """
    # extract initial conditions
    initial_conc = np.array([])
    last_row_index = df_network.index[-1]
    if isinstance(last_row_index, str):
        last_row_index_lower = last_row_index.lower()
        if last_row_index_lower in ["initial_conc", "c0", "initial conc"]:
            initial_conc = df_network.iloc[-1:].to_numpy()[0]
            df_network = df_network.drop(df_network.index[-1])
    initial_conc = initial_conc.astype(np.float64)
    rxn_network_all = df_network.to_numpy()[:, :].astype(np.int32)

    # energy data-------------------------------------------
    df_all = pd.DataFrame([dg], columns=tags)
    df_all = df_all.astype(np.float64)
    species_profile = tags  # %%
    all_df = []
    df_corr_profiles = pd.DataFrame({"R": np.zeros(len(df_all))})

    for i in range(1, len(species_profile)):
        if column_role(states, species_profile[i]) == "END":
            df_corr_profiles = pd.concat(
                [df_corr_profiles, df_all[species_profile[i]]],
                ignore_index=False,
                axis=1,
            )
            all_df.append(df_corr_profiles)
            df_corr_profiles = pd.DataFrame({"R": np.zeros(len(df_all))})
        else:
            df_corr_profiles = pd.concat(
                [df_corr_profiles, df_all[species_profile[i]]],
                ignore_index=False,
                axis=1,
            )

    for i in range(len(all_df) - 1):
        try:
            # step where branching is (the first 1)
            branch_step = np.where(
                df_network[all_df[i + 1].columns[1]].to_numpy() == 1
            )[0][0]
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[1])[0]
        except KeyError:
            # due to TS as the first column of the profile
            branch_step = np.where(
                df_network[all_df[i + 1].columns[2]].to_numpy() == 1
            )[0][0]
            loc_nx = np.where(np.array(states) == all_df[i + 1].columns[2])[0]
        # int to which new cycle is connected (the first -1)

        if species_is_product(states, df_network.columns.to_list()[branch_step + 1]):
            # conneting profiles
            cp_idx = branch_step
        else:
            # int to which new cycle is connected (the first -1)
            cp_idx = np.where(rxn_network_all[branch_step, :] == -1)[0][0]

        # state to insert
        if species_is_product(states, states[loc_nx[0] - 1]) and not species_is_product(
            states, states[loc_nx[0]]
        ):
            # conneting profiles
            state_insert = all_df[i].columns[-1]
        else:
            state_insert = states[cp_idx]
        all_df[i + 1]["R"] = df_all[state_insert].values
        all_df[i + 1].rename(columns={"R": state_insert}, inplace=True)

    energy_profile_all = []
    dgr_all = []
    coeff_ts_all = []
    for df in all_df:
        energy_profile = df.iloc[0, :-1].values
        rxn_species = df.columns.to_list()[:-1]
        dgr_all.append(df.values[0][-1])
        coeff_ts = np.array(
            [1 if column_role(states, element) == "TS" else 0 for element in rxn_species],
            dtype=np.int32,
        )
        coeff_ts_all.append(coeff_ts)
        energy_profile_all.append(np.array(energy_profile).astype(np.float64))
    rxn_network_all = np.ascontiguousarray(rxn_network_all)

    return initial_conc, energy_profile_all, dgr_all, coeff_ts_all, rxn_network_all


def test_process_data_mkm():
    """test data processor"""
    dg = np.array(
        [
            0.0,
            14.6,
            0.5,
            20.1,
            -1.7,
            20.1,
            2.2,
            20.2,
            0.1,
            20.7,
            5.5,
            20.7,
            -6.4,
            27.1,
            -13.4,
        ]
    )

    df_network_dg = [
        [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0],
    ]
    df_network = pd.DataFrame(
        df_network_dg,
        columns=[
            "INT1",
            "INT2",
            "INT3",
            "P-HCOO[Si]*",
            "INT4",
            "INT5",
            "R-CO$_2$",
            "R-SiPh$H_3$",
            "P-CH$_2$(O[Si])$_2$*",
            "P-CH$_3$(O[Si])*",
        ],
    )
    new_row_names = ["1_1", "1_2", "1_3", "2_1", "2_2", "2_3", "3_3", "c0"]
    df_network = df_network.rename(index=dict(enumerate(new_row_names)))
    tags = [
        "INT1",
        "TS1",
        "INT2",
        "TS2",
        "INT3",
        "TS3",
        "P-HCOO[Si]*",
        "TS4",
        "INT4",
        "TS5",
        "INT5",
        "TS6",
        "P-CH$_2$(O[Si])$_2$*",
        "TS7",
        "P-CH$_3$(O[Si])*",
    ]
    states = [
        "INT1",
        "INT2",
        "INT3",
        "P-HCOO[Si]*",
        "INT4",
        "INT5",
        "R-CO$_2$",
        "R-SiPh$H_3$",
        "P-CH$_2$(O[Si])$_2$*",
        "P-CH$_3$(O[Si])*",
    ]

    # Expected output
    initial_conc_expected = np.array(
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0]
    )
    energy_profile_all_expected = [
        np.array([0.0, 14.6, 0.5, 20.1, -1.7, 20.1]),
        np.array([2.2, 20.2, 0.1, 20.7, 5.5, 20.7]),
        np.array([5.5, 27.1]),
    ]
    dgr_all_expected = np.array([2.2, -6.4, -13.4])
    coeff_ts_all_expected = [
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0, 1, 0, 1, 0, 1]),
        np.array([0, 1]),
    ]
    rxn_network_all_expected = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Test the function
    (
        initial_conc,
        energy_profile_all,
        dgr_all,
        coeff_ts_all,
        rxn_network_all,
    ) = process_data_mkm(dg, df_network, tags, states)

    # Compare the results
    assert np.array_equal(initial_conc, initial_conc_expected)

    assert len(energy_profile_all) == len(energy_profile_all_expected)
    for profile, profile_expected in zip(
        energy_profile_all, energy_profile_all_expected
    ):
        assert np.array_equal(profile, profile_expected)

    assert np.array_equal(dgr_all, dgr_all_expected)
    assert len(coeff_ts_all) == len(coeff_ts_all_expected)
    for coeff_ts, coeff_ts_expected in zip(coeff_ts_all, coeff_ts_all_expected):
        assert np.array_equal(coeff_ts, coeff_ts_expected)

    assert np.array_equal(rxn_network_all, rxn_network_all_expected)
