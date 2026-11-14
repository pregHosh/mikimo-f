# mikimo: microkinetic modeling and microkinetic volcano plots for homogeneous catalytic reactions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12731466.svg)](https://doi.org/10.5281/zenodo.12731466)
[![PyPI version](https://badge.fury.io/py/navicat_mikimo.svg)](https://pypi.org/project/navicat-mikimo/)

![workflow](./images/logo.png)

<details>
    <summary style="cursor: pointer;">
        ☄️ Why use microkinetic modeling? ☄️
    </summary>
    <p>
        <li>Elegant way to deal with complex reaction pathway thermodynamics and kinetics.</li>
        <li>Accounts for reaction conditions: temperature effects, concentration effects, reaction time, etc.
    </p>
</details>


<details>
    <summary style="cursor: pointer;">
        ☄️ What are microkinetic volcano plots? ☄️
    </summary>
    <p>
        <li>Volcano plot: diagrams that show the activity (or selectivity) of catalysts plotted against a descriptor variable that identifies a specific catalyst. Based on linear free energy scaling relationships. </li>
        <li>Microkinetic volcano plot: volcano plots in which the activity/selectivity is expressed as the final product concentration, or a ratio of concentrations, after a given time.
    </p>
</details>

## Contents 
* [Dependencies](#dependencies-)
* [Install](#install-)
* [Input Preparation](#input-preparation-)
* [Command Line Interface](#command-line-interface-)
* [Running Simulations](#running-simulations-)
* [Outputs](#outputs-)
* [Replotting Utility](#replotting-utility-)
* [Examples](#examples-)
* [Quickstart Validation](#quickstart-validation-)
* [Troubleshooting](#troubleshooting-)
* [Development & Release Workflow](#development--release-workflow-)
* [Limitations](#limitations-)
* [Citation](#citation-)


## Dependencies [↑](#dependencies)
The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `autograd`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `h5py`
- `fire`
- `navicat_volcanic`
- `openpyxl`
- `tqdm` 
- `joblib`
- `PyYAML` (for `navicat_mikimo_config`)
- `hydra-core` (optional, the `[hydra]` extra, for `navicat_mikimo_hydra`)

## Install [↑](#install)

```python
pip install .
```

or 

```python
python setup.py install
```

## Input Preparation [↑](#input-preparation)

The code requires two essential inputs for the analysis: energy/kinetic data and reaction network (along with initial concentrations). All files must be in the working directory or in the directory targeted with the `-d` option.

### 1. Reaction Data (Energy Profile)
*   **Filename:** `reaction_data.csv` or `reaction_data.xlsx`
*   **Format:** Matches the `navicat_volcanic` format found [here](https://raw.githubusercontent.com/lcmd-epfl/volcanic).
*   **Details:** 
    *   In cases involving multiple reaction pathways, the final column of each pathway (including those leading to the resting state) is an end column: its name contains "prod" (any case), or starts with `P-` or `P_`, or is just `P`.
    *   The first energy column is the reference state: an intermediate that is exactly 0.0 in every row. A table whose first column is not 0 is refused, never zeroed.
    *   Headers must be unique and non-empty, and every cell must be a number; a duplicate header is refused and named.
    *   Used by default.

### 2. Kinetic Data (Rate Constants)
*   **Filename:** `kinetic_data.csv` or `kinetic_data.xlsx`
*   **Format:** One row per catalyst/profile; the first column holds its name, the other columns hold rate constants as **log10 k** (the code uses 10**value; k in s⁻¹ or M⁻¹ s⁻¹ according to the molecularity of the step).
*   **Column names:** forward constants are `k_1 ... k_n` (or `kf_1 ... kf_n`), reverse constants `k_-1 ... k_-n` (or `kr_1 ... kr_n`). Step `i` is row `i` of `rxn_network` (counted from 1), so `n` must equal the number of network steps.
*   **Order:** columns are matched **by name**, not by position: forward `k_i` and reverse `k_-i` may come in any order (for example interleaved `k_1, k_-1, k_2, k_-2, ...`). A table whose names do not cover steps 1..n in both directions is refused, and so is a column that is neither a rate constant nor a `Descriptor` column; there is no positional fallback.
*   **Descriptors:** a column whose name contains `Descriptor` (for example `Descriptor pKa`) is a descriptor for `vp`, never a rate constant.
*   **Details:** 
    *   Used when the `-k` or `--kinetic` flag is active.
    *   Allows inputting direct rate constants, bypassing Eyring equation calculations.
    *   Plot axes in this mode are labelled log10(k).
    *   **Note:** Screening over temperature ranges is not supported in this mode.

### 3. Reaction Network
*   **Filename:** `rxn_network.csv` or `rxn_network.xlsx`
*   **Format:** Pandas-compatible CSV/XLSX.
*   **Structure:**
    *   **Rows:** Elementary steps.
    *   **Columns:** Chemical species involved (excluding transition states).
    *   **Values:** Stoichiometric coefficients.
        *   Left side (Reactants): `-n`
        *   Right side (Products): `+n`
        *   Not involved: `0` or empty.
    *   **Matching:** Species names must match those in the `reaction_data`/`kinetic_data`.
    *   **Initial Concentrations:** Specified in the last row, labeled "c0", "initial_conc", or "initial conc".

Each input may be csv or xlsx (first sheet). If both `<name>.csv` and `<name>.xlsx` are in the input directory the run stops and names both; keep one.

### Input check and roles.json
Every run reads its tables through the shared `reaction_profile/v1` check of `navicat_volcanic` (`navicat_volcanic.profile`), which `volcanic` uses as well. A table it refuses stops the run with every reason named; `python -m navicat_volcanic.profile validate <table> --consumer mikimo --network <rxn_network>` prints the full report. The resolved role of every column is saved in `mikimo_result.json` (`role_map`).

*   **Columns.** Each column of `reaction_data` is `INT` (a stationary point), `TS` (a name with `TS` in it, for example `TS1`, `B-TS1`), `END` (see above) or a descriptor (the name contains `Descriptor`). When the names are ambiguous, put a `roles.json` next to the table (or `reaction_data.roles.json`): `{"roles": {"column": "INT"}}`. A column the file does not list falls back to the name rule, with a warning.
*   **Species.** Each species of `rxn_network` must be an `INT` column of the table (species 0 is the catalyst), a reactant (`R`, `R-x`, `R_x`, `R<digit>x`), or a product (`prod` in the name, `P-x`, `P_x`, `P<digit>x`, `P*`; a name holding `*` is a target). Any other species is refused, naming it. Declare its role under `"species"` in `roles.json`: `{"roles": {}, "species": {"L": "intermediate"}}`, with one of `reactant`, `product`, `target`, `intermediate`, `spectator`. A ligand that is released and bound again (net 0) is an `intermediate`, not a reactant. A role declared this way is also the role the solver uses in `mkm` and `vp` (the bounds of the ODE and the limiting amount behind `-p`); without a `roles.json` the solver reads the names (`R...` reactant, `P...` product), as before. The compiled solver that `cond` uses for its time-temperature map reads the names only. `test_cases/6_suzuki` and `test_cases/pd_carbocylic_borylation` ship one.
*   **Missing values and outliers.** The default `-is none` drops a profile with a missing value and names it. `-is knn`, `iterative` or `simple` fill missing energies and list every filled cell. Values beyond 3 standard deviations of their column (`vp`) are treated as missing.

## Command Line Interface [↑](#command-line-interface)

The main tool is `navicat_mikimo`. Below are the available flags:

| Flag | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| `-d` | `--dir` | Directory containing input files. | `.` |
| `-o` | `--outdir` | Directory where output artifacts are written. | mode-specific timestamped folder |
| `-e` | `--eprofile_choice` | Index of energy profile to use (for single runs). | `0` |
| `-t` | `--temp` | Temperature(s) in Kelvin. | `298.15` |
| `-Tf` | `--Time` | Total reaction time(s) in seconds. | `86400` (1 day) |
| `-k` | `--kinetic` | Toggle to use `kinetic_data` instead of `reaction_data`. | `False` |
| `-nd` | `--run_mode` | `0`: MKM for all profiles, `1`: Volcano Plot, `2`: Activity Map. | `1` |
| `-id` | | Descriptor column(s) of `vp -nd 1` (one) and `-nd 2` (two): `auto` takes the candidate with the best mean r² (a tie is broken with a fixed seed and reported), or give the column by index or by name. An index counts from 0 over the columns after the name column, in energy and in kinetic mode alike (with `-k`: the rate constants in step order, forward then reverse, then the descriptor columns). The choice is recorded in `mikimo_result.json` (`vp.descriptor`). Without `-id` the run asks at the terminal; when stdin is not a terminal it stops and lists the candidates. | none |
| `-ncore`| `--ncore` | Number of CPU cores for parallel computing. | `1` |
| `-v` | `--verb` | Verbosity level (2 generates output files). | `2` |
| `-q` | `--quiet` | Minimize console output (sets verbosity to 0). | `False` |
| `-pm` | `--plotmode` | Plot detail level (`0`-`3`). | `1` |
| `-pq` | `--pq` | Plot quality (`0`-`2`): number of points on the volcano line and map. | `1` |
| `-iq` | `--iq` | Integration quality (`0`-`2`): tighter solver steps and tolerances. | `1` |
| `-p` | `--percent` | Report activity as percent yield instead of concentration. | `False` |
| `-ev` | `--plot_evo` | Toggle to generate evolution plots for all profiles. | `False` |
| `-x` | `--xscale` | Time scale for plots (`ls`, `s`, `lmin`, `min`, `h`, `d`). | `ls` (log seconds) |
| `-a` | `--a` | Indices of additional species to draw in the evolution plot. | none |
| `-ci` | `--ci` | Compute confidence intervals (requires statistical data). | `False` |
| `-lfesr` | `--lfesr` | Plot the linear free energy scaling relationships. | `False` |
| `-xbase` | `--xbase` | Tick interval of the descriptor axis. | `20` |
| `-lm` | `--lm` | Left margin of the descriptor range, in descriptor units. | `20` |
| `-rm` | `--rm` | Right margin of the descriptor range, in descriptor units. | `20` |
| `-is` | `--imputer_strat`| Strategy to fill missing data (`none`, `knn`, `iterative`, `simple`). With `none` a profile with a missing value is dropped and named in the output and in `mikimo_result.json`; with another strategy every filled cell is listed there. | `none` |
| `-tt` | `--map` | Construct a Time-Temperature activity map. | `False` |
| `-version` | `--version` | Print the version and exit. | |

## Running Simulations [↑](#running-simulations)

The code offers three main modes of operation:

### 1. Single MKM Run (`mkm`)
Performs a single microkinetic modeling run. If multiple profiles exist, the top-most row (or specified via `-e`) is used.
```python
navicat_mikimo mkm -d [DIR]
```

### 2. Screening / Volcano Plots (`vp`)
Screens over all profiles in the reaction data file.
*   **Run MKM for all profiles (no plot):** `-nd 0`
*   **Volcano Plot:** `-nd 1`
*   **Activity/Selectivity Map:** `-nd 2`

```python
navicat_mikimo vp -d [DIR] -nd 1 -id auto -is knn
```

`-id auto` chooses the descriptor without asking, so the command also runs unattended (`-is knn` fills the gaps of `pincer_CO2`, where every profile has one; a table without gaps needs no `-is`).

### 3. Conditions Screening (`cond`)
Screens over reaction time and/or temperature ranges.
```python
navicat_mikimo cond -d [DIR]
```

## Outputs [↑](#outputs)

Default output locations:
- `mkm`: `outputs/mkm/<timestamp>/`
- `vp`: `outputs/vp/<timestamp>/`
- `cond`: `outputs/cond/<timestamp>/`

Timestamp format: `YYYYmmdd_HHMMSS` (for example `20260424_195500`).
Use `--outdir` to override.

Runtime logging:
- At startup, every mode prints `Run Configuration` with the resolved `Output directory`.
- At completion, `vp` and `cond` print `Output Summary` listing generated artifacts and the same output directory.

*   **Every run (`mkm`, `vp`, `cond`):** `mikimo_result.json` (schema `mikimo_result/v1`): `status` (`ok`, `partial` when some profiles failed, `failed` when a single solve failed, `error`), `error`, the inputs with their sha256, the validation issues, every species with its role (`catalyst`, `intermediate`, `reactant`, `product`, `target`), `c0` and (for `mkm`) `c_final`, and per mode `mkm` (target concentrations and yields, solver method/success/fallbacks; each target's `yield_pct` is its concentration over `yield_reference_M`, the most of it the steps can make from the initial concentrations (a small linear program over the step extents with every concentration kept at or above 0), and is null with a warning saying why when there is no finite maximum, e.g. when unbalanced steps can make the target without consuming anything), `vp` (descriptor, per-profile status and target concentrations, dropped rows and imputed cells, grid file) or `cond` (grid file and axes). A single `mkm` run whose solve fails exits non-zero.
*   **Single Run (`mkm`):** 
    *   `mkm_output.json`: Time-course data (old keys kept for older readers).
    *   `mkm_timecourse.csv`: time and every species.
    *   `mkm_[name].png`: Concentration evolution plot.
*   **Volcano/Screening (`vp`, `cond`):** 
    *   **HDF5 Files:** `mkm_vp.h5`, `mkm_descr_phys.h5`, `mkm_vp_3d.h5`. Contain raw data for replotting.
    *   **Plots:** Combo plots, profile plots, and activity/selectivity maps (PNG).

## Replotting Utility [↑](#replotting-utility)

The `replot` mode (module `navicat_mikimo/replot.py`) smooths or adjusts a 2D volcano from an `mkm_vp.h5` file written by `vp -nd 1`. Outputs go to the working directory.

```python
navicat_mikimo replot [h5_file] [options]
```

**Options:**
*   `-f`: Filter method (`savgol`, `wiener`, `None`). Default: `savgol`.
*   `-w`: Window length (list of integers, one per product). Default: 20 for every product.
*   `-p`: Polynomial order (list of integers, one per product, for Savitzky-Golay). Default: 3 for every product.
*   `-s`: Save polished data to new HDF5 file.
*   `-pm`: Plot mode.

**Example:**
```python
navicat_mikimo replot examples/data/pincer_co2/vp/data_a.h5 -p 3 3 3 -w 20 20 20
```

## Examples [↑](#examples)

1. Microkinetic modeling for Pd-catalyzed carbocyclization-borylation of enallene in the presence of chiral phosphoric acid (298.15 K, 1 min): 
```python
navicat_mikimo mkm -d test_cases/pd_carbocylic_borylation/ -t 298.15 -Tf 60
```

2. Microkinetic modeling for all profiles of the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 0 -is knn
```

3. Constructing microkinetic volcano plot for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 1 -id auto -is knn -ncore 24
```

4. Constructing microkinetic activity and selectivity maps for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (323.15 K, 2 h):
```python
navicat_mikimo vp -d test_cases/pincer_CO2/ -t 323.15 -Tf 7200 -nd 2 -id auto -is knn -ncore 24
```

5. Constructing microkinetic activity and selectivity maps with a descriptor variable representing catalyst and temperature [273.15-423.15 K] for the catalytic hydrosilylation of carbon dioxide with metal pincer complexes (2h):
```python
navicat_mikimo vp -d test_cases/pincer_CO2/ -t 273.15 423.15 -Tf 7200 -nd 1 -id auto -is knn -ncore 24
```

6. Constructing microkinetic activity and selectivity maps with reaction time [2-24 hr] and temperature [273.15-423.15 K] as descriptors for the catalytic hydrosilylation of carbon dioxide with the Co pincer complex:
```python
navicat_mikimo cond -d test_cases/pincer_CO2_jacs/ -tt -Tf 7200 86400 -t 273.15 423.15 -ncore 24
```

7. Generate evolution plots for all profiles in a dataset:
```python
navicat_mikimo vp -d [DIR] -nd 0 -ev -is knn
```

8. Using filtering to smooth the plot:
```python
navicat_mikimo replot examples/data/pincer_co2/vp/data_a.h5 -p 3 3 3 -w 20 20 20
```

You can find examples demonstrating how to read h5 files and regenerate plots in the "examples" folder.

## Config-Driven Runs (Optional) [↑](#config-driven-runs-optional)

For reproducible runs, you can use either a YAML config file or Hydra overrides.

1. YAML config runner (simple, explicit file):
```bash
navicat_mikimo_config --config examples/configs/mkm_test1.yaml
```

2. Hydra entrypoint (composed config + overrides):
```bash
python -m pip install -e .[hydra]
navicat_mikimo_hydra mode=mkm shared.dir=test_cases/test_1 shared.temp=[298.15] shared.time=[60]
```

3. Hydra multirun sweep example:
```bash
navicat_mikimo_hydra --multirun mode=mkm shared.dir=test_cases/test_1 shared.temp='[298.15]' shared.time='[30],[60]' shared.verb=0
```

When to use which:
- `navicat_mikimo_hydra`: best for interactive overrides and sweeps (`--multirun`).
- `navicat_mikimo_config`: best for stable, file-based reproducible runs.

Notes:
- All three entry points share the same defaults (verbosity 2, outputs in `outputs/<mode>/<timestamp>` unless `outdir` is set).
- In a YAML config a relative `shared.dir` is relative to the config file.
- With Hydra, `mode=vp` or `mode=cond` also selects that mode's options, so `mode_config.run_mode=0` or `mode_config.map=true` can be overridden directly. In a `--multirun` sweep without `shared.outdir`, each job writes to its own Hydra job folder (`multirun/<date>/<time>/<job>`).

Current recommendation: keep using legacy `navicat_mikimo ...` commands for day-to-day work while this config workflow is stabilized.

## Quickstart Validation [↑](#quickstart-validation)

After installation, run:

```bash
pytest -q
navicat_mikimo mkm -d test_cases/test_1 -t 298.15 -Tf 60 -v 1
navicat_mikimo vp -d test_cases/pincer_CO2 -t 323.15 -Tf 7200 -nd 0 -ncore 1 -is knn -v 1
navicat_mikimo cond -d test_cases/pincer_CO2_jacs -Tf 7200 7300 -t 298.15 -ncore 1 -v 1
```

## Troubleshooting [↑](#troubleshooting)

1. **`rxn_network.csv` / `reaction_data` not found**  
   Use an absolute directory for `-d` or run from the intended input folder.
2. **Matplotlib/font cache permission warnings**  
   Set writable cache paths:
   `MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/cache`.
3. **Large `cond -tt` map is too slow for smoke tests**  
   Set `MIKIMO_TT_NPOINTS` (for example `MIKIMO_TT_NPOINTS=8`) to reduce grid size.
4. **Unexpected solver behavior in stiff systems**  
   Solver fallback diagnostics are attached on `solve_ivp` results as
   `mikimo_solver_diagnostics` (attempt list, fallback usage, selected method).
5. **Console color output**  
   Color is auto-enabled on interactive terminals. Force behavior with:
   `MIKIMO_COLOR=1` (force on), `MIKIMO_COLOR=0` or `NO_COLOR=1` (force off).

## Development & Release Workflow [↑](#development--release-workflow)

The package's own tests are in `navicat_mikimo/test_module.py`; `pyproject.toml` points pytest at them:

```bash
pytest -q
```

Numerical behavior is validated with `numpy.testing.assert_allclose`.

Release build:

```bash
python -m build
```

CI runs `ruff check` on `navicat_mikimo/__main__.py` and `navicat_mikimo/screen_reaction_condition.py`, then `pytest`, on Python 3.9 and 3.11. PyPI publish workflow builds tagged releases (`v*`).

## Limitations [↑](#limitations)

1. **Overlapping states** of different pathways before the reference state (starting point).
2. **Bridging states** between otherwise separate pathways.
3. **Different TSs** connecting the same 2 intermediates: just choose the lowest one or compute an effective TS energy that corresponds to the sum of the two rate constants.

To overcome these limitations and offer more flexibility, users have the option to input a kinetic profile named "kinetic_data" in either csv or xlsx format, replacing the conventional energy profile. However, this choice comes with the trade-off of disabling the ability to screen over a range of temperatures or use different temperature settings.

## Citation [↑](#citation)

If you use navicat_mikimo in your work, please cite our work and the publication.

```
Worakul, T., Laplaza, R., Das, S., Wodrich, M.D., Corminboeuf, C., Microkinetic Molecular Volcano Plots for Enhanced Catalyst Selectivity and Activity Predictions. ACS Catalysis 2024 14 (13), 9829-9839. 
```

BibTeX:

```bibtex
@article{worakul_microkinetic_2024,
	title = {Microkinetic {Molecular} {Volcano} {Plots} for {Enhanced} {Catalyst} {Selectivity} and {Activity} {Predictions}},
	volume = {14},
	copyright = {https://creativecommons.org/licenses/by-nc-nd/4.0/},
	issn = {2155-5435, 2155-5435},
	url = {https://pubs.acs.org/doi/10.1021/acscatal.4c01175},
	doi = {10.1021/acscatal.4c01175},
	language = {en},
	number = {13},
	urldate = {2026-04-24},
	journal = {ACS Catalysis},
	author = {Worakul, Thanapat and Laplaza, Rubén and Das, Shubhajit and Wodrich, Matthew D. and Corminboeuf, Clemence},
	month = jul,
	year = {2024},
	pages = {9829--9839},
}
```
[![DOI](https://img.shields.io/badge/DOI-10.1021/acscatal.4c01175-red)](https://pubs.acs.org/doi/10.1021/acscatal.4c01175)
