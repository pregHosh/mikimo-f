import os
import sys
from typing import Dict, List

from navicat_mikimo.__main__ import _get_mode_handler

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception as exc:  # pragma: no cover - handled at runtime
    hydra = None
    _HYDRA_IMPORT_ERROR = exc
else:
    _HYDRA_IMPORT_ERROR = None

# the directory the command was launched in (Hydra's runtime.cwd is re-read per multirun job)
_LAUNCH_CWD = os.getcwd()


def _append_list_option(argv: List[str], flag: str, values):
    if values is None:
        return
    if isinstance(values, (str, int, float)):  # `idx: auto` or `idx: 4` rather than a list
        values = [values]
    argv.append(flag)
    argv.extend([str(v) for v in values])


def build_legacy_argv(cfg: DictConfig) -> List[str]:
    mode = cfg.mode
    shared: Dict[str, object] = OmegaConf.to_container(cfg.shared, resolve=True)
    mode_cfg: Dict[str, object] = OmegaConf.to_container(cfg.mode_config, resolve=True)

    argv: List[str] = ["-d", str(shared["dir"]), "-e", str(shared["profile_choice"])]
    argv.extend(["--outdir", str(shared.get("outdir", "."))])
    _append_list_option(argv, "-Tf", shared.get("time"))
    _append_list_option(argv, "-t", shared.get("temp"))
    argv.extend(["-x", str(shared["xscale"])])
    _append_list_option(argv, "-a", shared.get("addition"))
    argv.extend(["-iq", str(shared["int_quality"])])
    argv.extend(["-ncore", str(shared["ncore"])])
    argv.extend(["-is", str(shared["imputer_strat"])])
    argv.extend(["-v", str(shared["verb"])])
    if shared.get("quiet"):
        argv.append("--quiet")
    if shared.get("kinetic_mode"):
        argv.append("-k")

    if mode == "vp":
        argv.extend(["-xbase", str(mode_cfg["xbase"])])
        argv.extend(["-lm", str(mode_cfg["lmargin"])])
        argv.extend(["-rm", str(mode_cfg["rmargin"])])
        _append_list_option(argv, "-id", mode_cfg.get("idx"))
        argv.extend(["-pm", str(mode_cfg["plotmode"])])
        argv.extend(["-pq", str(mode_cfg["plot_quality"])])
        argv.extend(["-nd", str(mode_cfg["run_mode"])])
        if mode_cfg.get("percent"):
            argv.append("-p")
        if mode_cfg.get("confidence_interval"):
            argv.append("-ci")
        if mode_cfg.get("lfesr"):
            argv.append("-lfesr")
    elif mode == "cond":
        if mode_cfg.get("plot_evo"):
            argv.append("-ev")
        if mode_cfg.get("map"):
            argv.append("-tt")

    return argv


def _run(cfg: DictConfig):
    # A mode changes into its output directory; start every job (also each job of a
    # --multirun sweep, which runs in this process) from the directory the command was
    # launched in, so relative paths such as shared.dir resolve the same way for every job.
    os.chdir(_LAUNCH_CWD)
    mode = cfg.mode
    argv = build_legacy_argv(cfg)
    # In a --multirun sweep the jobs can start within the same second, so the default
    # outputs/<mode>/<timestamp> folder would be shared; each job writes to its own Hydra job
    # directory instead (multirun/<date>/<time>/<job number>) unless shared.outdir is set.
    from hydra.core.hydra_config import HydraConfig
    from hydra.types import RunMode

    hc = HydraConfig.get()
    multirun = hc.mode == RunMode.MULTIRUN
    if multirun and str(cfg.shared.get("outdir", ".")) == ".":
        argv[argv.index("--outdir") + 1] = str(hc.runtime.output_dir)
    handler = _get_mode_handler(mode)
    if not multirun:
        return handler(argv)
    try:
        return handler(argv)
    finally:
        # Hydra sets up the next job's directory from the current one
        os.chdir(_LAUNCH_CWD)


if hydra is not None:

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(cfg: DictConfig):
        return _run(cfg)

else:

    def main():
        sys.exit(
            "Hydra support requires hydra-core. Install with "
            "`python -m pip install -e .[hydra]`. "
            f"Import error: {_HYDRA_IMPORT_ERROR}"
        )


if __name__ == "__main__":
    main()
