import importlib
import sys

import fire

BANNER = r"""
███    ███ ██ ██   ██ ██ ███    ███  ██████
████  ████ ██ ██  ██  ██ ████  ████ ██    ██
██ ████ ██ ██ █████   ██ ██ ████ ██ ██    ██
██  ██  ██ ██ ██  ██  ██ ██  ██  ██ ██    ██
██      ██ ██ ██   ██ ██ ██      ██  ██████
"""


def _get_mode_handler(mode):
    modules = {
        "mkm": "navicat_mikimo.kinetic_solver_runner",
        "vp": "navicat_mikimo.km_volcanic",
        "cond": "navicat_mikimo.screen_reaction_condition",
        "replot": "navicat_mikimo.replot",
    }
    if mode not in modules:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes are: mkm, vp, cond, replot."
        )
    module = importlib.import_module(modules[mode])
    if mode == "replot":
        return module.main
    return _recorded(module.main)


def _recorded(mode_main):
    """Run a mode and write its mikimo_result.json (run_record) whether it succeeds or fails."""
    from navicat_mikimo import run_record

    def handler(arguments=None):
        try:
            out = mode_main(arguments)
        except SystemExit as exc:
            if exc.code in (None, 0):
                run_record.finish()
            else:
                run_record.finish("error", error=exc.code)
            raise
        except BaseException as exc:
            run_record.finish("error", error=repr(exc))
            raise
        run_record.finish()
        return out

    return handler


def run(mode):
    """
    MIKIMO: microkinetic modeling toolkit

    Perform microkinetic modeling for homogeneous catalytic reactions.

    Modes:
    - mkm  : single profile simulation and concentration evolution
    - vp   : profile screening, volcano plots, activity/selectivity maps
    - cond : reaction-condition screening over time and/or temperature
    - replot: smooth a 2D volcano from an mkm_vp.h5 file (navicat_mikimo replot <h5> [options])

    Quick start:
    - navicat_mikimo mkm  -d test_cases/test_1 -t 298.15 -Tf 60
    - navicat_mikimo vp   -d test_cases/pincer_CO2 -t 323.15 -Tf 7200 -nd 0
    - navicat_mikimo cond -d test_cases/pincer_CO2_jacs -Tf 7200 7300 -t 298.15

    Args:
        mode (str): One of: mkm, vp, cond, replot.

    Returns:
        None
    """
    sys.exit(_get_mode_handler(mode)())


def main():
    print(BANNER)
    fire.Fire(run)


if __name__ == "__main__":
    main()
