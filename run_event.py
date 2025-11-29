#first draft 9/17/2025
# run_event.py
# Purpose: Load one event from a "data module" and call the solver.
# Usage:
#   python run_event.py --data data_event1
# or just set DATA_MODULE below and run without args.

import argparse
import importlib
from main import f, pretty_print_result  # import from the library
import numpy as np

def load_event(module_name: str):
    """Import a data module that defines x1, x2, x3, e1, e2, e3."""
    m = importlib.import_module(module_name)
    required = ("x1", "x2", "x3", "e1", "e2", "e3")
    missing = [k for k in required if not hasattr(m, k)]
    if missing:
        raise AttributeError(f"{module_name} is missing: {', '.join(missing)}")
    return m.x1, m.x2, m.x3, m.e1, m.e2, m.e3

def main():
    parser = argparse.ArgumentParser(description="Run single-event 3γ vertex reconstruction.")
    parser.add_argument("--data", default="data_event1", help="Python module providing x1,x2,x3,e1,e2,e3")
    args = parser.parse_args()

    x1, x2, x3, e1, e2, e3 = load_event(args.data)
    r, info = f(x1, x2, x3, e1, e2, e3)
    pretty_print_result(r, info)
    # 新增：打印角度求和（应≈360）
    print(f"sum(angles_theory_deg)         = {float(info['angles_theory_deg'].sum()):.6f}")
    print(f"sum(angles_measured_cyclic_deg) = {float(info['angles_measured_cyclic_deg'].sum()):.6f}")
    import math
    assert math.isclose(float(info['angles_theory_deg'].sum()), 360.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(float(info['angles_measured_cyclic_deg'].sum()), 360.0, rel_tol=0, abs_tol=1e-6)

if __name__ == "__main__":
    main()
