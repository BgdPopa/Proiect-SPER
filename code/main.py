"""
SPER — Traiectorii Dubins + TSP (nearest neighbor)
Rulare: python main.py
Figurile se salvează în ../output/figuri/
"""

import os
import sys

# asigură că importurile din același folder funcționează
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dubins_planning import (
    compare_nn_vs_random,
    run_experiment_radii,
    scenario_circular,
    scenario_few_points,
    scenario_many_points,
)

RADII = (0.5, 1.0, 2.0, 5.0)
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "output", "figuri"))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Figuri salvate in: {OUT_DIR}\n")

    scenarios = {
        "scenariu_putine": scenario_few_points(),
        "scenariu_multe": scenario_many_points(12),
        "scenariu_circular": scenario_circular(10),
    }

    for name, pts in scenarios.items():
        print(f"--- {name} ({len(pts)} puncte) ---")
        run_experiment_radii(pts, RADII, OUT_DIR, base_name=name, use_nn=True)
        print(f"  Figuri generate pentru R = {RADII}")

        len_nn, len_rnd = compare_nn_vs_random(pts, radius=1.0, seed=42)
        print(f"  Lungime TSP NN : {len_nn:.3f}")
        print(f"  Lungime random : {len_rnd:.3f}")
        print(f"  Diferenta      : {abs(len_nn - len_rnd):.3f}\n")

    print("Gata. Deschide folderul output/figuri/ pentru a vedea rezultatele.")

if __name__ == "__main__":
    main()