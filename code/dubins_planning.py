"""
Planificare traiectorii Dubins între puncte intermediare, cu ordonare TSP (nearest neighbor).

Motor Dubins: adaptare Python după fgabbert/dubins_py (vezi `dubins_fgabbert_engine.py`) — fără binare
compilate, funcționează pe Windows fără Microsoft Visual C++ Build Tools.

Teorie: primitive CSC/CCC, rază minimă de viraj R_min.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence, Tuple

from dubins_fgabbert_engine import (
    Waypoint,
    calc_dubins_param_radius,
    dubins_path_world,
    path_length,
)

# Aliases de tip: punct 2D (x, y) și configurație completă a robotului (x, y, unghi_radiani)
Point2 = Tuple[float, float]
Config = Tuple[float, float, float]


def _config_to_waypoint(cfg: Config) -> Waypoint:
    """
    fgabbert folosește `psi` în grade, convenție NED (0° = Nord).
    Noi primim `theta` în radiani, convenție matematică (0 rad = axa +x, sens trigonometric).
    Legătura folosită: psi_NED = (90° - theta) mod 360°, astfel încât headingToStandard să revină la theta.
    """
    # Convertim theta (radiani, matematic) → psi (grade, NED) pentru motorul Dubins
    x, y, theta_rad = cfg
    psi_deg = (90.0 - math.degrees(theta_rad)) % 360.0
    return Waypoint(float(x), float(y), float(psi_deg))


def get_dubins_path(
    start: Config,
    end: Config,
    radius: float,
    step_size: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Calculează o polilinie (eșantionare) a celei mai scurte traiectorii Dubins între două configurații.

    Dubins: vehicul cu viteză constantă, rază minimă de viraj R = `radius` (metri sau unități coerente).
    Traiectoria e compusă din arce de cerc (L/R) și segmente drepte (S) — tipic CSC sau CCC.

    :param start: (x, y, theta) — theta în radiani, sens trigonometric (x axă, y în sus în matematică;
                  matplotlib folosește același convențional pentru unghiuri în plot).
    :param end: configurația țintă.
    :param radius: R_min — rază minimă de viraj (cu cât e mai mică, cu atât virajele sunt mai strânse).
    :param step_size: distanță între eșantioane de-a lungul traiectoriei (rezoluție grafică).
    :return: listă de (x, y) de-a lungul traiectoriei (fără duplicarea punctului final dacă e aproape).
    """
    if radius <= 0:
        raise ValueError("radius (R_min) trebuie să fie strict pozitiv.")

    # Convertim cele două configurații în waypoints pentru motorul Dubins
    w1 = _config_to_waypoint(start)
    w2 = _config_to_waypoint(end)
    param = calc_dubins_param_radius(w1, w2, radius)
    total_length = path_length(param)

    # Caz degenerat: start și end sunt practic identice
    if total_length < 1e-9:
        return [(start[0], start[1])]

    # Eșantionăm traiectoria la intervale fixe de step_size
    xs_ys: List[Tuple[float, float]] = []
    d = 0.0
    while d <= total_length + 1e-9:
        q = dubins_path_world(param, d)
        xs_ys.append((float(q[0]), float(q[1])))
        d += step_size

    # Adăugăm explicit punctul final pentru a evita gap-uri din cauza pasului discret
    qf = dubins_path_world(param, total_length)
    last = (float(qf[0]), float(qf[1]))
    if not xs_ys or (
        abs(xs_ys[-1][0] - last[0]) > 1e-6 or abs(xs_ys[-1][1] - last[1]) > 1e-6
    ):
        xs_ys.append(last)

    return xs_ys


def dubins_path_length(start: Config, end: Config, radius: float) -> float:
    """Lungimea analitică a traiectoriei Dubins optime între două configurații."""
    if radius <= 0:
        raise ValueError("radius trebuie să fie strict pozitiv.")
    w1 = _config_to_waypoint(start)
    w2 = _config_to_waypoint(end)
    param = calc_dubins_param_radius(w1, w2, radius)
    return path_length(param)


def tsp_nearest_neighbor(points: Sequence[Point2], start_index: int = 0) -> List[Point2]:
    """
    Heuristică TSP: nearest neighbor (cel mai apropiat vecin nevizitat).

    Nu garantează optimalitatea ordinii — e acceptabilă în cerință ca soluție simplă și rapidă.

    :param points: listă de (x, y) — puncte de vizitat o singură dată fiecare.
    :param start_index: indexul punctului de start în lista originală (implicit primul).
    :return: aceleași puncte, reordonate după parcurgerea greedy.
    """
    if not points:
        return []
    n = len(points)
    if n == 1:
        return [points[0]]

    # Inițializăm cu punctul de start; restul sunt „nevizitate"
    unvisited = set(range(n))
    current = start_index % n
    order_idx: List[int] = [current]
    unvisited.remove(current)

    while unvisited:
        cx, cy = points[current]
        best_j: Optional[int] = None
        best_d2 = math.inf

        # Găsim cel mai apropiat punct nevizitat (distanță euclidiană²)
        for j in unvisited:
            px, py = points[j]
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        assert best_j is not None
        order_idx.append(best_j)
        unvisited.remove(best_j)
        current = best_j

    return [points[i] for i in order_idx]


def _bearing_rad(ax: float, ay: float, bx: float, by: float) -> float:
    """Unghiul direcției de la A către B (radiani), atan2 în convenția standard."""
    return math.atan2(by - ay, bx - ax)


def polyline_headings(ordered_points: Sequence[Point2]) -> List[float]:
    """
    Asociază fiecărui punct din polilinie un unghi de orientare pentru concatenarea segmentelor Dubins:
    - la fiecare punct intermediar, orientarea = direcția către următorul punct;
    - la ultimul punct, păstrăm aceeași orientare ca la penultimul (sosire „înainte").
    """
    n = len(ordered_points)
    if n == 0:
        return []
    if n == 1:
        return [0.0]

    thetas: List[float] = []
    # Calculăm orientarea pentru fiecare segment dintre puncte consecutive
    for i in range(n - 1):
        x0, y0 = ordered_points[i]
        x1, y1 = ordered_points[i + 1]
        thetas.append(_bearing_rad(x0, y0, x1, y1))

    # Ultimul punct moștenește orientarea segmentului de intrare
    thetas.append(thetas[-1])
    return thetas


def concatenate_dubins_tour(
    ordered_points: Sequence[Point2],
    radius: float,
    step_size: float = 0.05,
) -> Tuple[List[Tuple[float, float]], float, Config, Config]:
    """
    Construiește traiectoria continuă: Dubins între puncte consecutive în ordinea dată.

    :return: (polyline_xy, lungime_totală, config_start, config_end)
    """
    # Caz degenerat: un singur punct → traiectorie vidă
    if len(ordered_points) < 2:
        p0 = ordered_points[0]
        th = 0.0
        cfg0: Config = (p0[0], p0[1], th)
        return [(p0[0], p0[1])], 0.0, cfg0, cfg0

    headings = polyline_headings(ordered_points)
    poly: List[Tuple[float, float]] = []
    total_len = 0.0

    # Parcurgem fiecare pereche de puncte consecutive și concatenăm segmentele Dubins
    for i in range(len(ordered_points) - 1):
        x0, y0 = ordered_points[i]
        x1, y1 = ordered_points[i + 1]
        q0: Config = (x0, y0, headings[i])
        q1: Config = (x1, y1, headings[i + 1])

        seg = get_dubins_path(q0, q1, radius, step_size=step_size)

        # Evităm duplicarea punctului de joncțiune între două segmente consecutive
        if poly and seg:
            if math.hypot(seg[0][0] - poly[-1][0], seg[0][1] - poly[-1][1]) < 1e-4:
                seg = seg[1:]
        poly.extend(seg)
        total_len += dubins_path_length(q0, q1, radius)

    # Configurațiile de start și end ale întregii traiectorii
    start_cfg: Config = (
        ordered_points[0][0],
        ordered_points[0][1],
        headings[0],
    )
    end_cfg: Config = (
        ordered_points[-1][0],
        ordered_points[-1][1],
        headings[-1],
    )
    return poly, total_len, start_cfg, end_cfg


def plot_tour(
    ordered_points: Sequence[Point2],
    trajectory_xy: Sequence[Tuple[float, float]],
    total_length: float,
    title: str,
    radius: float,
    start_cfg: Config,
    end_cfg: Config,
    save_path: Optional[str] = None,
) -> None:
    """
    Vizualizare matplotlib: puncte intermediare, traiectorie, start/final, lungime totală.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    # Punctele intermediare de vizitat
    xs = [p[0] for p in ordered_points]
    ys = [p[1] for p in ordered_points]
    ax.scatter(xs, ys, c="tab:blue", s=60, zorder=3, label="Puncte intermediare")

    # Traiectoria Dubins eșantionată
    if trajectory_xy:
        tx = [p[0] for p in trajectory_xy]
        ty = [p[1] for p in trajectory_xy]
        ax.plot(tx, ty, "-", color="tab:orange", linewidth=2, label="Traiectorie Dubins")

    # Marcatori speciali pentru start (verde, cerc) și final (roșu, pătrat)
    ax.scatter(
        [start_cfg[0]],
        [start_cfg[1]],
        c="green",
        s=140,
        marker="o",
        zorder=4,
        edgecolors="black",
        label="Start",
    )
    ax.scatter(
        [end_cfg[0]],
        [end_cfg[1]],
        c="red",
        s=160,
        marker="s",
        zorder=4,
        edgecolors="black",
        label="Final",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.set_title(f"{title}\nR_min = {radius:.2f}, lungime totală = {total_length:.3f}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def scenario_few_points() -> List[Point2]:
    """Scenariu mic: 4–5 puncte."""
    return [(0.0, 0.0), (2.0, 0.5), (3.5, 2.0), (1.0, 3.0), (0.5, 1.5)]


def scenario_many_points(n: int = 12, seed: int = 42) -> List[Point2]:
    """Scenariu cu multe puncte: distribuție pseudo-aleatoare în pătrat."""
    rng = random.Random(seed)
    return [(rng.uniform(0, 5), rng.uniform(0, 5)) for _ in range(n)]


def scenario_circular(n: int = 10, radius: float = 2.0, center: Point2 = (2.5, 2.5)) -> List[Point2]:
    """Puncte aproximativ pe cerc — TSP simplu poate da ordine suboptimă față de parcurgerea circulară."""
    pts: List[Point2] = []
    cx, cy = center
    # Distribuim uniform n puncte pe circumferință
    for k in range(n):
        ang = 2 * math.pi * k / n
        pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
    return pts


def run_experiment_radii(
    points: Sequence[Point2],
    radii: Iterable[float],
    out_dir: str,
    base_name: str,
    use_nn: bool = True,
    step_size: float = 0.05,
) -> None:
    """
    Rulează concatenarea Dubins pentru mai multe R_min și salvează figuri PNG (prezentare).
    """
    import os

    os.makedirs(out_dir, exist_ok=True)

    # Opțional: reordonăm punctele cu TSP nearest neighbor înainte de a trasa traiectoriile
    ordered = tsp_nearest_neighbor(list(points), start_index=0) if use_nn else list(points)

    for r in radii:
        traj, length, s_cfg, e_cfg = concatenate_dubins_tour(ordered, radius=r, step_size=step_size)

        # Înlocuim '.' cu 'p' în numele fișierului pentru compatibilitate cross-platform (ex. R0p5.png)
        safe_r = str(r).replace(".", "p")
        fname = os.path.join(out_dir, f"{base_name}_R{safe_r}.png")
        plot_tour(
            ordered,
            traj,
            length,
            title=f"{base_name} ({'TSP NN' if use_nn else 'ordine dată'})",
            radius=r,
            start_cfg=s_cfg,
            end_cfg=e_cfg,
            save_path=fname,
        )


def compare_nn_vs_random(
    points: Sequence[Point2],
    radius: float,
    seed: int = 123,
    step_size: float = 0.05,
) -> Tuple[float, float]:
    """
    Bonus: compară lungimea traiectoriei Dubins cu ordine NN vs ordine aleatoare (aceleași puncte).
    Returnează (lungime_nn, lungime_random).
    """
    # Calculăm lungimea traiectoriei cu ordinea optimizată TSP nearest neighbor
    nn_order = tsp_nearest_neighbor(list(points), start_index=0)
    _, len_nn, _, _ = concatenate_dubins_tour(nn_order, radius=radius, step_size=step_size)

    # Calculăm lungimea traiectoriei cu ordine aleatoare (același seed pentru reproductibilitate)
    idx = list(range(len(points)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    rnd_order = [points[i] for i in idx]
    _, len_rnd, _, _ = concatenate_dubins_tour(rnd_order, radius=radius, step_size=step_size)

    return len_nn, len_rnd