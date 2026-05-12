"""
Motor Dubins (pure Python, fără compilare C++), adaptat după:
https://github.com/fgabbert/dubins_py (Fischer / Shkel & Lumelsky).

Modificări față de original:
- `psi` la `Waypoint` = orientare NED în **grade** (0° = Nord), ca în repo-ul sursă.
- `calc_dubins_param_radius` folosește direct **raza minimă de viraj** R (nu vel/phi_lim).
- Fără `matplotlib`; mesajele `print` din cazuri invalide sunt eliminate.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Tuple

import numpy as np


class TurnType(Enum):
    LSL = 1
    LSR = 2
    RSL = 3
    RSR = 4
    RLR = 5
    LRL = 6


class Waypoint:
    def __init__(self, x: float, y: float, psi: float):
        self.x = x
        self.y = y
        self.psi = psi


class Param:
    def __init__(self, p_init: Waypoint, seg_final: List[float], turn_radius: float):
        self.p_init = p_init
        self.seg_final = seg_final
        self.turn_radius = turn_radius
        self.type: TurnType = TurnType.LSL


def wrapTo360(angle: float) -> float:
    pos_in = angle > 0
    angle = angle % 360
    if angle == 0 and pos_in:
        angle = 360
    return angle


def wrapTo180(angle: float) -> float:
    q = (angle < -180) or (180 < angle)
    if q:
        angle = wrapTo360(angle + 180) - 180
    return angle


def headingToStandard(hdg: float) -> float:
    """Conversie NED (grade) -> unghi „standard” în grade (vezi codul sursă fgabbert)."""
    return wrapTo360(90 - wrapTo180(hdg))


def dubinsLSL(alpha, beta, d):
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    tmp1 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp0)
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
    if p_squared < 0:
        return -1, -1, -1
    t = (tmp1 - alpha) % (2 * math.pi)
    p = math.sqrt(p_squared)
    q = (beta - tmp1) % (2 * math.pi)
    return t, p, q


def dubinsRSR(alpha, beta, d):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp0)
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + 2 * d * (math.sin(beta) - math.sin(alpha))
    if p_squared < 0:
        return -1, -1, -1
    t = (alpha - tmp1) % (2 * math.pi)
    p = math.sqrt(p_squared)
    q = (-1 * beta + tmp1) % (2 * math.pi)
    return t, p, q


def dubinsRSL(alpha, beta, d):
    tmp0 = d - math.sin(alpha) - math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        return -1, -1, -1
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((math.cos(alpha) + math.cos(beta)), tmp0) - math.atan2(2, p)
    t = (alpha - tmp2) % (2 * math.pi)
    q = (beta - tmp2) % (2 * math.pi)
    return t, p, q


def dubinsLSR(alpha, beta, d):
    tmp0 = d + math.sin(alpha) + math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        return -1, -1, -1
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-1 * math.cos(alpha) - math.cos(beta)), tmp0) - math.atan2(-2, p)
    t = (tmp2 - alpha) % (2 * math.pi)
    q = (tmp2 - beta) % (2 * math.pi)
    return t, p, q


def dubinsRLR(alpha, beta, d):
    tmp_rlr = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8
    if abs(tmp_rlr) > 1:
        return -1, -1, -1
    p = (2 * math.pi - math.acos(tmp_rlr)) % (2 * math.pi)
    t = (alpha - math.atan2((math.cos(alpha) - math.cos(beta)), d - math.sin(alpha) + math.sin(beta)) + p / 2 % (2 * math.pi)) % (2 * math.pi)
    q = (alpha - beta - t + (p % (2 * math.pi))) % (2 * math.pi)
    return t, p, q


def dubinsLRL(alpha, beta, d):
    tmp_lrl = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-1 * math.sin(alpha) + math.sin(beta))) / 8
    if abs(tmp_lrl) > 1:
        return -1, -1, -1
    p = (2 * math.pi - math.acos(tmp_lrl)) % (2 * math.pi)
    t = (-1 * alpha - math.atan2((math.cos(alpha) - math.cos(beta)), d + math.sin(alpha) - math.sin(beta)) + p / 2) % (2 * math.pi)
    q = ((beta % (2 * math.pi)) - alpha - t + (p % (2 * math.pi))) % (2 * math.pi)
    return t, p, q


def calc_dubins_param_radius(wpt1: Waypoint, wpt2: Waypoint, turn_radius: float) -> Param:
    """Alege combinația LSL/LSR/... cu cost total minim; `turn_radius` = R_min."""
    if turn_radius <= 0:
        raise ValueError("turn_radius trebuie > 0.")

    param = Param(wpt1, [0.0, 0.0, 0.0], turn_radius)
    tz = [0, 0, 0, 0, 0, 0]
    pz = [0, 0, 0, 0, 0, 0]
    qz = [0, 0, 0, 0, 0, 0]

    psi1 = headingToStandard(wpt1.psi) * math.pi / 180
    psi2 = headingToStandard(wpt2.psi) * math.pi / 180

    dx = wpt2.x - wpt1.x
    dy = wpt2.y - wpt1.y
    d_big = math.hypot(dx, dy)
    d = d_big / turn_radius

    theta = math.atan2(dy, dx) % (2 * math.pi)
    alpha = (psi1 - theta) % (2 * math.pi)
    beta = (psi2 - theta) % (2 * math.pi)

    best_word = -1
    best_cost = -1.0

    tz[0], pz[0], qz[0] = dubinsLSL(alpha, beta, d)
    tz[1], pz[1], qz[1] = dubinsLSR(alpha, beta, d)
    tz[2], pz[2], qz[2] = dubinsRSL(alpha, beta, d)
    tz[3], pz[3], qz[3] = dubinsRSR(alpha, beta, d)
    tz[4], pz[4], qz[4] = dubinsRLR(alpha, beta, d)
    tz[5], pz[5], qz[5] = dubinsLRL(alpha, beta, d)

    for x in range(6):
        if tz[x] != -1:
            cost = tz[x] + pz[x] + qz[x]
            if cost < best_cost or best_cost == -1:
                best_word = x + 1
                best_cost = cost
                param.seg_final = [tz[x], pz[x], qz[x]]

    if best_word == -1:
        raise RuntimeError("Nu s-a găsit nicio traiectorie Dubins validă între waypoints.")

    param.type = TurnType(best_word)
    return param


def dubins_segment(seg_param, seg_init, seg_type):
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    seg_end = np.array([0.0, 0.0, 0.0])
    if seg_type == L_SEG:
        seg_end[0] = seg_init[0] + math.sin(seg_init[2] + seg_param) - math.sin(seg_init[2])
        seg_end[1] = seg_init[1] - math.cos(seg_init[2] + seg_param) + math.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif seg_type == R_SEG:
        seg_end[0] = seg_init[0] - math.sin(seg_init[2] - seg_param) + math.sin(seg_init[2])
        seg_end[1] = seg_init[1] + math.cos(seg_init[2] - seg_param) - math.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif seg_type == S_SEG:
        seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]
    return seg_end


def dubins_path_world(param: Param, t: float) -> np.ndarray:
    """Punct (x,y,theta) la distanță arcului `t` de-a lungul traiectoriei alese (metri)."""
    tprime = t / param.turn_radius
    p_init = np.array([0.0, 0.0, headingToStandard(param.p_init.psi) * math.pi / 180])
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    dirdata = np.array(
        [
            [L_SEG, S_SEG, L_SEG],
            [L_SEG, S_SEG, R_SEG],
            [R_SEG, S_SEG, L_SEG],
            [R_SEG, S_SEG, R_SEG],
            [R_SEG, L_SEG, R_SEG],
            [L_SEG, R_SEG, L_SEG],
        ]
    )
    types = dirdata[param.type.value - 1][:]
    param1 = param.seg_final[0]
    param2 = param.seg_final[1]
    mid_pt1 = dubins_segment(param1, p_init, types[0])
    mid_pt2 = dubins_segment(param2, mid_pt1, types[1])

    if tprime < param1:
        end_pt = dubins_segment(tprime, p_init, types[0])
    elif tprime < (param1 + param2):
        end_pt = dubins_segment(tprime - param1, mid_pt1, types[1])
    else:
        end_pt = dubins_segment(tprime - param1 - param2, mid_pt2, types[2])

    end_pt[0] = end_pt[0] * param.turn_radius + param.p_init.x
    end_pt[1] = end_pt[1] * param.turn_radius + param.p_init.y
    end_pt[2] = end_pt[2] % (2 * math.pi)
    return end_pt


def path_length(param: Param) -> float:
    return float(sum(param.seg_final)) * param.turn_radius
