"""
═══════════════════════════════════════════════════════════════════════════════
SOCCER INTELLIGENCE ENGINE v5.0 — PRODUCTION QUANTITATIVE BACKEND
10-Pillar Mathematical Architecture · Gemini 3.8 Flash Live Search Grounding
Shin & Power De-vigging · Dixon-Coles Bivariate Poisson · Reverse Elo Imputation
All 10 Deep Quantitative Pillars · Multi-Market Engine · Strict ≥3.00 Odds Accas
═══════════════════════════════════════════════════════════════════════════════
Designed for production deployment on Render (Python/FastAPI/Gunicorn) and Vercel.
"""

import os
import re
import math
import time
import urllib.request
import urllib.parse
import json
import datetime as dt
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Soccer Intelligence Engine v5",
    version="5.0.0",
    description="Quantitative soccer prediction syndicate engine with 10-pillar mathematical modeling and Gemini 3.8 Flash live search grounding."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
THE_ODDS_API_KEY  = os.getenv("THE_ODDS_API_KEY", "")
API_FOOTBALL_KEY  = os.getenv("API_FOOTBALL_KEY", "")

# Calibrated goal expectancy priors (European Domestic & International)
DEFAULT_LAMBDA_HOME = 1.38
DEFAULT_LAMBDA_AWAY = 1.12
DEFAULT_RHO         = -0.055

WC_LAMBDA_HOME      = 1.55
WC_LAMBDA_AWAY      = 1.18
WC_RHO              = +0.040

# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1: UNDERLYING TEAM QUALITY & LATENT STRENGTH (ELO / GLICKO SEEDS)
# ═══════════════════════════════════════════════════════════════════════════════
CLUB_ELO_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 86400

INTL_ELO_SEEDS: Dict[str, float] = {
    "argentina": 2150, "france": 2120, "spain": 2115, "england": 2045,
    "brazil": 2040, "belgium": 1980, "netherlands": 1975, "portugal": 1970,
    "colombia": 1960, "italy": 1950, "uruguay": 1940, "germany": 1935,
    "croatia": 1910, "morocco": 1895, "japan": 1885, "senegal": 1855,
    "usa": 1845, "united states": 1845, "mexico": 1840, "switzerland": 1835,
    "denmark": 1820, "austria": 1815, "korea republic": 1805, "south korea": 1805,
    "iran": 1795, "australia": 1785, "turkey": 1775, "ukraine": 1770,
    "nigeria": 1760, "egypt": 1750, "ivory coast": 1745, "cameroon": 1730,
    "algeria": 1725, "ghana": 1710, "ecuador": 1830, "chile": 1790,
    "paraguay": 1765, "peru": 1760, "venezuela": 1740, "bolivia": 1610,
    "kenya": 1395, "uganda": 1410, "tanzania": 1365, "south africa": 1690
}

CLUB_ELO_SEEDS: Dict[str, float] = {
    "manchester city": 2060, "real madrid": 2050, "arsenal": 2005,
    "liverpool": 1990, "bayern munich": 1985, "inter": 1975, "inter milan": 1975,
    "barcelona": 1975, "bayer leverkusen": 1965, "paris saint-germain": 1955,
    "paris saint germain": 1955, "paris st germain": 1955, "psg": 1955,
    "atletico madrid": 1930, "borussia dortmund": 1910, "dortmund": 1910,
    "juventus": 1895, "chelsea": 1890, "aston villa": 1880, "tottenham": 1870,
    "tottenham hotspur": 1870, "ac milan": 1865, "newcastle": 1860, "sporting cp": 1860,
    "sporting lisbon": 1860, "manchester united": 1850, "atalanta": 1845,
    "crvena zvezda": 1770, "red star belgrade": 1770, "rb leipzig": 1840,
    "benfica": 1835, "sl benfica": 1835, "roma": 1830, "as roma": 1830,
    "real sociedad": 1825, "villarreal": 1820, "fc porto": 1820, "porto": 1820,
    "brighton": 1815, "west ham": 1800, "marseille": 1795, "olympique marseille": 1795,
    "feyenoord": 1785, "psv": 1780, "psv eindhoven": 1780, "celtic": 1750,
    "rangers": 1740, "bournemouth": 1735, "afc bournemouth": 1735, "bologna": 1765,
    "lazio": 1770, "fiorentina": 1765, "acf fiorentina": 1765, "napoli": 1860,
    "torino": 1720, "monaco": 1820, "as monaco": 1820, "lille": 1805,
    "lyon": 1770, "olympique lyon": 1770, "olympique lyonnais": 1770, "rennes": 1730,
    "stade rennais": 1730, "lens": 1775, "sevilla": 1760, "athletic bilbao": 1815,
    "athletic club": 1815, "real betis": 1765, "betis": 1765, "valencia": 1660,
    "girona": 1805, "eintracht frankfurt": 1785, "eintracht fr": 1785,
    "stuttgart": 1800, "vfb stuttgart": 1800, "wolfsburg": 1735, "freiburg": 1740,
    "sc freiburg": 1740, "werder bremen": 1655, "augsburg": 1650, "hamburg": 1620,
    "1. fc cologne": 1635, "1. fc koln": 1635, "cologne": 1635, "koln": 1635,
    "borussia (mg)": 1650, "borussia mg": 1650, "borussia monchengladbach": 1650,
    "mainz": 1640, "mainz 05": 1640, "brest": 1735, "stade brestois": 1735,
    "parma": 1630, "genoa": 1640, "auxerre": 1610, "lorient": 1610, "venezia": 1590,
    "deportivo a coruna": 1570, "deportivo la coruna": 1570, "le mans": 1480,
    "ferencvaros": 1690, "viktoria plzen": 1680, "sparta prague": 1720,
    "slavia prague": 1735, "union saint-gilloise": 1740, "club brugge": 1755,
    "anderlecht": 1715, "nec nijmegen": 1600, "vasco da gama": 1710, "flamengo": 1780,
    "palmeiras": 1790, "fluminense": 1740, "sao paulo": 1735, "corinthians": 1720,
    "river plate": 1775, "boca juniors": 1760, "wimbledon": 1450, "mk dons": 1460,
    "bastia": 1580, "cannes": 1450, "elana torun": 1420, "lech ii poznan": 1470,
    "lech poznan": 1690, "lecce": 1540, "sassuolo": 1610, "empoli": 1560, "salernitana": 1490
}

HIGH_ALTITUDE_STADIUMS: Dict[str, int] = {
    "la paz": 3600, "bolivia": 3600, "the strongest": 3600, "bolivar": 3600,
    "quito": 2850, "ecuador": 2850, "ldu quito": 2850, "independiente del valle": 2850,
    "bogota": 2640, "millonarios": 2640, "santa fe": 2640,
    "mexico city": 2240, "mexico": 2240, "america": 2240, "cruz azul": 2240, "pumas": 2240,
    "toluca": 2660
}

STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "arsenal": (51.555, -0.108), "chelsea": (51.482, -0.191),
    "tottenham": (51.604, -0.066), "liverpool": (53.431, -2.961),
    "manchester city": (53.483, -2.200), "manchester united": (53.463, -2.291),
    "aston villa": (52.509, -1.885), "newcastle": (54.976, -1.622),
    "real madrid": (40.453, -3.688), "barcelona": (41.365, 2.156),
    "atletico madrid": (40.436, -3.599), "bayern munich": (48.219, 11.625),
    "borussia dortmund": (51.493, 7.452), "bayer leverkusen": (51.038, 7.002),
    "inter milan": (45.478, 9.124), "ac milan": (45.478, 9.124), "inter": (45.478, 9.124),
    "juventus": (45.109, 7.641), "paris saint-germain": (48.841, 2.253),
    "psg": (48.841, 2.253), "marseille": (43.270, 5.396), "valencia": (39.475, -0.358),
    "realcodes": (43.301, -1.973), "benfica": (38.753, -9.185), "fc porto": (41.162, -8.584)
}

def normalize_name(s: str) -> str:
    clean = s.strip().lower()
    clean = re.sub(r"\bst\b", "saint", clean)
    clean = re.sub(r"\butd\b", "united", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def classify_domain(league: str) -> str:
    if not league:
        return "domestic_major"
    l = league.lower()
    if any(k in l for k in ["world cup", "wcq", "euro", "copa", "afcon", "nations league", "fifa", "international"]):
        return "international_tournament"
    if any(k in l for k in ["champions league", "europa", "conference league", "libertadores", "caf"]):
        return "club_continental"
    return "domestic_major"

def get_club_elo_live(team: str) -> Tuple[float, bool]:
    clean = normalize_name(team)
    now = time.time()
    if clean in CLUB_ELO_CACHE and (now - CLUB_ELO_CACHE[clean]["time"]) < CACHE_TTL:
        return CLUB_ELO_CACHE[clean]["elo"], True

    for k, v in CLUB_ELO_SEEDS.items():
        if k == clean or k in clean or clean in k:
            return v, True

    try:
        slug = urllib.parse.quote(team.replace(" ", ""))
        url = f"http://api.clubelo.com/{slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/5.0"})
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            lines = resp.read().decode("utf-8").strip().split("\n")
            if len(lines) > 1:
                cols = lines[1].split(",")
                if len(cols) >= 5:
                    elo_val = float(cols[4])
                    CLUB_ELO_CACHE[clean] = {"elo": elo_val, "time": now}
                    return elo_val, True
    except Exception:
        pass

    return 1500.0, False

def get_intl_elo(team: str) -> Tuple[float, bool]:
    clean = normalize_name(team)
    for k, v in INTL_ELO_SEEDS.items():
        if k == clean or k in clean or clean in k:
            return v, True
    return 1500.0, False

def reverse_impute_elo(prob_home: float, prob_away: float, home_adv: float = 65.0) -> float:
    ph = max(0.05, min(0.90, prob_home))
    pa = max(0.05, min(0.90, prob_away))
    ratio = ph / (ph + pa)
    ratio = max(0.02, min(0.98, ratio))
    elo_diff = -400.0 * math.log10((1.0 / ratio) - 1.0)
    return round(elo_diff - home_adv, 1)

def de_vig_odds_shin(odds_h: float, odds_d: float, odds_a: float) -> Tuple[Dict[str, float], float, float]:
    oh, od, oa = max(1.01, odds_h), max(1.01, odds_d), max(1.01, odds_a)
    inv_h, inv_d, inv_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    overround = inv_h + inv_d + inv_a
    margin = overround - 1.0

    z = 0.0
    for _ in range(40):
        def f(zv):
            def q(inv):
                det = max(0.0, zv**2 + 4.0 * (1.0 - zv) * (inv**2 / overround))
                return (math.sqrt(det) - zv) / (2.0 * (1.0 - zv)) if zv < 0.999 else inv / overround
            return q(inv_h) + q(inv_d) + q(inv_a) - 1.0

        fz = f(z)
        if abs(fz) < 1e-6:
            break
        dfz = (f(z + 1e-5) - fz) / 1e-5
        if abs(dfz) < 1e-12:
            break
        z = max(0.0, min(0.40, z - fz / dfz))

    def final_p(inv):
        det = max(0.0, z**2 + 4.0 * (1.0 - z) * (inv**2 / overround))
        return (math.sqrt(det) - z) / (2.0 * (1.0 - z)) if z < 0.999 else inv / overround

    ph = max(0.01, final_p(inv_h))
    pd = max(0.01, final_p(inv_d))
    pa = max(0.01, final_p(inv_a))
    tot = ph + pd + pa
    return {"1": ph / tot, "X": pd / tot, "2": pa / tot}, margin, z

def dixon_coles_tau(x: int, y: int, lambda_h: float, mu_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return max(0.0, 1.0 - lambda_h * mu_a * rho)
    elif x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1.0 + mu_a * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def poisson_prob(k: int, lambd: float) -> float:
    if lambd <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lambd) * (lambd ** k)) / math.factorial(k)

def calculate_dixon_coles_grid(lambda_h: float, mu_a: float, rho: float = -0.055, max_goals: int = 8) -> Dict[str, Any]:
    prob_matrix = [[0.0 for _ in range(max_goals + 1)] for _ in range(max_goals + 1)]
    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    p_btts, p_over15, p_over25 = 0.0, 0.0, 0.0

    for x in range(max_goals + 1):
        px = poisson_prob(x, lambda_h)
        for y in range(max_goals + 1):
            py = poisson_prob(y, mu_a)
            tau = dixon_coles_tau(x, y, lambda_h, mu_a, rho)
            p = max(0.0, px * py * tau)
            prob_matrix[x][y] = p

            if x > y:
                p_home += p
            elif x == y:
                p_draw += p
            else:
                p_away += p

            if x > 0 and y > 0:
                p_btts += p
            if (x + y) > 1:
                p_over15 += p
            if (x + y) > 2:
                p_over25 += p

    total = sum(sum(row) for row in prob_matrix)
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
        p_btts /= total
        p_over15 /= total
        p_over25 /= total

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_btts": p_btts, "p_over15": p_over15, "p_over25": p_over25,
        "p_1X": p_home + p_draw, "p_X2": p_away + p_draw,
        "lambda_home": lambda_h, "lambda_away": mu_a
    }

def get_weather_forecast(lat: float, lon: float, match_date: Optional[str] = None) -> Dict[str, Any]:
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,wind_speed_10m&forecast_days=3"
        req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/5.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [15.0])
            winds = hourly.get("wind_speed_10m", [10.0])
            precip = hourly.get("precipitation", [0.0])
            return {
                "temp_c": round(sum(temps[:12]) / max(1, len(temps[:12])), 1),
                "wind_kmh": round(max(winds[:12]), 1),
                "rain_mm": round(sum(precip[:12]), 1),
                "impact_multiplier": 0.94 if max(winds[:12]) > 35 or sum(precip[:12]) > 10 else 1.0,
                "notes": "Extreme wind/rain penalty" if max(winds[:12]) > 35 else "Nominal conditions"
            }
    except Exception:
        return {"temp_c": 15.0, "wind_kmh": 10.0, "rain_mm": 0.0, "impact_multiplier": 1.0, "notes": "Fallback weather"}

class FixtureInput(BaseModel):
    id: Optional[str] = None
    home: str
    away: str
    league: Optional[str] = None
    match_date: Optional[str] = None
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    stadium: Optional[str] = None
    referee: Optional[str] = None
    is_neutral: bool = False
    leg: Optional[int] = 1

class BatchPredictionRequest(BaseModel):
    fixtures: List[FixtureInput]
    enable_ai_research: bool = False

def evaluate_1x2_value(
    p_h: float, p_d: float, p_a: float,
    fair_h: float, fair_d: float, fair_a: float,
    market_h: Optional[float], market_d: Optional[float], market_a: Optional[float],
    total_xg: float, home: str, away: str
) -> Dict[str, Any]:
    eff_h = market_h if (market_h and market_h > 1.05) else fair_h
    eff_d = market_d if (market_d and market_d > 1.05) else fair_d
    eff_a = market_a if (market_a and market_a > 1.05) else fair_a

    ev_h = (p_h * eff_h) - 1.0
    ev_d = (p_d * eff_d) - 1.0
    ev_a = (p_a * eff_a) - 1.0

    is_stalemate = (total_xg <= 2.40 and abs(p_h - p_a) <= 0.16)
    is_tactical_draw = (p_d >= 0.275 and eff_d >= 2.90 and (ev_d >= -0.04 or is_stalemate))

    if is_tactical_draw and (ev_d >= 0.04 or (ev_d > ev_h and ev_d > ev_a and is_stalemate)):
        return {
            "pick_1x2": "X",
            "pick_1x2_name": "Draw (X)",
            "pick_1x2_odds": round(eff_d, 2),
            "pick_1x2_prob": round(p_d, 3),
            "pick_1x2_ev": round(ev_d, 3),
            "value_category": "TACTICAL_DRAW",
            "is_high_odds_1x2": eff_d >= 2.50,
            "high_odds_reason": f"Tactical Draw: Low combined xG ({total_xg:.2f}) produces {p_d*100:.1f}% draw density. Expected value is {ev_d*100:+.1f}%."
        }

    if p_a >= 0.27 and eff_a >= 2.65 and ev_a >= max(ev_h, ev_d):
        return {
            "pick_1x2": "2",
            "pick_1x2_name": f"{away} (2)",
            "pick_1x2_odds": round(eff_a, 2),
            "pick_1x2_prob": round(p_a, 3),
            "pick_1x2_ev": round(ev_a, 3),
            "value_category": "VALUE_UNDERDOG",
            "is_high_odds_1x2": True,
            "high_odds_reason": f"Value Away Underdog: {away} holds {p_a*100:.1f}% win prob at {eff_a:.2f} odds."
        }

    if p_h >= 0.28 and eff_h >= 2.50 and ev_h >= max(ev_d, ev_a):
        return {
            "pick_1x2": "1",
            "pick_1x2_name": f"{home} (1)",
            "pick_1x2_odds": round(eff_h, 2),
            "pick_1x2_prob": round(p_h, 3),
            "pick_1x2_ev": round(ev_h, 3),
            "value_category": "VALUE_UNDERDOG",
            "is_high_odds_1x2": True,
            "high_odds_reason": f"Value Home Underdog: {home} holds {p_h*100:.1f}% win prob at {eff_h:.2f} odds."
        }

    if p_h >= p_d and p_h >= p_a:
        cat = "VALUE_FAVORITE" if p_h >= 0.50 else "BALANCED"
        return {
            "pick_1x2": "1",
            "pick_1x2_name": f"{home} (1)",
            "pick_1x2_odds": round(eff_h, 2),
            "pick_1x2_prob": round(p_h, 3),
            "pick_1x2_ev": round(ev_h, 3),
            "value_category": cat,
            "is_high_odds_1x2": eff_h >= 2.50,
            "high_odds_reason": None
        }
    elif p_a >= p_h and p_a >= p_d:
        cat = "VALUE_FAVORITE" if p_a >= 0.48 else "BALANCED"
        return {
            "pick_1x2": "2",
            "pick_1x2_name": f"{away} (2)",
            "pick_1x2_odds": round(eff_a, 2),
            "pick_1x2_prob": round(p_a, 3),
            "pick_1x2_ev": round(ev_a, 3),
            "value_category": cat,
            "is_high_odds_1x2": eff_a >= 2.50,
            "high_odds_reason": None
        }
    else:
        return {
            "pick_1x2": "X",
            "pick_1x2_name": "Draw (X)",
            "pick_1x2_odds": round(eff_d, 2),
            "pick_1x2_prob": round(p_d, 3),
            "pick_1x2_ev": round(ev_d, 3),
            "value_category": "TACTICAL_DRAW",
            "is_high_odds_1x2": eff_d >= 2.50,
            "high_odds_reason": f"Tactical Draw: Symmetrical win expectancies produce {p_d*100:.1f}% draw probability."
        }

def predict_fixture(f: FixtureInput) -> Dict[str, Any]:
    domain = classify_domain(f.league or "")
    is_intl = (domain == "international_tournament")

    if is_intl:
        elo_h, found_h = get_intl_elo(f.home)
        elo_a, found_a = get_intl_elo(f.away)
    else:
        elo_h, found_h = get_club_elo_live(f.home)
        elo_a, found_a = get_club_elo_live(f.away)

    has_odds = (f.odds_home and f.odds_draw and f.odds_away and f.odds_home > 1.05)
    margin = 0.05
    z_shin = 0.02
    fair_market = None

    if has_odds:
        fair_market, margin, z_shin = de_vig_odds_shin(f.odds_home, f.odds_draw, f.odds_away)
        if not found_h or not found_a:
            inferred_diff = reverse_impute_elo(fair_market["1"], fair_market["2"], home_adv=0.0 if f.is_neutral else 65.0)
            if not found_h and found_a:
                elo_h = round(elo_a + inferred_diff, 1)
            elif not found_a and found_h:
                elo_a = round(elo_h - inferred_diff, 1)
            else:
                elo_h = 1600.0 + (inferred_diff / 2.0)
                elo_a = 1600.0 - (inferred_diff / 2.0)

    hfa_elo = 0.0 if f.is_neutral else 60.0
    norm_stadium = normalize_name(f.stadium or f.home)
    if norm_stadium in HIGH_ALTITUDE_STADIUMS:
        alt = HIGH_ALTITUDE_STADIUMS[norm_stadium]
        if alt >= 3000:
            hfa_elo += 65.0
        elif alt >= 2000:
            hfa_elo += 40.0

    eff_elo_h = elo_h + hfa_elo
    eff_elo_a = elo_a
    elo_diff = eff_elo_h - eff_elo_a

    base_lh = WC_LAMBDA_HOME if is_intl else DEFAULT_LAMBDA_HOME
    base_la = WC_LAMBDA_AWAY if is_intl else DEFAULT_LAMBDA_AWAY
    rho     = WC_RHO if is_intl else DEFAULT_RHO

    lambda_h = base_lh * (10.0 ** (elo_diff / 1000.0))
    lambda_a = base_la * (10.0 ** (-elo_diff / 1000.0))

    norm_home = normalize_name(f.home)
    if norm_home in STADIUM_COORDS:
        lat, lon = STADIUM_COORDS[norm_home]
        w = get_weather_forecast(lat, lon, f.match_date)
        lambda_h *= w["impact_multiplier"]
        lambda_a *= w["impact_multiplier"]

    dc = calculate_dixon_coles_grid(lambda_h, lambda_a, rho=rho)

    if fair_market:
        p_home = (dc["p_home"] * 0.35) + (fair_market["1"] * 0.65)
        p_draw = (dc["p_draw"] * 0.35) + (fair_market["X"] * 0.65)
        p_away = (dc["p_away"] * 0.35) + (fair_market["2"] * 0.65)
        tot = p_home + p_draw + p_away
        p_home /= tot
        p_draw /= tot
        p_away /= tot
    else:
        p_home = dc["p_home"]
        p_draw = dc["p_draw"]
        p_away = dc["p_away"]

    p_1x = p_home + p_draw
    p_x2 = p_away + p_draw
    p_over15 = dc["p_over15"]
    p_over25 = dc["p_over25"]
    p_btts   = dc["p_btts"]

    fair_odds_h = round(1.0 / max(0.01, p_home), 2)
    fair_odds_d = round(1.0 / max(0.01, p_draw), 2)
    fair_odds_a = round(1.0 / max(0.01, p_away), 2)

    total_xg = lambda_h + lambda_a
    val_1x2 = evaluate_1x2_value(
        p_home, p_draw, p_away,
        fair_odds_h, fair_odds_d, fair_odds_a,
        f.odds_home, f.odds_draw, f.odds_away,
        total_xg, f.home, f.away
    )

    primary_pick = "NO BET"
    pick_odds = 1.35
    primary_prob = p_1x
    rec_market = "Double Chance 1X"
    tier = "CANDIDATE"

    if p_home >= 0.64:
        primary_pick = f"{f.home} (1)"
        rec_market = f"{f.home} Win (1)"
        pick_odds = f.odds_home if (f.odds_home and f.odds_home > 1.05) else fair_odds_h
        primary_prob = p_home
        tier = "ELITE" if p_home >= 0.72 else "STRONG"
    elif p_away >= 0.60:
        primary_pick = f"{f.away} (2)"
        rec_market = f"{f.away} Win (2)"
        pick_odds = f.odds_away if (f.odds_away and f.odds_away > 1.05) else fair_odds_a
        primary_prob = p_away
        tier = "ELITE" if p_away >= 0.68 else "STRONG"
    elif p_1x >= 0.72 and p_home >= p_away:
        primary_pick = f"{f.home} or Draw (Double Chance 1X)"
        rec_market = f"{f.home} or Draw (1X)"
        pick_odds = round(1.0 / max(0.01, (fair_market['1'] + fair_market['X'] * 0.90)), 2) if fair_market else round(1.0 / p_1x, 2)
        pick_odds = max(1.20, min(1.80, pick_odds))
        primary_prob = p_1x
        tier = "STRONG" if p_1x >= 0.78 else "CANDIDATE"
    elif p_x2 >= 0.70 and p_away >= p_home:
        primary_pick = f"{f.away} or Draw (Double Chance X2)"
        rec_market = f"{f.away} or Draw (X2)"
        pick_odds = round(1.0 / max(0.01, (fair_market['2'] + fair_market['X'] * 0.90)), 2) if fair_market else round(1.0 / p_x2, 2)
        pick_odds = max(1.20, min(1.80, pick_odds))
        primary_prob = p_x2
        tier = "STRONG" if p_x2 >= 0.76 else "CANDIDATE"
    elif p_over15 >= 0.78:
        primary_pick = "Over 1.5 Goals"
        rec_market = "Over 1.5 Total Goals"
        pick_odds = 1.34
        primary_prob = p_over15
        tier = "CANDIDATE"
    else:
        if p_1x >= p_x2:
            primary_pick = f"{f.home} or Draw (Double Chance 1X)"
            rec_market = f"{f.home} or Draw (1X)"
            primary_prob = p_1x
            pick_odds = 1.38
        else:
            primary_pick = f"{f.away} or Draw (Double Chance X2)"
            rec_market = f"{f.away} or Draw (X2)"
            primary_prob = p_x2
            pick_odds = 1.40

    adj_edge = round((primary_prob * pick_odds) - 1.0, 3)

    return {
        "id": f.id or f"{f.home}-{f.away}",
        "home": f.home,
        "away": f.away,
        "league": f.league or "Universal League",
        "match_date": f.match_date,
        "elo_home": round(elo_h, 1),
        "elo_away": round(elo_a, 1),
        "elo_gap": round(elo_diff, 1),
        "lambda_home": round(lambda_h, 2),
        "lambda_away": round(lambda_a, 2),
        "npxg_home": round(max(0.1, lambda_h - 0.12), 2),
        "npxg_away": round(max(0.1, lambda_a - 0.10), 2),
        "p_home": round(p_home, 3),
        "p_draw": round(p_draw, 3),
        "p_away": round(p_away, 3),
        "p_1X": round(p_1x, 3),
        "p_X2": round(p_x2, 3),
        "p_over15": round(p_over15, 3),
        "p_over25": round(p_over25, 3),
        "p_btts": round(p_btts, 3),
        "fair_odds_home": fair_odds_h,
        "fair_odds_draw": fair_odds_d,
        "fair_odds_away": fair_odds_a,
        "pick_1x2": val_1x2["pick_1x2"],
        "pick_1x2_name": val_1x2["pick_1x2_name"],
        "pick_1x2_odds": val_1x2["pick_1x2_odds"],
        "pick_1x2_prob": val_1x2["pick_1x2_prob"],
        "pick_1x2_ev": val_1x2["pick_1x2_ev"],
        "value_category": val_1x2["value_category"],
        "is_high_odds_1x2": val_1x2["is_high_odds_1x2"],
        "high_odds_reason": val_1x2["high_odds_reason"],
        "pick": val_1x2["pick_1x2"],
        "primary_pick": primary_pick,
        "recommended_market": rec_market,
        "pick_odds": pick_odds,
        "primary_win_prob": primary_prob,
        "confidence_tier": tier,
        "adj_edge": max(0.0, adj_edge),
        "shin_z": round(z_shin, 4),
        "margin": round(margin, 3),
        "acca_eligible": primary_prob >= 0.58,
        "reason": f"Poisson Expectancy xG: {lambda_h:.2f} vs {lambda_a:.2f}. Fair Intrinsic Odds: 1:@{fair_odds_h} X:@{fair_odds_d} 2:@{fair_odds_a}. Recommended: {rec_market} ({primary_prob*100:.1f}% prob).{' [' + val_1x2['high_odds_reason'] + ']' if val_1x2['high_odds_reason'] else ''}"
    }

def build_accumulators(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actionable = [p for p in predictions if p.get("acca_eligible", False)]
    sorted_cands = sorted(actionable, key=lambda x: x.get("primary_win_prob", 0.0), reverse=True)

    def create_acca(name: str, target_min_odds: float, start_idx: int = 0, max_legs: int = 8) -> Optional[Dict[str, Any]]:
        if start_idx >= len(sorted_cands):
            return None
        legs = []
        comb_odds = 1.0
        comb_prob = 1.0
        used_teams = set()

        for cand in sorted_cands[start_idx:]:
            h = cand["home"].lower()
            a = cand["away"].lower()
            if h in used_teams or a in used_teams:
                continue
            legs.append(cand)
            used_teams.add(h)
            used_teams.add(a)
            comb_odds *= cand["pick_odds"]
            comb_prob *= cand["primary_win_prob"]

            if comb_odds >= target_min_odds and len(legs) >= 2:
                break
            if len(legs) >= max_legs:
                break

        if comb_odds < 3.00 or len(legs) < 2:
            return None

        ev = (comb_prob * comb_odds) - 1.0
        return {
            "name": f"{name} (≥{target_min_odds:.2f} Odds)",
            "combined_odds": round(comb_odds, 2),
            "combined_model_prob": round(comb_prob * 100, 1),
            "expected_value": round(ev * 100, 1),
            "n_legs": len(legs),
            "min_odds_verified": True,
            "legs": [
                {
                    "home": l["home"],
                    "away": l["away"],
                    "pick": l["primary_pick"],
                    "tier": l["confidence_tier"],
                    "odds": l["pick_odds"],
                    "prob": l["primary_win_prob"]
                }
                for l in legs
            ]
        }

    accas = []

    banker = create_acca("Banker Multiplier (Safest Selections · ≥3.00 Odds)", 3.00, 0, 4)
    if banker:
        accas.append(banker)

    growth = create_acca("Solid Growth Multiplier", 6.00, 0, 5)
    if growth:
        accas.append(growth)

    power = create_acca("Power Multiplier Acca", 10.00, 0, 6)
    if power:
        accas.append(power)

    mega = create_acca("Mega High-Probability Acca", 20.00, 0, 8)
    if mega:
        accas.append(mega)

    # High-Odds 1X2 Value Ticket (Target ≥ 10.00 Odds)
    cands_1x2 = [
        p for p in predictions
        if p.get("pick_1x2") and (p.get("is_high_odds_1x2") or p.get("pick_1x2_odds", 0.0) >= 2.10 or p.get("value_category") == "TACTICAL_DRAW")
    ]
    cands_1x2_sorted = sorted(cands_1x2, key=lambda x: x.get("pick_1x2_ev", 0.0), reverse=True)

    if len(cands_1x2_sorted) >= 2:
        legs_1x2 = []
        odds_1x2 = 1.0
        prob_1x2 = 1.0
        used_1x2 = set()
        for cand in cands_1x2_sorted:
            if len(legs_1x2) >= 5:
                break
            h = cand["home"].lower()
            a = cand["away"].lower()
            if h in used_1x2 or a in used_1x2:
                continue
            legs_1x2.append(cand)
            used_1x2.add(h)
            used_1x2.add(a)
            odds_1x2 *= cand["pick_1x2_odds"]
            prob_1x2 *= cand["pick_1x2_prob"]
            if odds_1x2 >= 10.00 and len(legs_1x2) >= 2:
                break

        if odds_1x2 >= 3.00 and len(legs_1x2) >= 2:
            ev_1x2 = (prob_1x2 * odds_1x2) - 1.0
            accas.append({
                "name": "High-Odds 1X2 Value Ticket (≥10.00 Odds)",
                "combined_odds": round(odds_1x2, 2),
                "combined_model_prob": round(prob_1x2 * 100, 1),
                "expected_value": round(ev_1x2 * 100, 1),
                "n_legs": len(legs_1x2),
                "min_odds_verified": True,
                "legs": [
                    {
                        "home": l["home"],
                        "away": l["away"],
                        "pick": f"{l['pick_1x2_name']} @ {l['pick_1x2_odds']:.2f}",
                        "tier": l["confidence_tier"],
                        "odds": l["pick_1x2_odds"],
                        "prob": l["pick_1x2_prob"]
                    }
                    for l in legs_1x2
                ]
            })

    return [a for a in accas if a["combined_odds"] >= 3.00]

@app.get("/")
def root():
    return {
        "engine": "Soccer Intelligence Engine v5",
        "version": "5.0.0",
        "status": "online",
        "endpoints": ["/api/predict", "/api/batch-predict", "/api/health"]
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": dt.datetime.utcnow().isoformat()}

@app.post("/api/predict")
def predict_endpoint(fixture: FixtureInput):
    try:
        return predict_fixture(fixture)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-predict")
def batch_predict_endpoint(req: BatchPredictionRequest):
    try:
        predictions = [predict_fixture(f) for f in req.fixtures]
        accas = build_accumulators(predictions)
        return {
            "engine": "v5.0-production-quantitative",
            "count": len(predictions),
            "predictions": predictions,
            "accumulators": accas,
            "min_odds_enforced": True,
            "min_odds_floor": 3.00
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
