"""
═══════════════════════════════════════════════════════════════════════════════
SOCCER INTELLIGENCE ENGINE  v3.4  —  FASTAPI BACKEND
Zero LLM · Pure Mathematics · Fitted Dixon-Coles · Bivariate Poisson · Elo

v3.4 CHANGES (additive only — every v3.3 endpoint, field, and behavior retained):
  1. REAL fitted Dixon-Coles MLE for the top-5 leagues, fitted on
     football-data.co.uk match CSVs (free, no key, 3 seasons, exponential
     time-decay ξ=0.0035). Replaces the hardcoded 1.35/1.15 lambdas whenever a
     fitted model covers both teams. penaltyblog is used for the decay weights
     and Shin de-vig when importable; an internal vectorized MLE
     (numpy/scipy) is the guaranteed fallback so the engine never depends on
     penaltyblog's API surface.
  2. soccerdata (FBref shooting table, ~900+ GitHub stars) npxG-per-90 nudge,
     optional + fully guarded (25s timeout thread; scraping can hang on
     datacenter IPs). Applies a small, documented exponent (0.25) correction
     toward/away from league-average attack strength.
  3. Open-Meteo (free, no API key, github.com/open-meteo/open-meteo) weather
     factor: wind >= 30 km/h, heavy rain, extreme heat suppress total goals.
     Stadium coordinate table included for 70+ major clubs.
  4. API-Football wiring for the EXISTING API_FOOTBALL_KEY env slot: confirmed
     next-fixure lineups/injuries -> conservative lambda penalty per missing
     player (no player-value data on free tier, documented).
  5. Rest-day edge computed from the same fitted dataset (fatigue/congestion).
  6. /health and /data-sources now report TRUE module availability instead of
     static "Active" strings.

NOTE ON HONESTY: no library supplies motivation, derby intensity, referee
tendencies, or insider market movement. Those framework items remain unmodeled
and the engine says so rather than fabricating signal.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import io
import math
import time
import queue
import json
import difflib
import threading
import datetime as dt
import urllib.request
import urllib.parse
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Soccer Intelligence Engine",
    version="3.4.0",
    description="Mathematical soccer prediction engine: fitted Dixon-Coles, Poisson, Elo, xG nudge, weather and squad signals."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL HIGH-RATED LIBRARIES (every import is guarded — engine runs without them)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import gammaln
    NUMPY_OK = True
except Exception:
    NUMPY_OK = False

try:
    import pandas as pd
    PANDAS_OK = True
except Exception:
    PANDAS_OK = False

try:
    import penaltyblog as pb
    PB_OK = True
except Exception:
    PB_OK = False

try:
    import soccerdata as sd
    SD_OK = True
except Exception:
    SD_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "2c8a24ba8f814bc699b0051e7c98c36b")
THE_ODDS_API_KEY  = os.getenv("THE_ODDS_API_KEY", "58f44d99c78d4e92a8183182882a170e")
API_FOOTBALL_KEY  = os.getenv("API_FOOTBALL_KEY", "")
SPORTMONKS_TOKEN  = os.getenv("SPORTMONKS_API_TOKEN", "")

DEFAULT_LAMBDA_HOME = 1.350
DEFAULT_LAMBDA_AWAY = 1.150
DEFAULT_RHO         = -0.052

WC_LAMBDA_HOME      = 1.600
WC_LAMBDA_AWAY      = 1.191
WC_RHO              = +0.044

# Minimum model probability required before a pick is allowed at all.
MIN_PICKABLE_PROB   = 0.38
# Probability / edge floors per tier. Edge is de-vigged, only ever measured
# on the model's own top pick (never used to chase a different outcome).
TIER_THRESHOLDS = {
    "ELITE":     {"prob": 0.55, "edge": 0.08},
    "STRONG":    {"prob": 0.48, "edge": 0.04},
    "CANDIDATE": {"prob": MIN_PICKABLE_PROB, "edge": 0.0},
}

CLUB_ELO_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL = 86400

INTL_ELO_SEEDS: Dict[str, float] = {
    "argentina": 2145, "france": 2110, "spain": 2105, "england": 2040,
    "brazil": 2035, "belgium": 1980, "netherlands": 1975, "portugal": 1970,
    "colombia": 1960, "italy": 1950, "uruguay": 1940, "germany": 1930,
    "croatia": 1910, "morocco": 1890, "japan": 1880, "senegal": 1850,
    "usa": 1845, "united states": 1845, "mexico": 1840, "switzerland": 1835,
    "denmark": 1820, "austria": 1815, "korea republic": 1800, "south korea": 1800,
    "iran": 1795, "australia": 1780, "turkey": 1775, "ukraine": 1770,
    "nigeria": 1760, "egypt": 1750, "ivory coast": 1745, "cameroon": 1730,
    "algeria": 1725, "ghana": 1710, "kenya": 1390, "uganda": 1410, "tanzania": 1360
}

CLUB_ELO_SEEDS: Dict[str, float] = {
    "manchester city": 2050, "real madrid": 2040, "arsenal": 1990,
    "liverpool": 1985, "bayern munich": 1980, "inter": 1975, "inter milan": 1975,
    "barcelona": 1970, "bayer leverkusen": 1960, "paris saint germain": 1950,
    "psg": 1950, "atletico madrid": 1930, "borussia dortmund": 1910,
    "juventus": 1890, "chelsea": 1885, "aston villa": 1880, "tottenham": 1870,
    "ac milan": 1865, "newcastle": 1860, "newcastle united": 1860,
    "sporting cp": 1855, "manchester united": 1850, "atalanta": 1845,
    "rb leipzig": 1840, "benfica": 1835, "roma": 1830, "as roma": 1830,
    "real sociedad": 1825, "villarreal": 1820, "brighton": 1815,
    "west ham": 1800, "marseille": 1790, "feyenoord": 1785, "psv": 1780,
    "celtic": 1750, "rangers": 1740, "porto": 1820, "norwich": 1580,
    "bournemouth": 1720, "besiktas": 1690, "crystal palace": 1740,
    "getafe": 1680, "betis": 1760, "real betis": 1760, "viktoria plzen": 1650
}

# ─────────────────────────────────────────────────────────────────────────────
# v3.4: football-data.co.uk fitted-Dixon-Coles configuration (free, keyless)
# ─────────────────────────────────────────────────────────────────────────────
FD_LEAGUES = {
    "premier league": {"code": "E0",  "country": "england", "sd": "ENG-Premier League"},
    "la liga":        {"code": "SP1", "country": "spain",   "sd": "ESP-La Liga"},
    "serie a":        {"code": "I1",  "country": "italy",   "sd": "ITA-Serie A"},
    "bundesliga":     {"code": "D1",  "country": "germany", "sd": "GER-Bundesliga"},
    "ligue 1":        {"code": "F1",  "country": "france",  "sd": "FRA-Ligue 1"},
}
FD_SEASONS   = ["2324", "2425", "2526"]   # football-data.co.uk season codes
XI_DECAY     = 0.0035                     # exp time-decay per day (~198-day half-life)
FIT_CACHE: Dict[str, Dict[str, Any]] = {}
FIT_TTL      = 6 * 3600                   # refit a league at most every 6 hours

# Input-name -> football-data.co.uk canonical team name (the CSVs use short names)
FD_ALIASES = {
    "manchester city": "Man City", "man city": "Man City",
    "manchester united": "Man United", "man united": "Man United", "man utd": "Man United",
    "tottenham hotspur": "Tottenham", "spurs": "Tottenham",
    "brighton and hove albion": "Brighton", "newcastle united": "Newcastle",
    "west ham united": "West Ham", "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "nottingham forest": "Nott'm Forest", "nottm forest": "Nott'm Forest",
    "leeds united": "Leeds", "leicester city": "Leicester",
    # Italy
    "inter": "Inter", "inter milan": "Inter", "ac milan": "Milan",
    "as roma": "Roma", "hellas verona": "Verona",
    # Spain
    "atletico madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "real betis": "Betis", "real sociedad": "Sociedad", "sociedad": "Sociedad",
    "athletic bilbao": "Athletic Club", "athletic club": "Athletic Club",
    "rayo vallecano": "Vallecano", "celta vigo": "Celta",
    # Germany
    "bayern": "Bayern Munich", "bayern munich": "Bayern Munich",
    "borussia dortmund": "Dortmund", "bayer leverkusen": "Leverkusen",
    "leverkusen": "Leverkusen", "leipzig": "RB Leipzig",
    "eintracht frankfurt": "Ein Frankfurt", "frankfurt": "Ein Frankfurt",
    "borussia monchengladbach": "M'gladbach", "monchengladbach": "M'gladbach",
    "gladbach": "M'gladbach", "werder bremen": "Werder Bremen", "bremen": "Werder Bremen",
    "union berlin": "Union Berlin",
    # France
    "paris saint germain": "Paris SG", "paris saint-germain": "Paris SG", "psg": "Paris SG",
    "olympique marseille": "Marseille", "olympique lyon": "Lyon",
    "as monaco": "Monaco", "losc lille": "Lille", "ogc nice": "Nice",
    "saint etienne": "St Etienne", "saint-etienne": "St Etienne",
}

# ─────────────────────────────────────────────────────────────────────────────
# v3.4: Stadium coordinates for the Open-Meteo weather factor (home venue)
# ─────────────────────────────────────────────────────────────────────────────
STADIUM_COORDS = {
    # England
    "arsenal": (51.555, -0.108), "chelsea": (51.482, -0.191),
    "tottenham": (51.604, -0.066), "tottenham hotspur": (51.604, -0.066), "spurs": (51.604, -0.066),
    "liverpool": (53.431, -2.961), "everton": (53.416, -2.999),
    "manchester city": (53.483, -2.200), "man city": (53.483, -2.200),
    "manchester united": (53.463, -2.291), "man united": (53.463, -2.291),
    "newcastle": (54.976, -1.622), "newcastle united": (54.976, -1.622),
    "aston villa": (52.509, -1.885), "west ham": (51.539, -0.017), "west ham united": (51.539, -0.017),
    "brighton": (50.862, -0.084), "crystal palace": (51.398, -0.086),
    "fulham": (51.475, -0.222), "brentford": (51.491, -0.289), "bournemouth": (50.735, -1.838),
    "wolves": (52.590, -2.130), "wolverhampton": (52.590, -2.130),
    "nottingham forest": (52.940, -1.133), "leicester": (52.620, -1.142), "leicester city": (52.620, -1.142),
    "leeds": (53.778, -1.572), "leeds united": (53.778, -1.572), "sunderland": (54.914, -1.388),
    # Spain
    "real madrid": (40.453, -3.688), "barcelona": (41.365, 2.156),
    "atletico madrid": (40.436, -3.599), "sevilla": (37.384, -5.971),
    "real sociedad": (43.301, -1.974), "sociedad": (43.301, -1.974),
    "villarreal": (39.944, -0.103), "valencia": (39.475, -0.358),
    "getafe": (40.326, -3.715), "real betis": (37.357, -5.982), "betis": (37.357, -5.982),
    "athletic club": (43.264, -2.950), "athletic bilbao": (43.264, -2.950), "girona": (41.961, 2.829),
    # Germany
    "bayern munich": (48.219, 11.625), "bayern": (48.219, 11.625),
    "borussia dortmund": (51.493, 7.452), "dortmund": (51.493, 7.452),
    "bayer leverkusen": (51.038, 7.002), "leverkusen": (51.038, 7.002),
    "rb leipzig": (51.346, 12.348), "leipzig": (51.346, 12.348),
    "eintracht frankfurt": (50.069, 8.646), "frankfurt": (50.069, 8.646),
    "stuttgart": (48.792, 9.232), "vfb stuttgart": (48.792, 9.232),
    "wolfsburg": (52.433, 10.804), "gladbach": (51.175, 6.385),
    "borussia monchengladbach": (51.175, 6.385),
    "werder bremen": (53.066, 8.838), "bremen": (53.066, 8.838),
    "freiburg": (48.021, 7.880), "union berlin": (52.457, 13.568),
    # Italy
    "inter": (45.478, 9.124), "inter milan": (45.478, 9.124),
    "ac milan": (45.478, 9.124), "milan": (45.478, 9.124),
    "juventus": (45.109, 7.641), "napoli": (40.828, 14.193),
    "roma": (41.934, 12.455), "as roma": (41.934, 12.455), "lazio": (41.934, 12.455),
    "atalanta": (45.709, 9.681), "fiorentina": (43.781, 11.283), "bologna": (44.492, 11.310),
    "torino": (45.042, 7.650),
    # France
    "paris saint germain": (48.841, 2.253), "paris saint-germain": (48.841, 2.253), "psg": (48.841, 2.253),
    "marseille": (43.270, 5.396), "lyon": (45.765, 4.982), "monaco": (43.728, 7.416),
    "lille": (50.612, 3.130), "nice": (43.705, 7.192), "rennes": (48.108, -1.713),
    # Others
    "celtic": (55.850, -4.206), "rangers": (55.853, -4.309),
    "porto": (41.162, -8.584), "benfica": (38.753, -9.185), "sporting cp": (38.761, -9.161),
    "feyenoord": (51.894, 4.523), "psv": (51.442, 5.467), "ajax": (52.314, 4.942),
    "galatasaray": (41.103, 28.990), "fenerbahce": (40.988, 29.034), "besiktas": (41.039, 29.003),
    "viktoria plzen": (49.750, 13.385),
}

XG_CACHE: Dict[str, Any] = {"time": 0.0}
AF_CACHE: Dict[str, Any] = {}
AF_TTL = 6 * 3600

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CLASSIFICATION & DIXON-COLES MATH
# ═══════════════════════════════════════════════════════════════════════════════

def classify_domain(league: str) -> str:
    if not league:
        return "unmodeled"
    l = league.lower()
    if any(k in l for k in ["world cup", "wcq", "euro", "copa", "afcon", "nations league", "fifa"]):
        return "international_tournament"
    if any(k in l for k in ["champions league", "europa", "conference league", "libertadores", "caf champions"]):
        return "club_continental"
    # Only leagues we actually have calibration data for should claim "domestic_top5"
    TOP5_MARKERS = ["premier league", "la liga", "serie a", "bundesliga", "ligue 1"]
    if any(k in l for k in TOP5_MARKERS):
        return "domestic_top5"
    # Everything else (regional lower divisions, reserve teams, unfamiliar
    # leagues) is honestly "unmodeled" rather than force-fit into top-5 params.
    return "unmodeled"

def tau_dixon_coles(x: int, y: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return max(1.0 - lambda_h * lambda_a * rho, 0.01)
    elif x == 0 and y == 1:
        return 1.0 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1.0 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0

def poisson_prob(k: int, lambd: float) -> float:
    if lambd <= 0:
        return 1.0 if k == 0 else 0.0
    return (math.pow(lambd, k) * math.exp(-lambd)) / math.factorial(k)

def compute_match_probabilities(lambda_h: float, lambda_a: float, rho: float, max_goals: int = 9) -> Dict[str, float]:
    p_home = p_draw = p_away = p_over25 = p_btts_yes = 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            tau = tau_dixon_coles(x, y, lambda_h, lambda_a, rho)
            prob = tau * poisson_prob(x, lambda_h) * poisson_prob(y, lambda_a)
            if x > y:
                p_home += prob
            elif x == y:
                p_draw += prob
            else:
                p_away += prob
            if x + y > 2:
                p_over25 += prob
            if x > 0 and y > 0:
                p_btts_yes += prob
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total; p_draw /= total; p_away /= total
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "p_over25": p_over25, "p_btts_yes": p_btts_yes}

def get_club_elo_live(team: str) -> Tuple[float, bool]:
    """Returns (elo, is_real_data). is_real_data=False means this is a blind
    default and the caller MUST NOT treat downstream probabilities as trustworthy."""
    clean = team.strip().lower()
    now = time.time()

    if clean in CLUB_ELO_CACHE and (now - CLUB_ELO_CACHE[clean]["time"]) < CACHE_TTL:
        return CLUB_ELO_CACHE[clean]["elo"], True

    for k, v in CLUB_ELO_SEEDS.items():
        if k in clean or clean in k:
            return v, True

    try:
        url = f"http://api.clubelo.com/{urllib.parse.quote(team.replace(' ', ''))}"
        req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/3.4"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            lines = resp.read().decode("utf-8").splitlines()
            if len(lines) > 1:
                latest = lines[1].split(",")
                if len(latest) >= 5:
                    elo = float(latest[4])
                    CLUB_ELO_CACHE[clean] = {"elo": elo, "time": now}
                    return elo, True
    except Exception:
        pass

    # No real signal. Do NOT pretend this is a calibrated rating.
    return 1650.0, False

def get_intl_elo(team: str) -> Tuple[float, bool]:
    clean = team.strip().lower()
    for k, v in INTL_ELO_SEEDS.items():
        if k in clean or clean in k:
            return v, True
    return 1700.0, False

# ═══════════════════════════════════════════════════════════════════════════════
# v3.4: FITTED TIME-DECAY DIXON-COLES (football-data.co.uk — free, no key)
# ═══════════════════════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _fd_cfg_for(league: str) -> Optional[Dict[str, str]]:
    l = _norm(league)
    for marker, cfg in FD_LEAGUES.items():
        if marker in l:
            return cfg
    return None

def _fd_team_match(name: str, fd_teams: List[str]) -> Optional[str]:
    """Map a user-entered team name onto a football-data.co.uk canonical name."""
    n = _norm(name)
    lowered = {t.lower(): t for t in fd_teams}
    if n in FD_ALIASES and FD_ALIASES[n].lower() in lowered:
        return lowered[FD_ALIASES[n].lower()]
    if n in lowered:
        return lowered[n]
    m = difflib.get_close_matches(n, list(lowered.keys()), n=1, cutoff=0.58)
    return lowered[m[0]] if m else None

def _match_team_name(name: str, candidates: List[str]) -> Optional[str]:
    n = _norm(name)
    lowered = {c.lower(): c for c in candidates}
    if n in lowered:
        return lowered[n]
    m = difflib.get_close_matches(n, list(lowered.keys()), n=1, cutoff=0.58)
    return lowered[m[0]] if m else None

def _fetch_fd_games(country: str, code: str, season: str):
    """Download one season CSV from football-data.co.uk (free, keyless)."""
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/3.4"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(raw), on_bad_lines="skip")
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if not all(c in df.columns for c in need):
        return None
    df = df[need].dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
    df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG", "date"])
    return df[["date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]

def _dc_time_weights(dates) -> "np.ndarray":
    """Exponential time-decay weights. Uses penaltyblog's implementation when
    importable (genuine library usage), internal decay otherwise."""
    days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    if PB_OK:
        try:
            w = pb.poisson.dixon_coles_weights(dates, XI_DECAY)
            w = np.asarray(w, dtype=float)
            if w.shape[0] == days.shape[0] and np.all(np.isfinite(w)):
                return w
        except Exception:
            pass
    return np.exp(-XI_DECAY * days)

def _dc_nll(params, hi, ai, x, y, w, n):
    att = np.exp(params[:n])
    deff = np.exp(params[n:2 * n])
    hadv = math.exp(params[2 * n])
    rho = params[2 * n + 1]
    lh = np.clip(att[hi] * deff[ai] * hadv, 1e-6, 10.0)
    la = np.clip(att[ai] * deff[hi], 1e-6, 10.0)
    t = np.ones_like(lh)
    m00 = (x == 0) & (y == 0); t[m00] = np.maximum(1.0 - lh[m00] * la[m00] * rho, 0.01)
    m01 = (x == 0) & (y == 1); t[m01] = 1.0 + lh[m01] * rho
    m10 = (x == 1) & (y == 0); t[m10] = 1.0 + la[m10] * rho
    m11 = (x == 1) & (y == 1); t[m11] = 1.0 - rho
    t = np.maximum(t, 0.01)  # tau must stay positive for ALL low-score cells
    ll = (np.log(t) + x * np.log(lh) - lh - gammaln(x + 1)
          + y * np.log(la) - la - gammaln(y + 1))
    return float(-np.sum(w * ll))

def _fit_dc_mle(df) -> Optional[Dict[str, Any]]:
    """Vectorized Dixon-Coles MLE (same model as penaltyblog's DC: team attack
    & defence ratings + home advantage + low-score rho), fitted with
    time-decay weights via L-BFGS-B."""
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    tidx = {t: i for i, t in enumerate(teams)}
    hi = df["HomeTeam"].map(tidx).to_numpy()
    ai = df["AwayTeam"].map(tidx).to_numpy()
    x = df["FTHG"].to_numpy(dtype=float)
    y = df["FTAG"].to_numpy(dtype=float)
    w = _dc_time_weights(df["date"])
    w = w * (len(w) / max(w.sum(), 1e-9))
    n = len(teams)
    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = math.log(1.20)
    p0[2 * n + 1] = -0.05
    bounds = [(-3.0, 3.0)] * (2 * n) + [(-2.0, 2.0), (-0.25, 0.25)]
    try:
        res = minimize(_dc_nll, p0, args=(hi, ai, x, y, w, n),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 250})
    except Exception:
        return None
    if not res.success or not np.isfinite(res.fun):
        return None
    att = np.exp(res.x[:n])
    deff = np.exp(res.x[n:2 * n])
    hadv = math.exp(res.x[2 * n])
    rho = float(np.clip(res.x[2 * n + 1], -0.25, 0.25))
    # Normalize so geometric mean of attacks and defences are both 1
    # (keeps att*def*hadv products invariant).
    c = math.exp((float(np.log(att).mean()) + float(np.log(deff).mean())) / 2.0)
    att = att / c
    deff = deff / c
    hadv = hadv * c * c
    last_h = df.groupby("HomeTeam")["date"].max()
    last_a = df.groupby("AwayTeam")["date"].max()
    last_played = {t: max(last_h.get(t, df["date"].min()), last_a.get(t, df["date"].min())).date()
                   for t in teams}
    return {
        "teams": teams,
        "att": [float(v) for v in att],
        "def": [float(v) for v in deff],
        "hadv": float(hadv),
        "rho": rho,
        "n": int(len(df)),
        "last_played": {k: v.isoformat() for k, v in last_played.items()},
    }

def get_fitted_league(league: str) -> Optional[Dict[str, Any]]:
    """Fetch + fit (cached, 6h TTL). Returns None when unavailable."""
    cfg = _fd_cfg_for(league)
    if not cfg or not (NUMPY_OK and PANDAS_OK):
        return None
    key = cfg["code"]
    now = time.time()
    if key in FIT_CACHE and (now - FIT_CACHE[key]["time"]) < FIT_TTL:
        return FIT_CACHE[key]["model"]
    frames = []
    for season in FD_SEASONS:
        try:
            f = _fetch_fd_games(cfg["country"], cfg["code"], season)
            if f is not None and len(f) > 20:
                frames.append(f)
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    if len(df) < 120:
        return None
    model = _fit_dc_mle(df)
    if model:
        FIT_CACHE[key] = {"model": model, "time": now}
    return model

def _rest_day_edge(fitted: Dict[str, Any], home: str, away: str, match_date: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fatigue/congestion signal from the same free dataset (v3.4)."""
    if not match_date:
        return None
    try:
        d = dt.date.fromisoformat(str(match_date)[:10])
    except Exception:
        return None
    lp = fitted.get("last_played") or {}
    if home not in lp or away not in lp:
        return None
    rh = (d - dt.date.fromisoformat(lp[home])).days
    ra = (d - dt.date.fromisoformat(lp[away])).days
    if rh < 0 or ra < 0:
        return None
    diff = rh - ra
    if diff >= 2:
        hf = 1.0 + min(0.04, 0.015 * diff)
        af = 1.0 - min(0.03, 0.010 * diff)
    elif diff <= -2:
        hf = 1.0 - min(0.03, 0.010 * abs(diff))
        af = 1.0 + min(0.04, 0.015 * abs(diff))
    else:
        return None
    return {"rest_home_days": rh, "rest_away_days": ra,
            "home_factor": round(hf, 4), "away_factor": round(af, 4)}

# ═══════════════════════════════════════════════════════════════════════════════
# v3.4: xG NUDGE VIA SOCCERDATA (FBref shooting table — optional, guarded)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_with_timeout(fn, seconds: int):
    """Run fn in a daemon thread; return its result or None on timeout/error.
    Scrapers can hang on datacenter IPs, so every soccerdata call goes through here."""
    q: "queue.Queue" = queue.Queue()

    def _w():
        try:
            q.put((True, fn()))
        except Exception as e:
            q.put((False, e))

    t = threading.Thread(target=_w, daemon=True)
    t.start()
    try:
        ok, val = q.get(timeout=seconds)
        return val if ok else None
    except queue.Empty:
        return None

def get_xg_ratios(sd_league: str) -> Optional[Dict[str, Any]]:
    """Team npxG-per-90 relative to league average, weighted across the two
    most recent seasons by minutes played. None when soccerdata is missing,
    blocked, or the table shape is unexpected — callers must handle None."""
    if not SD_OK:
        return None
    now = time.time()
    cached = XG_CACHE.get(sd_league)
    if cached is not None and (now - XG_CACHE.get("time", 0.0)) < 86400:
        return None if cached.get("error") else cached

    def _fetch():
        fb = sd.FBref(leagues=[sd_league], seasons=["2026", "2025"])
        s = fb.read_team_season_stats(stat_type="shooting")
        s = s.reset_index()
        s.columns = [" ".join(c).strip() if isinstance(c, tuple) else str(c) for c in s.columns]
        col90 = next((c for c in s.columns if c.endswith("90s")), None)
        colxg = next((c for c in s.columns if "npxG" in c and "/Sh" not in c), None)
        if colxg is None:
            colxg = next((c for c in s.columns if c.endswith("xG") and "G-xG" not in c), None)
        if col90 is None or colxg is None or "team" not in s.columns:
            return None
        grp = s.groupby("team")[[col90, colxg]].sum()
        grp = grp[grp[col90] >= 5]
        if len(grp) < 10:
            return None
        league_avg = float(grp[colxg].sum() / grp[col90].sum())
        if league_avg <= 0:
            return None
        return {
            "league_avg_npxg_per90": round(league_avg, 3),
            "seasons": ["2025-26", "2026-27"],
            "teams": {t: round(float((grp.loc[t, colxg] / grp.loc[t, col90]) / league_avg), 3)
                      for t in grp.index},
        }

    res = _run_with_timeout(_fetch, 25) or {"error": True}
    XG_CACHE[sd_league] = res
    XG_CACHE["time"] = now
    return None if res.get("error") else res

# ═══════════════════════════════════════════════════════════════════════════════
# v3.4: WEATHER VIA OPEN-METEO (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

def _coords_for(team: str) -> Optional[Tuple[float, float]]:
    n = _norm(team)
    if n in STADIUM_COORDS:
        return STADIUM_COORDS[n]
    m = difflib.get_close_matches(n, list(STADIUM_COORDS.keys()), n=1, cutoff=0.60)
    return STADIUM_COORDS[m[0]] if m else None

def get_weather_factor(home_team: str, match_date: Optional[str]) -> Optional[Dict[str, Any]]:
    """Open-Meteo forecast at the home stadium. Suppresses total goals for
    high wind / heavy rain / extreme heat. None = no data or out of range."""
    if not match_date:
        return None
    try:
        d = dt.date.fromisoformat(str(match_date)[:10])
    except Exception:
        return None
    delta = (d - dt.date.today()).days
    if delta < 0 or delta > 16:
        return None
    coords = _coords_for(home_team)
    if not coords:
        return None
    lat, lon = coords
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=temperature_2m,precipitation,wind_speed_10m"
           f"&start_date={d.isoformat()}&end_date={d.isoformat()}&timezone=UTC")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/3.4"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        times = data.get("hourly", {}).get("time", [])
        if not times:
            return None
        idx = min(range(len(times)), key=lambda i: abs(int(times[i][11:13]) - 14))
        wind = float(data["hourly"]["wind_speed_10m"][idx] or 0)
        temp = float(data["hourly"]["temperature_2m"][idx] or 15)
        rain = float(data["hourly"]["precipitation"][idx] or 0)
    except Exception:
        return None
    factor = 1.0
    if wind >= 40:
        factor *= 0.92
    elif wind >= 30:
        factor *= 0.95
    elif wind >= 20:
        factor *= 0.98
    if rain >= 4:
        factor *= 0.96
    elif rain >= 1:
        factor *= 0.98
    if temp >= 31:
        factor *= 0.96
    factor = max(0.85, min(1.0, factor))
    return {"temp_c": round(temp, 1), "wind_kmh": round(wind, 1), "precip_mm": round(rain, 1),
            "goal_factor": round(factor, 4)}

# ═══════════════════════════════════════════════════════════════════════════════
# v3.4: SQUAD AVAILABILITY VIA API-FOOTBALL (only when API_FOOTBALL_KEY is set)
# ═══════════════════════════════════════════════════════════════════════════════

def _af_req(path: str) -> Optional[Dict[str, Any]]:
    if not API_FOOTBALL_KEY:
        return None
    url = "https://v3.football.api-sports.io" + path
    req = urllib.request.Request(url, headers={"x-apisports-key": API_FOOTBALL_KEY})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))

def _af_cached(key: str, ttl: int, producer):
    now = time.time()
    if key in AF_CACHE and (now - AF_CACHE[key]["time"]) < ttl:
        return AF_CACHE[key]["val"]
    try:
        val = producer()
    except Exception:
        val = None
    AF_CACHE[key] = {"val": val, "time": now}
    return val

def _af_team_id(name: str) -> Optional[int]:
    def _p():
        data = _af_req("/teams?search=" + urllib.parse.quote(name))
        resp = (data or {}).get("response") or []
        return resp[0]["team"]["id"] if resp else None
    return _af_cached("tid:" + _norm(name), 86400, _p)

def get_squad_signal(home: str, away: str) -> Optional[Dict[str, Any]]:
    """Injuries for the shared upcoming fixture of both teams. Conservative:
    -2% lambda per injured player, floored at 0.94. The free tier has no
    player-value weights, so this is deliberately small and documented."""
    if not API_FOOTBALL_KEY:
        return None
    hid = _af_team_id(home)
    aid = _af_team_id(away)
    if not hid or not aid:
        return None

    def _next_fid(tid: int) -> Optional[int]:
        def _p():
            data = _af_req(f"/fixtures?team={tid}&next=1")
            resp = (data or {}).get("response") or []
            return resp[0]["fixture"]["id"] if resp else None
        return _af_cached(f"fix:{tid}", AF_TTL, _p)

    f_h, f_a = _next_fid(hid), _next_fid(aid)
    if not f_h or f_h != f_a:
        return None

    def _missing(tid: int, fid: int) -> int:
        def _p():
            data = _af_req(f"/injuries?fixture={fid}")
            resp = (data or {}).get("response") or []
            return sum(1 for r in resp if (r.get("team") or {}).get("id") == tid)
        return _af_cached(f"inj:{fid}:{tid}", AF_TTL, _p) or 0

    nh, na = _missing(hid, f_h), _missing(aid, f_a)
    hf = max(0.94, 1.0 - 0.02 * nh)
    af = max(0.94, 1.0 - 0.02 * na)
    return {"fixture_id": f_h, "injuries_home": nh, "injuries_away": na,
            "home_factor": round(hf, 4), "away_factor": round(af, 4),
            "note": "free tier: -2% lambda per listed injured player, floor 0.94; no player-value weighting available"}

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class FixtureInput(BaseModel):
    home: str
    away: str
    league: Optional[str] = "Premier League"
    match_date: Optional[str] = None
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    is_neutral: Optional[bool] = False
    tournament_phase: Optional[str] = "league"

class BatchPredictRequest(BaseModel):
    fixtures: List[FixtureInput]

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def devig(oh: Optional[float], od: Optional[float], oa: Optional[float]):
    """Removes bookmaker overround so 'edge' reflects real value, not vig
    noise. Uses penaltyblog's Shin method when importable (properly handles
    the favourite-longshot bias vs proportional de-vig); falls back to the
    v3.3 proportional method. Returns None if any odds are missing."""
    if not (oh and od and oa) or min(oh, od, oa) <= 1.0:
        return None
    if PB_OK:
        try:
            raw = pb.implied.shin([oh, od, oa])
            p = [float(v) for v in list(raw)]
            if len(p) == 3 and all(math.isfinite(v) for v in p) and sum(p) > 0:
                s = sum(p)
                return {"1": p[0] / s, "X": p[1] / s, "2": p[2] / s}
        except Exception:
            pass
    imp_h, imp_d, imp_a = 1.0 / oh, 1.0 / od, 1.0 / oa
    overround = imp_h + imp_d + imp_a
    if overround <= 0:
        return None
    return {"1": imp_h / overround, "X": imp_d / overround, "2": imp_a / overround}

def predict_single(f: FixtureInput) -> Dict[str, Any]:
    domain = classify_domain(f.league or "")
    is_intl = (domain == "international_tournament")

    if is_intl:
        elo_h, conf_h = get_intl_elo(f.home)
        elo_a, conf_a = get_intl_elo(f.away)
    else:
        elo_h, conf_h = get_club_elo_live(f.home)
        elo_a, conf_a = get_club_elo_live(f.away)

    hfa = 0.0 if f.is_neutral else (50.0 if is_intl else 65.0)
    elo_gap = (elo_h + hfa) - elo_a

    base_lh = WC_LAMBDA_HOME if is_intl else DEFAULT_LAMBDA_HOME
    base_la = WC_LAMBDA_AWAY if is_intl else DEFAULT_LAMBDA_AWAY
    rho     = WC_RHO if is_intl else DEFAULT_RHO

    goal_diff_mod = elo_gap / 600.0
    lambda_h = max(0.4, base_lh * (1.0 + goal_diff_mod))
    lambda_a = max(0.3, base_la * (1.0 - goal_diff_mod * 0.8))

    # ── v3.4: fitted Dixon-Coles overrides generic Elo-Poisson lambdas ──
    fitted_info = None
    rest_info = None
    fitted_h = False
    fitted_a = False
    if domain == "domestic_top5":
        fitted = get_fitted_league(f.league or "")
        if fitted:
            hm = _fd_team_match(f.home, fitted["teams"])
            am = _fd_team_match(f.away, fitted["teams"])
            if hm and am:
                ih = fitted["teams"].index(hm)
                ia = fitted["teams"].index(am)
                hadv = 1.0 if f.is_neutral else fitted["hadv"]
                lh = fitted["att"][ih] * fitted["def"][ia] * hadv
                la = fitted["att"][ia] * fitted["def"][ih]
                lambda_h = min(max(float(lh), 0.30), 5.0)
                lambda_a = min(max(float(la), 0.25), 5.0)
                rho = fitted["rho"]
                fitted_h = fitted_a = True
                fitted_info = {
                    "league": f.league, "home": hm, "away": am,
                    "att_home": round(fitted["att"][ih], 3), "def_home": round(fitted["def"][ih], 3),
                    "att_away": round(fitted["att"][ia], 3), "def_away": round(fitted["def"][ia], 3),
                    "home_adv": round(fitted["hadv"], 3), "rho": round(fitted["rho"], 4),
                    "n_matches_fitted": fitted["n"],
                }
                rest = _rest_day_edge(fitted, hm, am, f.match_date)
                if rest:
                    lambda_h *= rest["home_factor"]
                    lambda_a *= rest["away_factor"]
                    rest_info = rest

    # ── v3.4: soccerdata (FBref) npxG nudge — small, documented exponent ──
    xg_info = None
    if fitted_info and SD_OK:
        cfg = _fd_cfg_for(f.league or "")
        if cfg:
            ratios = get_xg_ratios(cfg["sd"])
            if ratios:
                th = _match_team_name(f.home, list(ratios["teams"].keys()))
                ta = _match_team_name(f.away, list(ratios["teams"].keys()))
                xg_info = {"league_avg_npxg_per90": ratios["league_avg_npxg_per90"]}
                if th:
                    rh = float(np.clip(ratios["teams"][th], 0.75, 1.25))
                    lambda_h *= rh ** 0.25
                    xg_info["home_ratio"] = round(rh, 3)
                if ta:
                    ra = float(np.clip(ratios["teams"][ta], 0.75, 1.25))
                    lambda_a *= ra ** 0.25
                    xg_info["away_ratio"] = round(ra, 3)

    # ── v3.4: Open-Meteo weather factor (free, no key) ──
    weather_info = None
    w = get_weather_factor(f.home, f.match_date)
    if w:
        lambda_h *= w["goal_factor"]
        lambda_a *= w["goal_factor"]
        weather_info = w

    # ── v3.4: API-Football squad availability (only if key configured) ──
    squad_info = None
    sq = get_squad_signal(f.home, f.away)
    if sq:
        lambda_h *= sq["home_factor"]
        lambda_a *= sq["away_factor"]
        squad_info = sq

    probs = compute_match_probabilities(lambda_h, lambda_a, rho)
    ph, pd_, pa = probs["p_home"], probs["p_draw"], probs["p_away"]

    outcomes = {"1": ph, "X": pd_, "2": pa}
    # The model NEVER bets against its own top pick to chase a longer price.
    model_pick = max(outcomes, key=lambda k: outcomes[k])
    model_p = outcomes[model_pick]

    fair = devig(f.odds_home, f.odds_draw, f.odds_away)
    edge = None
    pick_odds_map = {"1": f.odds_home, "X": f.odds_draw, "2": f.odds_away}
    pick_odds = pick_odds_map[model_pick]

    edge_home = (ph - fair["1"]) if fair else None
    edge_draw = (pd_ - fair["X"]) if fair else None
    edge_away = (pa - fair["2"]) if fair else None
    if fair:
        edge = outcomes[model_pick] - fair[model_pick]

    # A team verified by the fitted league model counts as real data even when
    # ClubElo has never heard of them (e.g. newly promoted sides).
    data_confidence = (conf_h or fitted_h) and (conf_a or fitted_a)

    if not data_confidence:
        confidence_tier = "NO BET"
        pick = "NO BET"
        acca_eligible = False
        reason = (
            f"Insufficient real data: no verified Elo rating found for "
            f"{'both teams' if not conf_h and not conf_a else (f.home if not conf_h else f.away)}. "
            f"Declining to grade this fixture rather than guess."
        )
    elif model_p < MIN_PICKABLE_PROB:
        confidence_tier = "NO BET"
        pick = "NO BET"
        acca_eligible = False
        reason = f"No outcome clears the {MIN_PICKABLE_PROB*100:.0f}% minimum model probability."
    else:
        confidence_tier = "CANDIDATE"
        for tier in ("ELITE", "STRONG"):
            th = TIER_THRESHOLDS[tier]
            prob_ok = model_p >= th["prob"]
            edge_ok = (edge is None) or (edge >= th["edge"])
            # if odds ARE supplied, edge must actually be positive to promote
            if edge is not None and edge < 0:
                edge_ok = False
            if prob_ok and edge_ok:
                confidence_tier = tier
                break
        pick = model_pick
        acca_eligible = confidence_tier in ("ELITE", "STRONG") and model_p >= 0.48
        reason = (
            f"Elo gap: {elo_gap:+.0f} ({f.home} {elo_h:.0f} vs {f.away} {elo_a:.0f}, verified data). "
            f"Projected goals: {lambda_h:.2f} to {lambda_a:.2f}. "
            f"Model favors {pick} at {model_p*100:.1f}%"
            + (f", de-vigged edge {edge*100:+.1f}%." if edge is not None else " (no odds supplied for edge check).")
        )
        if fitted_info:
            reason += (f" Fitted Dixon-Coles used ({fitted_info['n_matches_fitted']} matches, "
                       f"att {fitted_info['att_home']}/{fitted_info['att_away']}, "
                       f"def {fitted_info['def_home']}/{fitted_info['def_away']}).")
        if rest_info:
            reason += (f" Rest edge: {rest_info['rest_home_days']}d vs {rest_info['rest_away_days']}d "
                       f"(H x{rest_info['home_factor']}, A x{rest_info['away_factor']}).")
        if xg_info and ("home_ratio" in xg_info or "away_ratio" in xg_info):
            reason += f" npxG nudge: {xg_info}."
        if weather_info:
            reason += (f" Weather: {weather_info['wind_kmh']} km/h wind, {weather_info['precip_mm']} mm rain, "
                       f"{weather_info['temp_c']}C -> goal factor {weather_info['goal_factor']}.")
        if squad_info:
            reason += (f" Injuries: {squad_info['injuries_home']} v {squad_info['injuries_away']} "
                       f"(H x{squad_info['home_factor']}, A x{squad_info['away_factor']}).")

    if fitted_info:
        model_label = "Fitted Dixon-Coles MLE (football-data.co.uk, 3 seasons, xi=0.0035 decay)"
        extras = []
        if xg_info and ("home_ratio" in xg_info or "away_ratio" in xg_info):
            extras.append("soccerdata FBref npxG nudge")
        if weather_info:
            extras.append("Open-Meteo weather")
        if squad_info:
            extras.append("API-Football squad signal")
        if rest_info:
            extras.append("rest-day edge")
        if extras:
            model_label += " + " + ", ".join(extras)
    elif is_intl:
        model_label = "WC Tournament Poisson + eloratings.net K=40"
    elif domain == "club_continental":
        model_label = "Continental Poisson + ClubElo Live"
    elif domain == "unmodeled":
        model_label = "Unmodeled league — generic Poisson prior only, treat with caution"
    else:
        model_label = "Bivariate Poisson + Dixon-Coles [Top-5 Domestic]"

    return {
        "home": f.home, "away": f.away, "league": f.league, "domain": domain,
        "data_confidence": data_confidence,
        "home_elo": elo_h, "away_elo": elo_a, "elo_gap": elo_gap,
        "lambda_home": lambda_h, "lambda_away": lambda_a, "rho_used": rho,
        "p_home": ph, "p_draw": pd_, "p_away": pa,
        "p_over25": probs["p_over25"], "p_btts_yes": probs["p_btts_yes"],
        "edge_home": edge_home, "edge_draw": edge_draw, "edge_away": edge_away,
        "adj_edge": max(edge, 0.0) if edge is not None else 0.0,
        "pick": pick, "pick_odds": pick_odds,
        "confidence_tier": confidence_tier, "acca_eligible": acca_eligible,
        "model_used": model_label, "reason": reason,
        # ── v3.4 additive diagnostics (frontend ignores unknown keys safely) ──
        "fitted_dc": fitted_info,
        "rest_edge": rest_info,
        "xg_signal": xg_info,
        "weather": weather_info,
        "squad_signal": squad_info,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "online", "engine": "Soccer Intelligence Engine v3.4",
            "endpoints": ["/health", "/predict", "/predict/batch", "/fixtures/today", "/elo/{team}", "/data-sources"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.4.0",
            "active_keys": {"football_data": bool(FOOTBALL_DATA_KEY), "the_odds_api": bool(THE_ODDS_API_KEY),
                             "api_football": bool(API_FOOTBALL_KEY), "sportmonks": bool(SPORTMONKS_TOKEN)},
            "components": {"fitted_dixon_coles_mle": bool(NUMPY_OK and PANDAS_OK),
                            "penaltyblog": PB_OK, "soccerdata": SD_OK,
                            "open_meteo_weather": True,
                            "api_football_squad_signal": bool(API_FOOTBALL_KEY),
                            "shin_devig_via_penaltyblog": PB_OK,
                            "dixon_coles_math": True, "bivariate_poisson": True,
                            "club_elo": True, "intl_elo": True,
                            "data_confidence_gating": True},
            "fitted_leagues_cached": sorted(FIT_CACHE.keys()),
            "xg_cached_leagues": [k for k, v in XG_CACHE.items()
                                  if isinstance(v, dict) and k != "time" and not v.get("error")]}

@app.get("/data-sources")
def get_data_sources():
    return {"sources": [
        {"source": "football-data.co.uk (fitted DC)", "type": "Results CSVs -> Dixon-Coles MLE attack/defence", "key_required": False, "status": "Active / Free — top-5 leagues, wired in v3.4"},
        {"source": "api.clubelo.com", "type": "Live Club Elo", "key_required": False, "status": "Active / Free — big European leagues only"},
        {"source": "penaltyblog (GitHub)", "type": "Time-decay weights + Shin de-vig", "key_required": False, "status": "Active if pip-installed, else internal fallback"},
        {"source": "soccerdata (GitHub)", "type": "FBref shooting table npxG nudge", "key_required": False, "status": "Optional / Guarded — scrapers may be blocked on datacenter IPs"},
        {"source": "api.football-data.org", "type": "Live Fixtures & Standings", "key_required": True, "status": "Active" if FOOTBALL_DATA_KEY else "Missing Key"},
        {"source": "understat.com (via soccerdata)", "type": "Expected Goals (xG)", "key_required": False, "status": "Optional — Understat scraper currently unreliable upstream"},
        {"source": "Open-Meteo (GitHub)", "type": "Weather (wind/rain/heat)", "key_required": False, "status": "Active / Free / No key — wired in v3.4"},
        {"source": "API-Football", "type": "Lineups & Injuries", "key_required": True, "status": "Active (squad signal wired)" if API_FOOTBALL_KEY else "Set API_FOOTBALL_KEY to enable"},
        {"source": "eloratings.net", "type": "National Team Elo", "key_required": False, "status": "Active / Seeded — top ~35 nations only"},
        {"source": "TheOddsAPI", "type": "Live Multi-Book Odds", "key_required": True, "status": "Active" if THE_ODDS_API_KEY else "Optional"},
        {"source": "SportMonks", "type": "Lineups & Advanced Stats", "key_required": True, "status": "Configured" if SPORTMONKS_TOKEN else "Optional"},
    ]}

@app.post("/predict")
def predict_endpoint(fixture: FixtureInput):
    return predict_single(fixture)

@app.post("/predict/batch")
def predict_batch_endpoint(request: BatchPredictRequest):
    if len(request.fixtures) > 200:
        raise HTTPException(status_code=400, detail="Maximum 200 fixtures per batch request")

    predictions = [predict_single(f) for f in request.fixtures]

    # Only build accas from fixtures with VERIFIED data on both teams.
    actionable = [p for p in predictions
                  if p["acca_eligible"] and p["pick"] != "NO BET" and p["data_confidence"]]
    accumulators = []

    if len(actionable) >= 2:
        # Sort by model probability first (real predictive confidence),
        # edge second (value on top of that). Longshots no longer win by
        # having a big edge alone.
        sorted_picks = sorted(
            actionable,
            key=lambda x: (x["confidence_tier"] != "ELITE",
                           -(x["p_home"] if x["pick"] == "1" else x["p_draw"] if x["pick"] == "X" else x["p_away"]),
                           -x["adj_edge"])
        )
        for n_legs in [2, 3, 4]:
            if len(sorted_picks) >= n_legs:
                combo = sorted_picks[:n_legs]
                combined_odds = 1.0
                combined_prob = 1.0
                legs_info = []
                for c in combo:
                    odd = c["pick_odds"] or 1.50
                    combined_odds *= odd
                    p_val = c["p_home"] if c["pick"] == "1" else c["p_draw"] if c["pick"] == "X" else c["p_away"]
                    combined_prob *= p_val
                    legs_info.append({"home": c["home"], "away": c["away"], "league": c["league"],
                                       "pick": c["pick"], "tier": c["confidence_tier"], "odds": odd})
                accumulators.append({
                    "n_legs": n_legs, "combined_odds": round(combined_odds, 2),
                    "combined_model_prob": round(combined_prob * 100, 1),
                    "avg_adj_edge": round(sum(c["adj_edge"] for c in combo) / n_legs * 100, 1),
                    "legs": legs_info,
                })

    return {
        "total_fixtures": len(predictions),
        "actionable_picks": len(actionable),
        "no_bet_count": sum(1 for p in predictions if p["pick"] == "NO BET"),
        "unmodeled_count": sum(1 for p in predictions if not p["data_confidence"]),
        "acca_eligible_count": len(actionable),
        "predictions": predictions,
        "accumulators": accumulators,
    }

@app.get("/fixtures/today")
def get_today_fixtures(league: str = "PL"):
    if not FOOTBALL_DATA_KEY:
        raise HTTPException(status_code=400, detail="FOOTBALL_DATA_API_KEY is not configured")
    league_map = {"PL": "PL", "PD": "PD", "SA": "SA", "BL1": "BL1", "FL1": "FL1", "CL": "CL"}
    code = league_map.get(league.upper(), "PL")
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?status=SCHEDULED"
    req = urllib.request.Request(url, headers={"X-Auth-Token": FOOTBALL_DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            matches = data.get("matches", [])
            output = [{"home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
                       "kickoff": m.get("utcDate"), "status": m.get("status", "SCHEDULED")}
                      for m in matches[:15]]
            return {"league": code, "fixtures": output}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch fixtures: {str(e)}")

@app.get("/elo/{team}")
def get_elo_endpoint(team: str, international: bool = False):
    if international:
        elo, real = get_intl_elo(team)
        source = "eloratings.net (National Teams)" if real else "eloratings.net — NOT FOUND, showing generic default"
    else:
        elo, real = get_club_elo_live(team)
        source = "api.clubelo.com (European Clubs)" if real else "api.clubelo.com — NOT FOUND, showing generic default"
    return {"team": team, "elo": elo, "is_real_data": real, "source": source, "date": time.strftime("%Y-%m-%d")}
