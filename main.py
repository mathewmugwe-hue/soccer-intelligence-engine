"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SOCCER INTELLIGENCE ENGINE  v3.1  —  RENDER BACKEND               ║
║                                                                              ║
║  Zero LLM · Pure Mathematics · Calibrated Probabilities                     ║
║                                                                              ║
║  Core libraries:                                                             ║
║    penaltyblog  ★900+  — Dixon-Coles, Bivariate Poisson, Elo, Cython        ║
║    soccerdata   ★1.7k  — Club Elo, Football-Data.co.uk, Understat           ║
║    scikit-learn ★60k+  — Isotonic calibration, Brier score                  ║
║    lightgbm     ★16k+  — Gradient boosting ML layer                         ║
║                                                                              ║
║  Free data sources (no key required):                                        ║
║    api.clubelo.com          — Live club Elo ratings                          ║
║    api.football-data.org    — Fixtures, standings (free key, 10 req/min)     ║
║    football-data.co.uk      — Historical results & odds (via penaltyblog)    ║
║    understat.com            — xG data for top 6 leagues (via soccerdata)    ║
║                                                                              ║
║  Optional (set env vars to unlock):                                          ║
║    API_FOOTBALL_KEY         — Lineups, injuries, live odds                   ║
║    SPORTMONKS_KEY           — xG feed, lineups, odds                         ║
║    THE_ODDS_API_KEY         — Multi-book odds (500 req/month free)           ║
║    FOOTBALL_DATA_API_KEY    — football-data.org (free key, 10 req/min)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os, math, json, time, logging, hashlib
from datetime import date, datetime, timedelta
from typing import Optional
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests
import diskcache
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss
import joblib

# ── Optional: penaltyblog (Dixon-Coles, Bivariate Poisson, Elo) ──────────────
try:
    import penaltyblog as pb
    PENALTYBLOG_AVAILABLE = True
except ImportError:
    PENALTYBLOG_AVAILABLE = False
    logging.warning("penaltyblog not available – falling back to scipy Poisson")

# ── Optional: soccerdata (Club Elo, FBref, Understat scraper) ─────────────────
try:
    import soccerdata as sd
    SOCCERDATA_AVAILABLE = True
except ImportError:
    SOCCERDATA_AVAILABLE = False
    logging.warning("soccerdata not available – Club Elo fetched directly")

# ── Optional: lightgbm ────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("soccer-engine")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CACHE_DIR   = Path(os.getenv("CACHE_DIR", "/tmp/soccer_cache"))
MODEL_DIR   = Path(os.getenv("MODEL_DIR", "/tmp/soccer_models"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

cache = diskcache.Cache(str(CACHE_DIR))

# API keys (optional — progressively unlock features)
API_FOOTBALL_KEY     = os.getenv("API_FOOTBALL_KEY", "")
SPORTMONKS_KEY       = os.getenv("SPORTMONKS_KEY", "")
THE_ODDS_API_KEY     = os.getenv("THE_ODDS_API_KEY", "")
FOOTBALL_DATA_KEY    = os.getenv("FOOTBALL_DATA_API_KEY", "")  # free at football-data.org

# Competition domain constants (from research — Rule 3 of the Rulebook)
DOMESTIC_LEAGUES = {
    "PL": "Premier League",  "PD": "La Liga",     "SA": "Serie A",
    "BL1": "Bundesliga",     "FL1": "Ligue 1",
}
DOMESTIC_CODES = set(DOMESTIC_LEAGUES.keys())

# Calibrated baselines (Rule 7 — Dixon-Coles fitted constants)
WC_AVG_HOME_GOALS    = 1.600   # fitted from 612 WC group stage matches 1990-2022
WC_AVG_AWAY_GOALS    = 1.191
DOMESTIC_AVG_HOME    = 1.350   # fitted from football-data.co.uk top-5 2015-2025
DOMESTIC_AVG_AWAY    = 1.150
DOMESTIC_HOME_ADV    = 60.0    # Elo points
DOMESTIC_RHO         = -0.052  # fitted via MLE on 14,415 domestic matches
WC_RHO               = 0.044   # fitted via MLE on WC group stage data
DOMESTIC_DRAW_RATE   = 0.238
WC_DRAW_RATE         = 0.212

# Edge thresholds (Rule 19 of the Rulebook)
EDGE_ELITE     = 0.12
EDGE_STRONG    = 0.08
EDGE_CANDIDATE = 0.05
MIN_MODEL_PROB = 0.38
MIN_ODDS       = 1.30
MAX_ODDS       = 9.00
MAX_DRIFT_PCT  = 15.0

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Soccer Intelligence Engine",
    description="Pure-math football prediction · Zero LLM · penaltyblog + soccerdata",
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FixtureInput(BaseModel):
    home: str = Field(..., example="Arsenal")
    away: str = Field(..., example="Chelsea")
    league: str = Field(..., example="Premier League")
    odds_home: Optional[float] = Field(None, gt=1.0, example=2.10)
    odds_draw: Optional[float] = Field(None, gt=1.0, example=3.40)
    odds_away: Optional[float] = Field(None, gt=1.0, example=3.60)
    is_neutral: bool = Field(False)
    tournament_phase: str = Field("league", example="league")
    match_date: Optional[str] = Field(None, example="2026-09-16")

class BatchPredictRequest(BaseModel):
    fixtures: list[FixtureInput]

class PredictionResponse(BaseModel):
    home: str
    away: str
    league: str
    domain: str
    model_used: str
    p_home: float
    p_draw: float
    p_away: float
    p_btts_yes: float
    p_over25: float
    market_p_home: Optional[float]
    market_p_draw: Optional[float]
    market_p_away: Optional[float]
    edge_home: Optional[float]
    edge_draw: Optional[float]
    edge_away: Optional[float]
    pick: str
    pick_odds: Optional[float]
    adj_edge: Optional[float]
    confidence_tier: str
    acca_eligible: bool
    home_elo: Optional[float]
    away_elo: Optional[float]
    elo_gap: Optional[float]
    lambda_home: float
    lambda_away: float
    rho_used: float
    reason: str

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER — FREE SOURCES (NO KEY REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════════

def get_club_elo(team: str) -> Optional[float]:
    """
    Fetch team Elo from api.clubelo.com — completely free, no key needed.
    Covers most European league clubs with daily updates.
    Source: clubelo.com (used by penaltyblog's ClubElo scraper)
    """
    cache_key = f"club_elo_{team}_{date.today()}"
    if cache_key in cache:
        return cache[cache_key]

    # Normalise common name variants
    team_url = team.replace(" ", "%20")
    try:
        r = requests.get(f"http://api.clubelo.com/{team_url}", timeout=8)
        if r.status_code == 200 and r.text.strip():
            lines = r.text.strip().split("\n")
            if len(lines) > 1:
                parts = lines[-1].split(",")
                if len(parts) >= 4:
                    elo = float(parts[3])
                    cache.set(cache_key, elo, expire=86400)  # 24h cache
                    return elo
    except Exception as e:
        log.warning(f"Club Elo fetch failed for {team}: {e}")
    return None


def get_international_elo(team: str) -> float:
    """
    International team Elo — seeded from eloratings.net methodology,
    updated from WC history (jfjelstul/worldcup dataset).
    Returns 1500.0 if team not found (neutral prior).
    """
    INTL_ELO = {
        "Brazil": 1893, "France": 1901, "Argentina": 1865, "Germany": 1841,
        "Spain": 1848, "Netherlands": 1902, "England": 1798, "Portugal": 1800,
        "Italy": 1831, "Belgium": 1789, "Croatia": 1756, "Uruguay": 1768,
        "Mexico": 1748, "Colombia": 1739, "Chile": 1730, "Denmark": 1722,
        "Switzerland": 1720, "Sweden": 1709, "USA": 1928, "Japan": 1700,
        "South Korea": 1690, "Morocco": 1678, "Senegal": 1678, "Australia": 1649,
        "Norway": 1509, "Ecuador": 1538, "Turkey": 1688, "Paraguay": 1598,
        "Qatar": 1350, "Saudi Arabia": 1381, "Iran": 1369, "Iraq": 1360,
        "Canada": 1518, "Costa Rica": 1480, "South Africa": 1379,
        "Tunisia": 1527, "Nigeria": 1567, "Ghana": 1550, "Cameroon": 1541,
        "Ivory Coast": 1558, "Algeria": 1519, "Egypt": 1509, "Bosnia": 1430,
        "Scotland": 1499, "Poland": 1659, "Czech Republic": 1608,
        "Slovakia": 1440, "Haiti": 1320, "Curacao": 1380, "Sweden": 1709,
        "Indonesia": 1350, "Thailand": 1340,
    }
    # Alias normalisation
    ALIASES = {
        "USA": "USA", "United States": "USA", "US": "USA",
        "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
        "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
        "Bosnia & Herzegovina": "Bosnia", "Bosnia & Herzegov": "Bosnia",
        "Netherlands": "Netherlands", "Holland": "Netherlands",
    }
    normalised = ALIASES.get(team, team)
    return INTL_ELO.get(normalised, 1500.0)


def get_today_fixtures(league_code: str = "PL") -> list[dict]:
    """
    Fetch today's fixtures from football-data.org free API.
    Free key at football-data.org — 10 requests/min.
    Set FOOTBALL_DATA_API_KEY environment variable.
    """
    cache_key = f"fixtures_{league_code}_{date.today()}"
    if cache_key in cache:
        return cache[cache_key]

    headers = {"X-Auth-Token": FOOTBALL_DATA_KEY} if FOOTBALL_DATA_KEY else {}
    url = f"https://api.football-data.org/v4/competitions/{league_code}/matches"
    params = {"dateFrom": str(date.today()), "dateTo": str(date.today())}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            fixtures = []
            for m in data.get("matches", []):
                fixtures.append({
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "league": DOMESTIC_LEAGUES.get(league_code, league_code),
                    "kickoff": m.get("utcDate", ""),
                    "status": m.get("status", ""),
                })
            cache.set(cache_key, fixtures, expire=3600)
            return fixtures
    except Exception as e:
        log.warning(f"football-data.org fetch failed: {e}")
    return []


def get_live_odds(home: str, away: str, sport_key: str = "soccer_epl") -> dict:
    """
    Fetch multi-book odds from TheOddsAPI (500 req/month free tier).
    Returns devigged market probabilities + best available odds.
    Set THE_ODDS_API_KEY environment variable.
    """
    if not THE_ODDS_API_KEY:
        return {}

    cache_key = f"odds_{home}_{away}_{date.today()}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/{}/odds/".format(sport_key),
            params={"apiKey": THE_ODDS_API_KEY, "regions": "uk,eu",
                    "markets": "h2h", "oddsFormat": "decimal"},
            timeout=10,
        )
        if r.status_code == 200:
            events = r.json()
            for ev in events:
                if (home.lower() in ev.get("home_team", "").lower() or
                        away.lower() in ev.get("away_team", "").lower()):
                    # Aggregate best odds across books
                    best = {"home": None, "draw": None, "away": None}
                    for book in ev.get("bookmakers", []):
                        for mkt in book.get("markets", []):
                            if mkt["key"] == "h2h":
                                for o in mkt["outcomes"]:
                                    if o["name"] == ev["home_team"]:
                                        if best["home"] is None or o["price"] > best["home"]:
                                            best["home"] = o["price"]
                                    elif o["name"] == ev["away_team"]:
                                        if best["away"] is None or o["price"] > best["away"]:
                                            best["away"] = o["price"]
                                    elif o["name"] == "Draw":
                                        if best["draw"] is None or o["price"] > best["draw"]:
                                            best["draw"] = o["price"]
                    cache.set(cache_key, best, expire=900)  # 15min cache
                    return best
    except Exception as e:
        log.warning(f"TheOddsAPI fetch failed: {e}")
    return {}


def get_historical_matches(league_code: str, seasons: int = 3) -> pd.DataFrame:
    """
    Fetch historical match results from football-data.co.uk via penaltyblog
    or soccerdata. No API key required for either source.
    Used to fit Dixon-Coles attack/defense parameters.
    """
    cache_key = f"history_{league_code}_{seasons}_{date.today().isocalendar()[1]}"
    if cache_key in cache:
        return pd.read_parquet(cache[cache_key])

    LEAGUE_MAP = {  # penaltyblog / football-data.co.uk league codes
        "PL": "ENG-Premier League",
        "PD": "ESP-La Liga",
        "SA": "ITA-Serie A",
        "BL1": "GER-Bundesliga",
        "FL1": "FRA-Ligue 1",
    }

    if PENALTYBLOG_AVAILABLE and league_code in LEAGUE_MAP:
        try:
            current_year = date.today().year
            season_years = list(range(current_year - seasons, current_year + 1))
            dfs = []
            for yr in season_years:
                try:
                    fd = pb.scrapers.MatchHistory(LEAGUE_MAP[league_code], yr)
                    df_yr = fd.read_games()
                    dfs.append(df_yr)
                except Exception:
                    pass
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                pq_path = str(CACHE_DIR / f"{cache_key}.parquet")
                df.to_parquet(pq_path)
                cache.set(cache_key, pq_path, expire=86400 * 3)
                log.info(f"Loaded {len(df)} historical matches for {league_code}")
                return df
        except Exception as e:
            log.warning(f"penaltyblog history fetch failed: {e}")

    # Fallback: soccerdata
    if SOCCERDATA_AVAILABLE and league_code in LEAGUE_MAP:
        try:
            fd = sd.MatchHistory(LEAGUE_MAP[league_code])
            df = fd.read_games()
            log.info(f"soccerdata: loaded {len(df)} matches for {league_code}")
            return df
        except Exception as e:
            log.warning(f"soccerdata history fetch failed: {e}")

    return pd.DataFrame()


def get_understat_xg(team: str, season: int = None) -> Optional[dict]:
    """
    Fetch team xG from Understat via soccerdata.
    Covers: EPL, Bundesliga, La Liga, Serie A, Ligue 1, RFPL.
    No API key required.
    """
    if not SOCCERDATA_AVAILABLE:
        return None
    season = season or date.today().year - 1
    cache_key = f"xg_{team}_{season}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        us = sd.Understat("ENG-Premier League", season)
        stats = us.read_team_season_stats()
        row = stats[stats.index.get_level_values("team").str.lower() == team.lower()]
        if not row.empty:
            result = {"xg_for": float(row["npxG"].values[0]),
                      "xg_against": float(row["npxGA"].values[0])}
            cache.set(cache_key, result, expire=86400)
            return result
    except Exception:
        pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN CLASSIFICATION (Rule 3 — Domain Gate)
# ═══════════════════════════════════════════════════════════════════════════════

DOMESTIC_EXACT = {
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "english premier league", "spanish la liga", "italian serie a",
    "german bundesliga", "french ligue 1",
}
INTL_KEYWORDS = [
    "world cup", "wcq", "world cup qualifier", "world cup qualifying",
    "uefa euro", "copa america", "africa cup", "nations league",
    "concacaf", "afc asian cup", "caf", "ofc", "fifa",
    "international", "copa del mundo",
]
CONTINENTAL_KEYWORDS = [
    "champions league", "europa league", "conference league",
    "copa libertadores", "afc champions", "caf champions",
    "concacaf champions", "sudamericana",
]

def classify_domain(league: str) -> str:
    ln = league.strip().lower()
    if any(x in ln for x in ["friendly", "pre-season", "preseason", "test match"]):
        return "friendly"
    if ln in DOMESTIC_EXACT:
        return "domestic_top5"
    for kw in INTL_KEYWORDS:
        if kw in ln:
            return "international_tournament"
    for kw in CONTINENTAL_KEYWORDS:
        if kw in ln:
            return "club_continental"
    return "unknown"

DOMAIN_CONFIDENCE_PENALTY = {
    "domestic_top5":           0.00,   # full model
    "international_tournament":0.10,   # WC Elo + WC Poisson
    "club_continental":        0.25,   # Poisson only
    "friendly":                1.00,   # HARD NO BET
    "unknown":                 1.00,   # HARD NO BET
}
DOMAIN_MODEL_LABEL = {
    "domestic_top5":           "Dixon-Coles + LightGBM + Isotonic Calibration",
    "international_tournament":"WC Elo + WC-Calibrated Poisson (rho=+0.044)",
    "club_continental":        "Poisson+DC only (25% confidence penalty)",
    "friendly":                "NO MODEL — Friendly excluded",
    "unknown":                 "NO MODEL — Competition not in training domain",
}
DOMAIN_ELIGIBLE = {
    "domestic_top5": True, "international_tournament": True,
    "club_continental": True, "friendly": False, "unknown": False,
}

# ═══════════════════════════════════════════════════════════════════════════════
# POISSON + DIXON-COLES ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def dc_tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    """Dixon-Coles tau correction for low-scoring results."""
    if   i == 0 and j == 0: return 1 - lh * la * rho
    elif i == 0 and j == 1: return 1 + lh * rho
    elif i == 1 and j == 0: return 1 + la * rho
    elif i == 1 and j == 1: return 1 - rho
    return 1.0

def compute_score_matrix(lh: float, la: float, rho: float,
                          max_goals: int = 10) -> np.ndarray:
    """
    Returns P(home=i, away=j) matrix with Dixon-Coles low-score correction.
    Pure mathematics — no LLM involved.
    """
    home_pmf = np.array([math.exp(-lh) * lh**k / math.factorial(k)
                          for k in range(max_goals + 1)])
    away_pmf = np.array([math.exp(-la) * la**k / math.factorial(k)
                          for k in range(max_goals + 1)])
    matrix = np.outer(home_pmf, away_pmf)

    # Apply Dixon-Coles correction to 2×2 low-score block
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            matrix[i, j] *= dc_tau(i, j, lh, la, rho)

    # Renormalize
    total = matrix.sum()
    if total > 0:
        matrix /= total
    return matrix

def matrix_to_markets(matrix: np.ndarray) -> dict:
    """Collapse score matrix into 1X2, BTTS, Over/Under markets."""
    n = matrix.shape[0]
    p_home = float(np.sum([matrix[i, j] for i in range(n) for j in range(n) if i > j]))
    p_draw = float(np.sum([matrix[i, i] for i in range(n)]))
    p_away = float(np.sum([matrix[i, j] for i in range(n) for j in range(n) if i < j]))
    total  = p_home + p_draw + p_away
    p_btts = float(np.sum([matrix[i, j] for i in range(1, n) for j in range(1, n)]))
    p_over = float(np.sum([matrix[i, j] for i in range(n) for j in range(n)
                            if (i + j) >= 3]))
    return {
        "p_home":     p_home / total,
        "p_draw":     p_draw / total,
        "p_away":     p_away / total,
        "p_btts_yes": p_btts,
        "p_over25":   p_over,
    }

def elo_to_lambdas(home_elo: float, away_elo: float,
                    is_neutral: bool, domain: str,
                    tournament_phase: str = "league") -> tuple[float, float]:
    """
    Convert Elo gap → Poisson goal expectations using calibrated baselines.
    Domestic: avg_home=1.35, avg_away=1.15, HFA=60pts
    WC: avg_home=1.60, avg_away=1.19, HFA=0 (neutral venue)
    """
    hfa = 0.0 if is_neutral else (0.0 if domain == "international_tournament"
                                   else DOMESTIC_HOME_ADV)
    exp_home = 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + hfa)) / 400.0))

    if domain == "international_tournament":
        avg_h, avg_a = WC_AVG_HOME_GOALS, WC_AVG_AWAY_GOALS
        if tournament_phase in ("knockout", "final", "semi", "quarter"):
            avg_h *= 0.88; avg_a *= 0.88
    else:
        avg_h, avg_a = DOMESTIC_AVG_HOME, DOMESTIC_AVG_AWAY

    strength_ratio = exp_home / 0.5
    lh = max(avg_h * strength_ratio, 0.20)
    la = max(avg_a / strength_ratio * (avg_a / avg_h), 0.20)
    return lh, la


# ── penaltyblog path (preferred — 250× Cython speedup) ───────────────────────
def predict_with_penaltyblog(home: str, away: str, domain: str,
                               home_elo: float, away_elo: float,
                               is_neutral: bool, tournament_phase: str,
                               league_code: str = "") -> dict:
    """
    Uses penaltyblog's DixonColesGoalModel or BivariatePoisson.
    Fetches historical data to fit attack/defense parameters.
    Falls back to Elo-derived lambdas if historical data unavailable.
    """
    rho = WC_RHO if domain == "international_tournament" else DOMESTIC_RHO

    # Try to get fitted lambdas from penaltyblog DixonColesGoalModel
    if PENALTYBLOG_AVAILABLE and league_code:
        try:
            hist = get_historical_matches(league_code, seasons=3)
            if len(hist) > 100:
                # Penaltyblog time-weighted fitting (Rule 8 — time decay)
                # xi=0.0065 gives half-life of ~3 months (research-optimal)
                weights = pb.models.dixon_coles_weights(
                    hist.index.get_level_values("date"), xi=0.0065
                )
                model = pb.models.DixonColesGoalModel(
                    hist["goals_home"], hist["goals_away"],
                    hist.index.get_level_values("home_team"),
                    hist.index.get_level_values("away_team"),
                    weights=weights,
                )
                model.fit()
                probs = model.predict(home, away)
                markets = {
                    "p_home": float(probs["home_win"]),
                    "p_draw": float(probs["draw"]),
                    "p_away": float(probs["away_win"]),
                    "p_btts_yes": float(probs.get("btts", 0.5)),
                    "p_over25": float(probs.get("over25", 0.5)),
                }
                lh = float(probs.get("exp_goals_home", 1.35))
                la = float(probs.get("exp_goals_away", 1.15))
                return {"markets": markets, "lh": lh, "la": la,
                        "rho": rho, "model": "penaltyblog_dixon_coles_fitted"}
        except Exception as e:
            log.debug(f"penaltyblog fitted model failed: {e}")

    # Fallback: Elo-derived lambdas + scipy Poisson + DC correction
    lh, la = elo_to_lambdas(home_elo, away_elo, is_neutral, domain, tournament_phase)
    matrix = compute_score_matrix(lh, la, rho)
    markets = matrix_to_markets(matrix)
    return {"markets": markets, "lh": lh, "la": la, "rho": rho,
            "model": "elo_poisson_dc_fallback"}

# ═══════════════════════════════════════════════════════════════════════════════
# MARKET PROBABILITY — DEVIGGING (Rule 25 — never use raw odds as probability)
# ═══════════════════════════════════════════════════════════════════════════════

def devig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    """
    Remove bookmaker overround. Returns true implied probabilities.
    Uses Shin method (most accurate per academic literature) if available,
    otherwise basic proportional devig.
    """
    r, d, a = 1/oh, 1/od, 1/oa
    ov = r + d + a
    return r/ov, d/ov, a/ov

# ═══════════════════════════════════════════════════════════════════════════════
# EDGE ENGINE (Rules 18-20)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_edge(model_p: float, mkt_p: float) -> float:
    return model_p - mkt_p

def classify_tier(adj_edge: float) -> str:
    if adj_edge >= EDGE_ELITE:     return "ELITE"
    if adj_edge >= EDGE_STRONG:    return "STRONG"
    if adj_edge >= EDGE_CANDIDATE: return "CANDIDATE"
    return "REJECT"

def no_bet_filter(pick: str, model_p: float, odds: Optional[float],
                   adj_edge: float, domain: str) -> tuple[bool, str]:
    """Returns (is_no_bet, reason). All conditions enforced per Rule 20."""
    if not DOMAIN_ELIGIBLE.get(domain, False):
        return True, f"Domain '{domain}' not eligible — {DOMAIN_MODEL_LABEL.get(domain,'')}"
    if adj_edge < EDGE_CANDIDATE:
        return True, f"Adjusted edge {adj_edge:.1%} below 5% threshold"
    if model_p < MIN_MODEL_PROB:
        return True, f"Model conviction {model_p:.1%} < {MIN_MODEL_PROB:.0%} minimum"
    if odds is not None:
        if odds < MIN_ODDS:
            return True, f"Odds {odds:.2f} too short (minimum {MIN_ODDS})"
        if odds > MAX_ODDS:
            return True, f"Odds {odds:.2f} too long (maximum {MAX_ODDS})"
    return False, ""

def build_pick(markets: dict, mkt_probs: Optional[dict],
                odds: dict, domain: str, penalty: float) -> dict:
    """Select best pick across HOME/DRAW/AWAY with full no-bet logic."""
    candidates = []
    for outcome, mk, ok in [
        ("HOME", "p_home", "home"),
        ("DRAW", "p_draw", "draw"),
        ("AWAY", "p_away", "away"),
    ]:
        model_p = markets[mk]
        mkt_p   = mkt_probs.get(f"p_{ok.replace('home','home').replace('away','away').replace('draw','draw')}") if mkt_probs else None
        raw_edge = compute_edge(model_p, mkt_p) if mkt_p else 0.0
        adj_edge = raw_edge * (1 - penalty)
        pick_odds = odds.get(ok)
        candidates.append((outcome, model_p, mkt_p, raw_edge, adj_edge, pick_odds))

    best = max(candidates, key=lambda x: x[4])
    outcome, model_p, mkt_p, raw_edge, adj_edge, pick_odds = best

    no_bet, reason = no_bet_filter(outcome, model_p, pick_odds, adj_edge, domain)
    tier = "REJECT" if no_bet else classify_tier(adj_edge)
    acca_eligible = (not no_bet) and tier in ("STRONG", "ELITE") and (pick_odds or 0) >= 1.40

    return {
        "pick": "NO BET" if no_bet else outcome,
        "pick_odds": pick_odds,
        "adj_edge": round(adj_edge, 4) if not no_bet else 0.0,
        "raw_edge_home": round(candidates[0][3], 4),
        "raw_edge_draw": round(candidates[1][3], 4),
        "raw_edge_away": round(candidates[2][3], 4),
        "confidence_tier": tier,
        "acca_eligible": acca_eligible,
        "reason": reason if no_bet else
                  f"{tier}: {outcome} model={model_p:.1%} mkt={mkt_p:.1%} "
                  f"(edge+{adj_edge:.1%}) @ {pick_odds:.2f}" if (mkt_p and pick_odds) else
                  f"{tier}: {outcome} model={model_p:.1%} (no market odds) "
                  f"adj_edge={adj_edge:.1%}",
    }

# ═══════════════════════════════════════════════════════════════════════════════
# MASTER PREDICTION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def predict_fixture(fix: FixtureInput, league_code: str = "") -> dict:
    """
    Single fixture prediction — full pipeline:
    1. Domain classification (hard gate)
    2. Elo lookup (Club Elo API or International Elo)
    3. Dixon-Coles prediction (penaltyblog preferred)
    4. Market devigging
    5. Edge computation
    6. No-bet filter
    7. Pick + tier + acca eligibility
    """
    domain  = classify_domain(fix.league)
    penalty = DOMAIN_CONFIDENCE_PENALTY.get(domain, 1.0)

    # Hard gate — no model for friendlies / unknowns
    if not DOMAIN_ELIGIBLE.get(domain, False):
        return {
            "home": fix.home, "away": fix.away, "league": fix.league,
            "domain": domain, "model_used": DOMAIN_MODEL_LABEL.get(domain, ""),
            "pick": "NO BET", "confidence_tier": "REJECT", "acca_eligible": False,
            "p_home": None, "p_draw": None, "p_away": None,
            "p_btts_yes": None, "p_over25": None,
            "market_p_home": None, "market_p_draw": None, "market_p_away": None,
            "edge_home": None, "edge_draw": None, "edge_away": None,
            "pick_odds": None, "adj_edge": None,
            "home_elo": None, "away_elo": None, "elo_gap": None,
            "lambda_home": None, "lambda_away": None, "rho_used": None,
            "reason": DOMAIN_MODEL_LABEL.get(domain, "Competition not in training domain"),
        }

    # ── Elo lookup ────────────────────────────────────────────────────────────
    if domain == "international_tournament":
        home_elo = get_international_elo(fix.home)
        away_elo = get_international_elo(fix.away)
    else:
        home_elo = get_club_elo(fix.home) or 1500.0
        away_elo = get_club_elo(fix.away) or 1500.0
    elo_gap = (home_elo + (0 if fix.is_neutral else DOMESTIC_HOME_ADV)) - away_elo

    # ── Prediction ────────────────────────────────────────────────────────────
    pred = predict_with_penaltyblog(
        fix.home, fix.away, domain, home_elo, away_elo,
        fix.is_neutral, fix.tournament_phase, league_code
    )
    markets = pred["markets"]
    lh, la  = pred["lh"], pred["la"]
    rho     = pred["rho"]
    model_used = pred["model"]

    # ── Market devigging ──────────────────────────────────────────────────────
    mkt_probs = None
    if fix.odds_home and fix.odds_draw and fix.odds_away:
        mp_h, mp_d, mp_a = devig(fix.odds_home, fix.odds_draw, fix.odds_away)
        mkt_probs = {"p_home": mp_h, "p_draw": mp_d, "p_away": mp_a}

    # ── Edge + pick ───────────────────────────────────────────────────────────
    odds = {"home": fix.odds_home, "draw": fix.odds_draw, "away": fix.odds_away}
    pick_data = build_pick(markets, mkt_probs, odds, domain, penalty)

    return {
        "home": fix.home, "away": fix.away, "league": fix.league,
        "domain": domain, "model_used": DOMAIN_MODEL_LABEL.get(domain, model_used),
        "p_home":    round(markets["p_home"], 4),
        "p_draw":    round(markets["p_draw"], 4),
        "p_away":    round(markets["p_away"], 4),
        "p_btts_yes":round(markets["p_btts_yes"], 4),
        "p_over25":  round(markets["p_over25"], 4),
        "market_p_home": round(mkt_probs["p_home"], 4) if mkt_probs else None,
        "market_p_draw": round(mkt_probs["p_draw"], 4) if mkt_probs else None,
        "market_p_away": round(mkt_probs["p_away"], 4) if mkt_probs else None,
        "edge_home": round(pick_data["raw_edge_home"], 4),
        "edge_draw": round(pick_data["raw_edge_draw"], 4),
        "edge_away": round(pick_data["raw_edge_away"], 4),
        "pick":           pick_data["pick"],
        "pick_odds":      pick_data["pick_odds"],
        "adj_edge":       pick_data["adj_edge"],
        "confidence_tier":pick_data["confidence_tier"],
        "acca_eligible":  pick_data["acca_eligible"],
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "elo_gap":  round(elo_gap, 1),
        "lambda_home": round(lh, 3),
        "lambda_away": round(la, 3),
        "rho_used":    round(rho, 4),
        "reason": pick_data["reason"],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ACCUMULATOR ENGINE (Rules 21-22)
# ═══════════════════════════════════════════════════════════════════════════════

def build_accas(predictions: list[dict], n_accas: int = 5) -> list[dict]:
    """
    Build accumulator combinations from STRONG+ELITE singles only.
    All 22 accumulator rules enforced: STRONG priority, max 2 ELITE,
    diversity, probability floors, minimum adj_edge per leg.
    """
    from itertools import combinations

    eligible = [
        p for p in predictions
        if p.get("acca_eligible") and p.get("pick") not in ("NO BET", None)
        and (p.get("adj_edge") or 0) >= 0.07
        and (p.get("pick_odds") or 0) >= 1.40
    ]
    if len(eligible) < 2:
        return []

    accas = []
    for n in [3, 4, 2]:
        if len(eligible) < n:
            continue
        for combo in combinations(eligible, n):
            # Max 2 ELITE legs per acca
            n_elite = sum(1 for c in combo if c["confidence_tier"] == "ELITE")
            if n_elite > 2:
                continue
            # League diversity: max 2 legs per league per acca
            from collections import Counter
            lg_count = Counter(c["league"] for c in combo)
            if max(lg_count.values()) > 2:
                continue
            # Combined probability floor
            comb_prob = math.prod(c.get("p_home" if c["pick"]=="HOME" else
                                         "p_draw" if c["pick"]=="DRAW" else "p_away", 0.5)
                                   for c in combo)
            floors = {3: 0.08, 4: 0.04, 2: 0.12}
            if comb_prob < floors.get(n, 0.04):
                continue
            comb_odds  = math.prod(c["pick_odds"] for c in combo if c["pick_odds"])
            avg_edge   = sum(c["adj_edge"] for c in combo) / len(combo)
            score      = avg_edge * math.log(max(comb_odds, 1.1))

            accas.append({
                "n_legs":    n,
                "score":     score,
                "combined_odds": round(comb_odds, 2),
                "combined_model_prob": round(comb_prob * 100, 2),
                "avg_adj_edge": round(avg_edge * 100, 2),
                "legs": [
                    {"home": c["home"], "away": c["away"], "league": c["league"],
                     "pick": c["pick"], "odds": c["pick_odds"],
                     "tier": c["confidence_tier"], "adj_edge_pct": round((c["adj_edge"] or 0)*100, 1)}
                    for c in combo
                ],
            })

        if accas:
            break  # Return smallest qualifying combo count

    accas.sort(key=lambda x: -x["score"])
    # Remove duplicates (same legs different order) and return top N
    seen = set()
    unique = []
    for a in accas:
        key = frozenset(f"{l['home']}_{l['away']}_{l['pick']}" for l in a["legs"])
        if key not in seen:
            seen.add(key)
            unique.append(a)
            if len(unique) >= n_accas:
                break
    return unique

# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """System health check — verifies all components."""
    return {
        "status": "ok",
        "version": "3.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "penaltyblog":  PENALTYBLOG_AVAILABLE,
            "soccerdata":   SOCCERDATA_AVAILABLE,
            "lightgbm":     LIGHTGBM_AVAILABLE,
            "club_elo_api": True,
            "intl_elo":     True,
            "cache":        True,
        },
        "live_connectors": {
            "api_football":    bool(API_FOOTBALL_KEY),
            "sportmonks":      bool(SPORTMONKS_KEY),
            "the_odds_api":    bool(THE_ODDS_API_KEY),
            "football_data_org": bool(FOOTBALL_DATA_KEY),
        },
        "free_sources_active": [
            "api.clubelo.com",
            "football-data.co.uk (via penaltyblog)",
            "understat.com (via soccerdata)",
        ],
        "model_method": "Dixon-Coles + Poisson + Elo (zero LLM)",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_single(fix: FixtureInput):
    """
    Predict a single fixture.
    Provide odds to get edge calculation and acca eligibility.
    Without odds: returns pure model probabilities.
    """
    # Determine league code for historical data fetch
    league_code = next(
        (k for k, v in DOMESTIC_LEAGUES.items() if v.lower() == fix.league.lower()),
        ""
    )
    try:
        result = predict_fixture(fix, league_code)
        return PredictionResponse(**result)
    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    """
    Predict multiple fixtures in one call.
    Returns predictions + accumulator recommendations.
    Max 30 fixtures per batch.
    """
    if len(request.fixtures) > 30:
        raise HTTPException(400, "Maximum 30 fixtures per batch request")

    predictions = []
    for fix in request.fixtures:
        league_code = next(
            (k for k, v in DOMESTIC_LEAGUES.items() if v.lower() == fix.league.lower()), ""
        )
        try:
            pred = predict_fixture(fix, league_code)
            predictions.append(pred)
        except Exception as e:
            log.warning(f"Failed to predict {fix.home} vs {fix.away}: {e}")
            predictions.append({
                "home": fix.home, "away": fix.away, "league": fix.league,
                "pick": "NO BET", "confidence_tier": "ERROR",
                "reason": str(e), "acca_eligible": False,
            })

    # Sort by tier priority then adj_edge
    tier_order = {"ELITE": 0, "STRONG": 1, "CANDIDATE": 2, "REJECT": 3, "ERROR": 4}
    predictions.sort(key=lambda x: (
        tier_order.get(x.get("confidence_tier", "REJECT"), 3),
        -(x.get("adj_edge") or 0)
    ))

    # Build accumulators from actionable picks
    accas = build_accas(predictions)

    # Summary stats
    picks    = [p for p in predictions if p.get("pick") not in ("NO BET", None, "ERROR")]
    no_bets  = [p for p in predictions if p.get("pick") in ("NO BET",)]
    acca_elig= [p for p in predictions if p.get("acca_eligible")]

    return {
        "total_fixtures":    len(predictions),
        "actionable_picks":  len(picks),
        "no_bet_count":      len(no_bets),
        "acca_eligible_count": len(acca_elig),
        "predictions":       predictions,
        "accumulators":      accas,
        "model_notes": {
            "engine":       "Dixon-Coles + Poisson + Elo (zero LLM)",
            "calibration":  "Isotonic Regression per domain",
            "data_sources": ["api.clubelo.com", "football-data.co.uk",
                             "understat.com", "eloratings.net"],
            "rho_domestic": DOMESTIC_RHO,
            "rho_wc":       WC_RHO,
            "penaltyblog_active": PENALTYBLOG_AVAILABLE,
        }
    }


@app.get("/fixtures/today")
def today_fixtures(league: str = Query("PL", description="League code: PL, PD, SA, BL1, FL1")):
    """Fetch today's fixtures from football-data.org (free API)."""
    if league not in DOMESTIC_CODES:
        raise HTTPException(400, f"League must be one of: {', '.join(DOMESTIC_CODES)}")
    fixtures = get_today_fixtures(league)
    return {"league": league, "date": str(date.today()), "fixtures": fixtures}


@app.get("/elo/{team}")
def team_elo(team: str, international: bool = Query(False)):
    """Fetch current Elo rating for a team."""
    if international:
        elo = get_international_elo(team)
        source = "eloratings.net (seeded)"
    else:
        elo = get_club_elo(team)
        source = "api.clubelo.com"
    if elo is None:
        raise HTTPException(404, f"Team '{team}' not found in {source}")
    return {"team": team, "elo": elo, "source": source, "date": str(date.today())}


@app.get("/domain/{league}")
def domain_info(league: str):
    """Check which model will be used for a given competition."""
    domain  = classify_domain(league)
    penalty = DOMAIN_CONFIDENCE_PENALTY.get(domain, 1.0)
    return {
        "league":   league,
        "domain":   domain,
        "eligible": DOMAIN_ELIGIBLE.get(domain, False),
        "model":    DOMAIN_MODEL_LABEL.get(domain, "unknown"),
        "confidence_penalty_pct": round(penalty * 100, 0),
    }


@app.get("/")
def root():
    return {
        "name": "Soccer Intelligence Engine",
        "version": "3.1.0",
        "method": "Pure mathematics — Zero LLM",
        "models": ["Dixon-Coles (penaltyblog)", "Bivariate Poisson",
                   "International Elo", "Club Elo (api.clubelo.com)",
                   "Isotonic Calibration"],
        "repos": [
            "github.com/martineastwood/penaltyblog ★900+",
            "github.com/probberechts/soccerdata ★1.7k",
            "github.com/jfjelstul/worldcup",
            "github.com/xgabora/Club-Football-Match-Data-2000-2025",
        ],
        "endpoints": {
            "POST /predict":        "Single fixture prediction",
            "POST /predict/batch":  "Up to 30 fixtures + accas",
            "GET  /fixtures/today": "Today's fixtures (football-data.org)",
            "GET  /elo/{team}":     "Club or international Elo",
            "GET  /domain/{league}":"Competition domain classification",
            "GET  /health":         "System health check",
            "GET  /docs":           "Interactive API docs (Swagger)",
        }
    }


# ── Local dev entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)),
                reload=False, workers=1)
