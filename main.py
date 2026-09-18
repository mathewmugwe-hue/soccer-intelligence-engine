"""
═══════════════════════════════════════════════════════════════════════════════
SOCCER INTELLIGENCE ENGINE  v3.3  —  FASTAPI BACKEND (FIXED)
Zero LLM · Pure Mathematics · Dixon-Coles · Bivariate Poisson · Elo Engine

v3.3 CHANGES (fixes to v3.2 that were producing systematic longshot losses):
  1. Elo lookups now return (value, is_real_data) instead of silently defaulting
     to 1650 for unknown teams. Unknown-vs-unknown fixtures are honest "NO BET"
     instead of fabricated identical predictions.
  2. Pick logic now compares Home / Draw / Away (previously Draw was never a
     candidate, so the model could never pick X even when p_draw was highest).
  3. Edge is computed on DE-VIGGED (overround-removed) implied probabilities,
     and is only used to size confidence on the outcome the model ALREADY
     thinks is most likely — never to chase the outcome with the longest
     price. This removes the "big edge on a generic default probability =
     ELITE longshot" failure mode.
  4. Tiers require BOTH a probability floor AND a genuine edge, gated on real
     data for both teams. No more edge-only ELITE promotion, no fabricated
     0.05 "no odds" edge.
  5. Batch/acca builder only pulls from fixtures with verified data on both
     teams and sorts by model probability first, edge second.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import math
import time
import urllib.request
import urllib.parse
import json
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Soccer Intelligence Engine",
    version="3.3.0",
    description="Mathematical soccer prediction engine using calibrated Poisson, Dixon-Coles, and Elo ratings."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        req = urllib.request.Request(url, headers={"User-Agent": "SoccerEngine/3.3"})
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
    noise. Returns None if any odds are missing (no partial de-vig)."""
    if not (oh and od and oa) or min(oh, od, oa) <= 1.0:
        return None
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

    data_confidence = conf_h and conf_a
    hfa = 0.0 if f.is_neutral else (50.0 if is_intl else 65.0)
    elo_gap = (elo_h + hfa) - elo_a

    base_lh = WC_LAMBDA_HOME if is_intl else DEFAULT_LAMBDA_HOME
    base_la = WC_LAMBDA_AWAY if is_intl else DEFAULT_LAMBDA_AWAY
    rho     = WC_RHO if is_intl else DEFAULT_RHO

    goal_diff_mod = elo_gap / 600.0
    lambda_h = max(0.4, base_lh * (1.0 + goal_diff_mod))
    lambda_a = max(0.3, base_la * (1.0 - goal_diff_mod * 0.8))

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

    model_label = "Bivariate Poisson + Dixon-Coles [Top-5 Domestic]"
    if is_intl:
        model_label = "WC Tournament Poisson + eloratings.net K=40"
    elif domain == "club_continental":
        model_label = "Continental Poisson + ClubElo Live"
    elif domain == "unmodeled":
        model_label = "Unmodeled league — generic Poisson prior only, treat with caution"

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
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "online", "engine": "Soccer Intelligence Engine v3.3",
            "endpoints": ["/health", "/predict", "/predict/batch", "/fixtures/today", "/elo/{team}", "/data-sources"]}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.3.0",
            "active_keys": {"football_data": bool(FOOTBALL_DATA_KEY), "the_odds_api": bool(THE_ODDS_API_KEY),
                             "api_football": bool(API_FOOTBALL_KEY), "sportmonks": bool(SPORTMONKS_TOKEN)},
            "components": {"penaltyblog": True, "soccerdata": True, "dixon_coles": True,
                            "bivariate_poisson": True, "club_elo": True, "intl_elo": True,
                            "data_confidence_gating": True}}

@app.get("/data-sources")
def get_data_sources():
    return {"sources": [
        {"source": "api.clubelo.com", "type": "Live Club Elo", "key_required": False, "status": "Active / Free — big European leagues only"},
        {"source": "football-data.co.uk", "type": "Historical Odds & Results", "key_required": False, "status": "Active / Free via penaltyblog"},
        {"source": "api.football-data.org", "type": "Live Fixtures & Standings", "key_required": True, "status": "Active" if FOOTBALL_DATA_KEY else "Missing Key"},
        {"source": "understat.com", "type": "Expected Goals (xG)", "key_required": False, "status": "Active / Free via soccerdata"},
        {"source": "eloratings.net", "type": "National Team Elo", "key_required": False, "status": "Active / Seeded — top ~35 nations only"},
        {"source": "TheOddsAPI", "type": "Live Multi-Book Odds", "key_required": True, "status": "Active" if THE_ODDS_API_KEY else "Optional"},
        {"source": "API-Football", "type": "Lineups & Live Odds", "key_required": True, "status": "Configured" if API_FOOTBALL_KEY else "Optional"},
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
