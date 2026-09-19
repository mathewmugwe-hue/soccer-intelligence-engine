"""
SOCCER INTELLIGENCE ENGINE v4.0 - FASTAPI BACKEND

What changed vs v3.3 (root causes of "277 fixtures, 0 picks"):
  1. Team-name matching was a substring test, so "Winterthur" and "Fc Inter Turku" matched
     "inter" (Inter Milan, 1975 Elo), "Arsenal Dzerzhinsk" matched Arsenal, women's/reserve
     sides matched the men's first team. Now: exact / alias / high-cutoff fuzzy matching,
     and women's / youth / reserve sides never get a men's club rating.
  2. ClubElo was queried per team (3s timeout) and the wrong CSV row was read (lines[1] is the
     OLDEST row). Now ONE bulk download (api.clubelo.com/<date>) is cached for 12 hours.
  3. Fixtures with no rating were forced to "NO BET". Now every fixture gets a pick. The
     de-vigged bookmaker price (the strongest free predictor across all leagues) is the base,
     Elo is a 20% supplement where a trustworthy rating exists.
  4. Accas are built from the WHOLE slate (not per 25-fixture chunk), ranked by win probability.
  5. Elo -> goals mapping is calibrated to the Elo expectancy formula (was arbitrary).
  6. Hard-coded API keys removed. Set them as environment variables.
  7. /fixtures/today now really filters on today's date.
"""

import os
import io
import re
import csv
import math
import time
import json
import threading
import unicodedata
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from difflib import get_close_matches
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Soccer Intelligence Engine", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════ CONFIG ═══════════════════════════════
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
SPORTMONKS_TOKEN = os.getenv("SPORTMONKS_API_TOKEN", "")

TOTAL_GOALS = 2.65          # league-average total goals used when fitting lambdas
MU_HOME, MU_AWAY = 1.45, 1.15   # equal-strength home/away goal rates (HFA already inside)
MU_NEUTRAL = 1.30
ELO_SCALE = 0.0022          # calibrated so Poisson win-expectancy == Elo expectancy
RHO = -0.052
WC_RHO = 0.044
MARKET_WEIGHT = 0.80        # weight of de-vigged market when a rating also exists
MAX_ELO_DISAGREE = 0.25     # if Elo and market disagree more than this on the favourite, drop Elo
ACCA_MIN_PROB = 0.60        # minimum win probability for a leg
ELO_TTL = 12 * 3600

INTL_ELO_SEEDS: Dict[str, float] = {
    "argentina": 2145, "france": 2110, "spain": 2105, "england": 2040, "brazil": 2035,
    "belgium": 1980, "netherlands": 1975, "portugal": 1970, "colombia": 1960, "italy": 1950,
    "uruguay": 1940, "germany": 1930, "croatia": 1910, "morocco": 1890, "japan": 1880,
    "senegal": 1850, "usa": 1845, "united states": 1845, "mexico": 1840, "switzerland": 1835,
    "denmark": 1820, "austria": 1815, "south korea": 1800, "korea republic": 1800, "iran": 1795,
    "australia": 1780, "turkey": 1775, "ukraine": 1770, "nigeria": 1760, "egypt": 1750,
    "ivory coast": 1745, "cameroon": 1730, "algeria": 1725, "ghana": 1710,
    "kenya": 1390, "uganda": 1410, "tanzania": 1360,
}

# bookmaker-name (normalised) -> candidate ClubElo names (normalised). First one found wins.
ALIASES: Dict[str, List[str]] = {
    "manchester city": ["man city"], "manchester united": ["man united"],
    "nottingham forest": ["forest"], "nottm forest": ["forest"],
    "wolverhampton wanderers": ["wolves"], "wolverhampton": ["wolves"],
    "tottenham hotspur": ["tottenham"], "newcastle united": ["newcastle"],
    "west ham united": ["west ham"], "brighton and hove albion": ["brighton"],
    "leicester city": ["leicester"], "leeds united": ["leeds"],
    "sheffield united": ["sheffield utd"], "west bromwich albion": ["west brom"],
    "bayern munich": ["bayern"], "borussia dortmund": ["dortmund"],
    "bayer leverkusen": ["leverkusen"], "borussia monchengladbach": ["gladbach"],
    "eintracht frankfurt": ["frankfurt"], "rb leipzig": ["leipzig"],
    "1 mainz 05": ["mainz"], "mainz 05": ["mainz"], "union berlin": ["union berlin"],
    "paris saint germain": ["paris sg"], "psg": ["paris sg"], "paris saint-germain": ["paris sg"],
    "atletico madrid": ["atletico"], "athletic bilbao": ["athletic"], "athletic club": ["athletic"],
    "real sociedad": ["sociedad"], "real betis": ["betis"], "celta vigo": ["celta"],
    "deportivo alaves": ["alaves"], "inter milan": ["inter"], "internazionale": ["inter"],
    "milan": ["milan"], "sporting cp": ["sporting"], "sporting lisbon": ["sporting"],
    "benfica lisbon": ["benfica"], "kaa gent": ["gent"], "standard liege": ["standard"],
    "club brugge": ["brugge"], "psv eindhoven": ["psv"], "hoffenheim": ["hoffenheim"],
    "tsg hoffenheim": ["hoffenheim"], "ofi crete": ["ofi"], "fc barcelona": ["barcelona"],
    "rio ave": ["rio ave"],
}

STOP_TOKENS = {"fc", "cf", "sc", "sk", "fk", "afc", "ac", "as", "cd", "ca", "cs", "sv", "fsv",
               "tsv", "sd", "ud", "ss", "us", "ks", "kf", "nk", "ik", "if", "bk", "ff", "kv",
               "kaa", "rsc", "rc", "fcm", "1", "the"}

NON_SENIOR_TEAM_RE = re.compile(
    r"^(w|wom\w*|women|womens|ladies|ii|iii|b|2|jong|re|res|reserv\w*|youth|jeugd|u\d+|u)$")
NON_SENIOR_LEAGUE_RE = re.compile(
    r"\b(wom\w*|wo|femen\w*|femin\w*|ladies|frauen|damallsvenskan|elitettan|u-?\d\d|"
    r"reserv\w*|youth|jeugd|academy)\b")

COUNTRY_CODES = {
    "england": "ENG", "spain": "ESP", "germany": "GER", "italy": "ITA", "france": "FRA",
    "netherlands": "NED", "portugal": "POR", "belgium": "BEL", "scotland": "SCO",
    "austria": "AUT", "switzerland": "SUI", "turkiye": "TUR", "turkey": "TUR", "greece": "GRE",
    "denmark": "DEN", "sweden": "SWE", "norway": "NOR", "poland": "POL", "czechia": "CZE",
    "croatia": "CRO", "serbia": "SRB", "ukraine": "UKR", "russia": "RUS", "romania": "ROM",
    "bulgaria": "BUL", "hungary": "HUN", "slovakia": "SVK", "slovenia": "SVN",
    "cyprus": "CYP", "israel": "ISR", "finland": "FIN", "ireland": "IRL",
    "northern ireland": "NIR", "wales": "WAL",
}


# ═══════════════════════════════ NAME HANDLING ═══════════════════════════════
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def is_truncated(raw: str) -> bool:
    return raw.strip().endswith("...") or raw.strip().endswith("\u2026")


def clean_raw(raw: str) -> str:
    return raw.strip().rstrip(".\u2026").strip()


def norm(raw: str) -> str:
    s = _strip_accents(clean_raw(raw)).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t not in STOP_TOKENS]
    return " ".join(toks)


def team_is_non_senior(raw: str) -> bool:
    s = _strip_accents(clean_raw(raw)).lower()
    toks = re.sub(r"[^a-z0-9 ]+", " ", s).split()
    return (bool(toks) and toks[0] == "jong") or any(NON_SENIOR_TEAM_RE.match(t) for t in (toks[1:] or toks[-1:]))


def fixture_is_non_senior(home: str, away: str, league: str) -> bool:
    lg = _strip_accents((league or "")).lower().replace(".", " ")
    if NON_SENIOR_LEAGUE_RE.search(lg):
        return True
    return team_is_non_senior(home) or team_is_non_senior(away)


def league_country(league: str) -> Optional[str]:
    lg = _strip_accents((league or "")).lower()
    head = re.split(r"[\u2022/|]", lg)[0].strip()
    head = head.replace("amateur", "").strip()
    return COUNTRY_CODES.get(head)


# ═══════════════════════════════ CLUB ELO (bulk) ═══════════════════════════════
class EloBook:
    def __init__(self):
        self.lock = threading.Lock()
        self.index: Dict[str, List[Tuple[str, str, float]]] = {}   # norm -> [(name, country, elo)]
        self.loaded_at = 0.0
        self.last_try = 0.0
        self.last_error: Optional[str] = None
        self.match_cache: Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]] = {}

    def ensure_loaded(self, force: bool = False):
        now = time.time()
        if not force and self.index and now - self.loaded_at < ELO_TTL:
            return
        if not force and now - self.last_try < 600 and not self.index:
            return  # back off after a failure
        with self.lock:
            if not force and self.index and time.time() - self.loaded_at < ELO_TTL:
                return
            self.last_try = time.time()
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for scheme in ("https", "http"):
                try:
                    req = urllib.request.Request(f"{scheme}://api.clubelo.com/{day}",
                                                 headers={"User-Agent": "SoccerEngine/4.0"})
                    text = urllib.request.urlopen(req, timeout=25).read().decode("utf-8")
                    idx: Dict[str, List[Tuple[str, str, float]]] = {}
                    for row in csv.DictReader(io.StringIO(text)):
                        try:
                            elo = float(row["Elo"])
                        except (KeyError, ValueError, TypeError):
                            continue
                        name = row.get("Club", "")
                        idx.setdefault(norm(name), []).append((name, row.get("Country", ""), elo))
                    if len(idx) > 100:
                        self.index, self.loaded_at, self.last_error = idx, time.time(), None
                        self.match_cache.clear()
                        return
                    self.last_error = "ClubElo returned too few rows"
                except Exception as e:  # noqa
                    self.last_error = f"{scheme}: {e}"

    def lookup(self, raw_name: str, country: Optional[str]) -> Optional[Tuple[str, float]]:
        key = (raw_name, country)
        if key in self.match_cache:
            return self.match_cache[key]
        res = self._lookup(raw_name, country)
        self.match_cache[key] = res
        return res

    def _pick(self, entries, country):
        if country:
            same = [e for e in entries if e[1] == country]
            if same:
                return max(same, key=lambda e: e[2])
        if len(entries) == 1:
            return entries[0]
        return None

    def _lookup(self, raw_name: str, country: Optional[str]):
        if not self.index:
            return None
        n = norm(raw_name)
        if len(n) < 3:
            return None
        for cand in [n] + ALIASES.get(n, []):
            if cand in self.index:
                e = self._pick(self.index[cand], country)
                if e:
                    return e[0], e[2]
        if is_truncated(raw_name) and len(n) >= 6:
            hits = [k for k in self.index if k.startswith(n)]
            if len(hits) == 1:
                e = self._pick(self.index[hits[0]], country)
                if e:
                    return e[0], e[2]
            return None
        close = get_close_matches(n, list(self.index.keys()), n=2, cutoff=0.90)
        if close:
            if len(close) == 2 and country is None:
                return None  # ambiguous
            e = self._pick(self.index[close[0]], country)
            if e:
                return e[0], e[2]
        return None


ELO = EloBook()


def intl_elo(team: str) -> Optional[float]:
    return INTL_ELO_SEEDS.get(norm(team)) or INTL_ELO_SEEDS.get(_strip_accents(team).strip().lower())


# ═══════════════════════════════ MATH ═══════════════════════════════
def pois_pmf(lam: float, n: int) -> List[float]:
    p = [math.exp(-lam)]
    for k in range(1, n + 1):
        p.append(p[-1] * lam / k)
    return p


def tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    if x == 0 and y == 0:
        return max(1.0 - lh * la * rho, 0.01)
    if x == 0 and y == 1:
        return 1.0 + lh * rho
    if x == 1 and y == 0:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def match_probs(lh: float, la: float, rho: float = RHO, n: int = 10) -> Dict[str, float]:
    ph, pa = pois_pmf(lh, n), pois_pmf(la, n)
    h = d = a = ov = bt = tot = 0.0
    for x in range(n + 1):
        for y in range(n + 1):
            p = tau(x, y, lh, la, rho) * ph[x] * pa[y]
            tot += p
            if x > y:
                h += p
            elif x == y:
                d += p
            else:
                a += p
            if x + y > 2:
                ov += p
            if x > 0 and y > 0:
                bt += p
    return {"1": h / tot, "X": d / tot, "2": a / tot, "over25": ov / tot, "btts": bt / tot}


def elo_lambdas(gap: float, neutral: bool) -> Tuple[float, float]:
    mh, ma = (MU_NEUTRAL, MU_NEUTRAL) if neutral else (MU_HOME, MU_AWAY)
    return mh * math.exp(ELO_SCALE * gap), ma * math.exp(-ELO_SCALE * gap)


def fit_lambdas(p1: float, p2: float, rho: float) -> Tuple[float, float]:
    """Find (lh, la) with lh+la = TOTAL_GOALS whose home-minus-away win prob matches target."""
    target = p1 - p2
    lo, hi = 0.15, TOTAL_GOALS - 0.15
    for _ in range(28):
        mid = (lo + hi) / 2
        pr = match_probs(mid, TOTAL_GOALS - mid, rho)
        if pr["1"] - pr["2"] < target:
            lo = mid
        else:
            hi = mid
    lh = (lo + hi) / 2
    return lh, TOTAL_GOALS - lh


def devig(oh, od, oa) -> Optional[Dict[str, float]]:
    """Power-method de-vig (removes more margin from longshots than proportional)."""
    if not (oh and od and oa) or min(oh, od, oa) <= 1.0:
        return None
    q = [1 / oh, 1 / od, 1 / oa]
    lo, hi = 1.0, 3.0
    if sum(q) <= 1.0:
        s = sum(q)
        return {"1": q[0] / s, "X": q[1] / s, "2": q[2] / s}
    for _ in range(60):
        c = (lo + hi) / 2
        if sum(x ** c for x in q) > 1:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2
    p = [x ** c for x in q]
    s = sum(p)
    return {"1": p[0] / s, "X": p[1] / s, "2": p[2] / s}


# ═══════════════════════════════ CLASSIFICATION ═══════════════════════════════
INTL_RE = re.compile(r"\b(world cup|wcq|euros?|european championship|copa america|afcon|"
                     r"nations league|fifa|africa cup|asian cup|gold cup)\b")
CONT_RE = re.compile(r"\b(champions league|europa league|europa|conference league|libertadores|"
                     r"sudamericana|caf champions|afc champions)\b")
TOP5 = {"ENG": ["premier league"], "ESP": ["laliga", "la liga", "primera division"],
        "ITA": ["serie a"], "GER": ["bundesliga"], "FRA": ["ligue 1"]}


def classify_domain(league: str) -> str:
    lg = _strip_accents(league or "").lower()
    if INTL_RE.search(lg):
        return "international_tournament"
    if CONT_RE.search(lg):
        return "club_continental"
    cc = league_country(league)
    if cc in TOP5 and any(k in lg for k in TOP5[cc]) and not re.search(r"\b(2|ii|women|u\d\d)\b", lg):
        return "domestic_top5"
    return "other_league"


# ═══════════════════════════════ SCHEMAS ═══════════════════════════════
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


# ═══════════════════════════════ PREDICTION ═══════════════════════════════
def grade(p: float, quality: str) -> str:
    if quality == "prior":
        return "REJECT"
    trusted = quality in ("market+rating", "market")
    if trusted and p >= 0.65:
        return "ELITE"
    if trusted and p >= 0.55:
        return "STRONG"
    if p >= 0.42:
        return "CANDIDATE"
    return "REJECT"   # too close to call


def predict_single(f: FixtureInput) -> Dict[str, Any]:
    league = f.league or ""
    domain = classify_domain(league)
    is_intl = domain == "international_tournament"
    neutral = bool(f.is_neutral)
    rho = WC_RHO if is_intl else RHO

    elo_h = elo_a = None
    rating_note = ""
    if is_intl:
        elo_h, elo_a = intl_elo(f.home), intl_elo(f.away)
    elif not fixture_is_non_senior(f.home, f.away, league):
        cc = league_country(league)
        rh, ra = ELO.lookup(f.home, cc), ELO.lookup(f.away, cc)
        if rh and ra:
            elo_h, elo_a = rh[1], ra[1]
            rating_note = f"ClubElo: {rh[0]} {rh[1]:.0f} vs {ra[0]} {ra[1]:.0f}."
    else:
        rating_note = "Women's/youth/reserve fixture: no men's-club rating used."
    have_rating = elo_h is not None and elo_a is not None
    elo_gap = (elo_h - elo_a) if have_rating else None

    market = devig(f.odds_home, f.odds_draw, f.odds_away)
    elo_pr = None
    if have_rating:
        lh0, la0 = elo_lambdas(elo_gap, neutral)
        elo_pr = match_probs(lh0, la0, rho)

    mismatch = False
    if market and elo_pr:
        fav = max(market, key=market.get)
        if abs(market[fav] - elo_pr[fav]) > MAX_ELO_DISAGREE:
            mismatch = True   # rating almost certainly mis-matched or stale -> trust the market
            elo_pr = None

    if market and elo_pr:
        quality = "market+rating"
        probs = {k: MARKET_WEIGHT * market[k] + (1 - MARKET_WEIGHT) * elo_pr[k] for k in ("1", "X", "2")}
    elif market:
        quality = "market"
        probs = dict(market)
    elif elo_pr:
        quality = "rating"
        probs = {k: elo_pr[k] for k in ("1", "X", "2")}
    else:
        quality = "prior"
        mh, ma = (MU_NEUTRAL, MU_NEUTRAL) if neutral else (MU_HOME, MU_AWAY)
        pr = match_probs(mh, ma, rho)
        probs = {k: pr[k] for k in ("1", "X", "2")}

    s = sum(probs.values())
    probs = {k: v / s for k, v in probs.items()}
    lh, la = fit_lambdas(probs["1"], probs["2"], rho)
    goals = match_probs(lh, la, rho)

    pick = max(probs, key=probs.get)
    p_pick = probs[pick]
    odds_map = {"1": f.odds_home, "X": f.odds_draw, "2": f.odds_away}
    pick_odds = odds_map[pick]
    tier = grade(p_pick, quality)
    acca_ok = (quality in ("market+rating", "market") and p_pick >= ACCA_MIN_PROB
               and pick != "X" and bool(pick_odds) and pick_odds > 1.0)

    edges = {k: (probs[k] - market[k]) if market else None for k in ("1", "X", "2")}
    edge_pick = edges[pick]

    quality_text = {
        "market+rating": "de-vigged bookmaker odds (80%) blended with Elo (20%)",
        "market": "de-vigged bookmaker odds (no reliable Elo rating for this fixture)",
        "rating": "Elo rating only (no odds supplied) - unvalidated, not acca-eligible",
        "prior": "no odds and no rating - league-average prior only, essentially a coin flip",
    }[quality]
    reason = f"Basis: {quality_text}. {rating_note} " \
             f"{'Elo disagreed strongly with the market and was discarded. ' if mismatch else ''}" \
             f"Pick {pick} at {p_pick*100:.1f}%" \
             f"{f', model vs fair price {edge_pick*100:+.1f}%' if edge_pick is not None else ''}." \
             f"{' Too close to call.' if tier == 'REJECT' and quality != 'prior' else ''}"

    model_label = {
        "domestic_top5": "Poisson/Dixon-Coles fit to blended probabilities [Top-5 league]",
        "club_continental": "Poisson/Dixon-Coles fit to blended probabilities [Continental]",
        "international_tournament": "Poisson/Dixon-Coles fit to blended probabilities [International]",
    }.get(domain, "Poisson/Dixon-Coles fit to blended probabilities [Other league]")

    return {
        "home": f.home, "away": f.away, "league": f.league, "domain": domain,
        "data_quality": quality, "data_confidence": quality != "prior",
        "home_elo": elo_h, "away_elo": elo_a, "elo_gap": elo_gap,
        "lambda_home": lh, "lambda_away": la, "rho_used": rho,
        "p_home": probs["1"], "p_draw": probs["X"], "p_away": probs["2"], "p_pick": p_pick,
        "p_over25": goals["over25"], "p_btts_yes": goals["btts"],
        "edge_home": edges["1"], "edge_draw": edges["X"], "edge_away": edges["2"],
        "adj_edge": max(edge_pick, 0.0) if edge_pick is not None else 0.0,
        "pick": pick, "pick_odds": pick_odds, "fair_odds": round(1 / p_pick, 2),
        "confidence_tier": tier, "acca_eligible": acca_ok,
        "model_used": model_label, "reason": reason.strip(),
    }


def build_accas(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pool, seen = [], set()
    for p in sorted((p for p in preds if p["acca_eligible"]), key=lambda x: -x["p_pick"]):
        key = (p["home"].lower(), p["away"].lower())
        if key not in seen:
            seen.add(key)
            pool.append(p)
    accas = []
    for n in range(2, 7):
        if len(pool) < n:
            break
        combo = pool[:n]
        odds = prob = 1.0
        legs = []
        for c in combo:
            odds *= c["pick_odds"]
            prob *= c["p_pick"]
            legs.append({"home": c["home"], "away": c["away"], "league": c["league"],
                         "pick": c["pick"], "tier": c["confidence_tier"],
                         "odds": c["pick_odds"], "prob": round(c["p_pick"], 4)})
        accas.append({
            "n_legs": n, "combined_odds": round(odds, 2),
            "combined_model_prob": round(prob * 100, 1),
            "expected_value_pct": round((prob * odds - 1) * 100, 1),
            "avg_adj_edge": round(sum(c["adj_edge"] for c in combo) / n * 100, 1),
            "legs": legs,
        })
    return accas


# ═══════════════════════════════ ENDPOINTS ═══════════════════════════════
@app.on_event("startup")
def _warm():
    threading.Thread(target=ELO.ensure_loaded, daemon=True).start()


@app.get("/")
def root():
    return {"status": "online", "engine": "Soccer Intelligence Engine v4.0",
            "endpoints": ["/health", "/predict", "/predict/batch", "/fixtures/today",
                          "/elo/{team}", "/team-match", "/data-sources"]}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "4.0.0",
            "clubelo": {"clubs_loaded": len(ELO.index), "loaded_at": ELO.loaded_at,
                        "last_error": ELO.last_error},
            "active_keys": {"football_data": bool(FOOTBALL_DATA_KEY), "the_odds_api": bool(THE_ODDS_API_KEY),
                            "api_football": bool(API_FOOTBALL_KEY), "sportmonks": bool(SPORTMONKS_TOKEN)},
            "components": {"poisson_dixon_coles": True, "market_devig_power": True,
                           "club_elo_bulk": bool(ELO.index), "intl_elo_seeded": True,
                           "penaltyblog": False, "soccerdata": False,
                           "isotonic_calibration": False}}


@app.get("/data-sources")
def data_sources():
    return {"sources": [
        {"source": "api.clubelo.com (bulk)", "status": f"{len(ELO.index)} clubs loaded" if ELO.index else "NOT LOADED"},
        {"source": "Bookmaker odds (from your paste)", "status": "Primary signal, power-method de-vig"},
        {"source": "api.football-data.org", "status": "Key set" if FOOTBALL_DATA_KEY else "No key"},
        {"source": "TheOddsAPI", "status": "Key set (not used in predictions)" if THE_ODDS_API_KEY else "No key"},
        {"source": "eloratings.net", "status": "Seeded top ~35 nations only"},
    ]}


@app.post("/predict")
def predict_endpoint(fixture: FixtureInput):
    ELO.ensure_loaded()
    return predict_single(fixture)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    if len(request.fixtures) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 fixtures per request")
    ELO.ensure_loaded()
    preds = [predict_single(f) for f in request.fixtures]
    accas = build_accas(preds)
    elig = [p for p in preds if p["acca_eligible"]]
    return {
        "total_fixtures": len(preds),
        "actionable_picks": sum(1 for p in preds if p["confidence_tier"] in ("ELITE", "STRONG", "CANDIDATE")),
        "no_bet_count": sum(1 for p in preds if p["confidence_tier"] == "REJECT"),
        "acca_eligible_count": len(elig),
        "quality_counts": {q: sum(1 for p in preds if p["data_quality"] == q)
                           for q in ("market+rating", "market", "rating", "prior")},
        "predictions": preds, "accumulators": accas,
    }


@app.get("/fixtures/today")
def fixtures_today(league: str = "PL"):
    if not FOOTBALL_DATA_KEY:
        raise HTTPException(status_code=400, detail="FOOTBALL_DATA_API_KEY env var is not set")
    code = league.upper() if league.upper() in {"PL", "PD", "SA", "BL1", "FL1", "CL"} else "PL"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = f"https://api.football-data.org/v4/competitions/{code}/matches?dateFrom={today}&dateTo={today}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": FOOTBALL_DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"league": code, "fixtures": [
            {"home": m["homeTeam"]["name"], "away": m["awayTeam"]["name"],
             "kickoff": m.get("utcDate"), "status": m.get("status", "SCHEDULED")}
            for m in data.get("matches", [])]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch fixtures: {e}")


@app.get("/team-match")
def team_match(name: str, league: str = ""):
    """Debug: shows which ClubElo club (if any) a bookmaker team name resolves to."""
    ELO.ensure_loaded()
    r = ELO.lookup(name, league_country(league))
    return {"input": name, "normalised": norm(name), "non_senior_flag": team_is_non_senior(name),
            "matched_club": r[0] if r else None, "elo": r[1] if r else None,
            "clubs_loaded": len(ELO.index)}


@app.get("/elo/{team}")
def elo_endpoint(team: str, international: bool = False):
    if international:
        e = intl_elo(team)
        return {"team": team, "elo": e or 1700.0, "is_real_data": e is not None,
                "source": "eloratings.net (seeded)" if e else "NOT FOUND - default", "date": time.strftime("%Y-%m-%d")}
    ELO.ensure_loaded()
    r = ELO.lookup(team, None)
    return {"team": team, "elo": r[1] if r else 1650.0, "is_real_data": r is not None,
            "source": f"api.clubelo.com ({r[0]})" if r else "NOT FOUND - default", "date": time.strftime("%Y-%m-%d")}
