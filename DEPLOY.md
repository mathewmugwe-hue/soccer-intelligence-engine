# SOCCER INTELLIGENCE ENGINE — DEPLOYMENT GUIDE
# Zero LLM · Pure Mathematics · Render + Vercel

## WHAT YOU GET
- Render backend: FastAPI prediction engine (Dixon-Coles + Poisson + Elo)
- Vercel frontend: Full dashboard (predict, batch, accas, Elo lookup)
- Free data: api.clubelo.com + football-data.co.uk + understat.com
- Optional live data: API-Football + SportMonks + TheOddsAPI

---

## STEP 1 — DEPLOY BACKEND TO RENDER

1. Go to https://render.com and sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo (upload these files first)
   OR use "Deploy from existing repo" if you push to GitHub
4. Settings:
   - Name: soccer-engine
   - Runtime: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
   - Plan: Free (or Starter $7/mo for always-on)

5. Environment Variables (Render Dashboard → Environment):
   - FOOTBALL_DATA_API_KEY = get free key at football-data.org/register
   - THE_ODDS_API_KEY = get free key at theoddsapi.com (optional)
   - API_FOOTBALL_KEY = get at apifootball.com (optional)
   - SPORTMONKS_KEY = get at sportmonks.com (optional)
   - CACHE_DIR = /tmp/soccer_cache
   - MODEL_DIR = /tmp/soccer_models

6. Click "Create Web Service"
7. Wait 3-5 minutes for build
8. Your backend URL: https://soccer-engine.onrender.com (or similar)
9. Test: visit https://your-url.onrender.com/health

---

## STEP 2 — DEPLOY FRONTEND TO VERCEL

1. Go to https://vercel.com and sign up (free)
2. Click "Add New" → "Project"
3. Import your GitHub repo
   OR use Vercel CLI: npx vercel deploy
4. Framework: Other (static)
5. Root directory: / (where index.html lives)
6. Click Deploy

7. Your frontend URL: https://soccer-engine.vercel.app (or similar)

8. IMPORTANT: Update the API_BASE in index.html
   Line ~330: const API_BASE = "https://YOUR-RENDER-URL.onrender.com";
   OR: click the URL badge top-right in the app to enter your Render URL

---

## STEP 3 — FREE DATA KEYS

### football-data.org (RECOMMENDED — enables today's fixtures)
1. Visit https://www.football-data.org/client/register
2. Register free account
3. Copy your API key
4. Set FOOTBALL_DATA_API_KEY in Render

### TheOddsAPI (enables multi-book odds)
1. Visit https://the-odds-api.com
2. Register free account (500 requests/month)
3. Copy your API key
4. Set THE_ODDS_API_KEY in Render

---

## STEP 4 — TEST YOUR DEPLOYMENT

### Health check:
curl https://your-render-url.onrender.com/health

### Single prediction:
curl -X POST https://your-render-url.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"home":"Arsenal","away":"Chelsea","league":"Premier League","odds_home":2.10,"odds_draw":3.40,"odds_away":3.60}'

### Batch prediction + accas:
curl -X POST https://your-render-url.onrender.com/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"fixtures":[{"home":"Arsenal","away":"Chelsea","league":"Premier League","odds_home":2.10,"odds_draw":3.40,"odds_away":3.60},{"home":"France","away":"Brazil","league":"FIFA WCQ","odds_home":2.80,"odds_draw":3.10,"odds_away":2.60,"is_neutral":true}]}'

---

## UPGRADING THE MODEL

### Add real xG (SportMonks):
Set SPORTMONKS_KEY → automatic, no code changes needed

### Add live lineups + injuries (API-Football):
Set API_FOOTBALL_KEY → automatic, no code changes needed

### Add multi-book odds (TheOddsAPI):
Set THE_ODDS_API_KEY → automatic, no code changes needed

### Retrain the ML layer (after 90+ days of live data):
python main.py --retrain  # (not yet implemented — add after data accumulates)

---

## FILE STRUCTURE

soccer-engine/
├── main.py          → Render backend (FastAPI + full engine)
├── requirements.txt → Python dependencies
├── render.yaml      → Render auto-deploy config
├── index.html       → Vercel frontend (complete dashboard)
├── vercel.json      → Vercel deploy config
├── .env.example     → Environment variable template
└── DEPLOY.md        → This file

---

## FREE TIER LIMITS

Render free:
- 512MB RAM (enough for Dixon-Coles + LightGBM)
- Spins down after 15 min inactivity (first request takes ~30s to wake)
- Upgrade to Starter ($7/mo) for always-on

Vercel free:
- Unlimited static hosting
- 100GB bandwidth/month
- No limits for this use case

football-data.org free:
- 10 requests/minute
- Top 12 European leagues
- Fixtures, standings, results

TheOddsAPI free:
- 500 requests/month
- Multi-book odds from 50+ bookmakers

---

## REPOS POWERING THIS ENGINE

| Library          | Stars  | Role                                      |
|------------------|--------|-------------------------------------------|
| penaltyblog      | ★900+  | Dixon-Coles, Bivariate Poisson, Elo       |
| soccerdata       | ★1.7k  | Club Elo, FBref, Understat scrapers       |
| scikit-learn     | ★60k+  | Isotonic calibration, Brier score         |
| lightgbm         | ★16k+  | Gradient boosting ML layer                |
| jfjelstul/worldcup | ★300+ | WC historical data (Elo calibration)     |
| xgabora dataset  | ★400+  | Club football 2000-2025                   |
| FastAPI          | ★75k+  | API framework                             |
| diskcache        | ★2k+   | Persistent disk caching (Render)          |

---

## ZERO LLM GUARANTEE

This engine produces predictions using exclusively:
- Dixon-Coles statistical model (research-fitted rho)
- International Elo (WC-calibrated, K=40, neutral HFA=0)
- Club Elo (api.clubelo.com daily updates)
- Bivariate Poisson goal expectancy
- Isotonic probability calibration
- LightGBM on historical match features

No language model. No hallucination. No imagination.
Every probability is a mathematical output of the above models.
