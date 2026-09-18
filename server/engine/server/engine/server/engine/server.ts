import express, { Request, Response } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';

import { FixtureInput, BatchPredictionResponse } from './src/types';
import { predictFixture, buildAccumulators } from './server/engine/predictor';
import { parseUniversalSlip } from './server/engine/parser';
import { HISTORICAL_SLIPS } from './server/engine/autopsy';
import { lookupElo } from './server/engine/eloDatabase';
import { researchMatchWithAi } from './server/gemini';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// ═══════════════════════════════════════════════════════════════════════════════
// API ROUTES
// ═══════════════════════════════════════════════════════════════════════════════

app.get('/api/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    engine: 'Soccer Intelligence Engine v4.0 (Anti-Longshot & Bayesian Calibrated)',
    version: '4.0.0',
    capabilities: {
      dixon_coles: true,
      bivariate_poisson: true,
      shin_devigging: true,
      bayesian_market_shrinkage: true,
      anti_longshot_protection: true,
      slip_autopsy_diagnostics: true,
      gemini_live_search: Boolean(process.env.GEMINI_API_KEY),
    },
    active_keys: {
      gemini_ai: Boolean(process.env.GEMINI_API_KEY),
    },
  });
});

app.post('/api/predict', (req: Request, res: Response) => {
  try {
    const fixture: FixtureInput = req.body;
    if (!fixture.home || !fixture.away) {
      res.status(400).json({ error: 'home and away team names are required' });
      return;
    }
    const result = predictFixture(fixture);
    res.json(result);
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Prediction failed' });
  }
});

app.post('/api/predict/batch', (req: Request, res: Response) => {
  try {
    const { fixtures } = req.body as { fixtures: FixtureInput[] };
    if (!Array.isArray(fixtures)) {
      res.status(400).json({ error: 'fixtures must be an array' });
      return;
    }
    if (fixtures.length > 1000) {
      res.status(400).json({ error: 'Maximum 1000 fixtures allowed per batch' });
      return;
    }

    const predictions = fixtures.map(f => predictFixture(f));
    const actionable = predictions.filter(p => p.pick !== 'NO BET');
    const noBetCount = predictions.filter(p => p.pick === 'NO BET').length;
    const longshotTrapsBlocked = predictions.filter(p => p.is_longshot_trap).length;
    const accaEligibleCount = predictions.filter(p => p.acca_eligible).length;

    const accumulators = buildAccumulators(predictions);

    const response: BatchPredictionResponse = {
      total_fixtures: predictions.length,
      actionable_picks: actionable.length,
      no_bet_count: noBetCount,
      longshot_traps_blocked: longshotTrapsBlocked,
      acca_eligible_count: accaEligibleCount,
      predictions,
      accumulators,
    };

    res.json(response);
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Batch prediction failed' });
  }
});

app.get('/api/sample-277', (req: Request, res: Response) => {
  const topTeams = [
    ['Manchester City', 'Arsenal'], ['Real Madrid', 'Barcelona'], ['Liverpool', 'Chelsea'],
    ['Bayern Munich', 'Bayer Leverkusen'], ['Inter', 'Juventus'], ['Paris Saint-Germain', 'Monaco'],
    ['Atletico Madrid', 'Sevilla'], ['Sporting CP', 'Porto'], ['Atalanta', 'Milan'],
    ['Tottenham', 'Newcastle'], ['Aston Villa', 'Manchester United'], ['Borussia Dortmund', 'RB Leipzig'],
    ['Lazio', 'Roma'], ['Real Sociedad', 'Athletic Club'], ['Ajax', 'PSV'],
    ['Feyenoord', 'Benfica'], ['Galatasaray', 'Fenerbahce'], ['Flamengo', 'Palmeiras'],
    ['River Plate', 'Boca Juniors'], ['Celtic', 'Rangers']
  ];

  const intlTeams = [
    ['Argentina', 'France'], ['Spain', 'England'], ['Brazil', 'Colombia'],
    ['Portugal', 'Netherlands'], ['Germany', 'Italy'], ['Uruguay', 'Belgium'],
    ['Morocco', 'Senegal'], ['Japan', 'South Korea'], ['United States', 'Mexico'],
    ['Croatia', 'Switzerland'], ['Nigeria', 'Ivory Coast'], ['Egypt', 'Ghana']
  ];

  const fixtures: FixtureInput[] = [];
  let id = 1;

  for (const [h, a] of topTeams) {
    fixtures.push({
      id: `slate-${id}`,
      home: h,
      away: a,
      league: 'Elite European Top Flight',
      odds_home: parseFloat((1.55 + ((id % 5) * 0.18)).toFixed(2)),
      odds_draw: 3.40,
      odds_away: parseFloat((3.60 + ((id % 7) * 0.30)).toFixed(2)),
    });
    id++;
  }

  for (const [h, a] of intlTeams) {
    fixtures.push({
      id: `slate-${id}`,
      home: h,
      away: a,
      league: 'International Tournament',
      odds_home: parseFloat((1.75 + ((id % 4) * 0.15)).toFixed(2)),
      odds_draw: 3.30,
      odds_away: parseFloat((3.90 + ((id % 6) * 0.25)).toFixed(2)),
    });
    id++;
  }

  const leagues = [
    'England Championship', 'Spain Segunda', 'Italy Serie B', 'Germany 2. Bundesliga',
    'France Ligue 2', 'Brazil Serie A', 'MLS', 'Colombia Primera A', 'Ecuador Liga Pro'
  ];
  const cities = ['Bristol', 'Preston', 'Oviedo', 'Zaragoza', 'Brescia', 'Modena', 'Nurnberg', 'Dusseldorf', 'Grenoble', 'Pau', 'Santos', 'Gremio', 'Orlando', 'Dallas', 'Medellin', 'Cali', 'Quito', 'Cuenca'];

  while (fixtures.length < 277) {
    const lg = leagues[id % leagues.length];
    const c1 = cities[id % cities.length];
    const c2 = cities[(id + 3) % cities.length];
    fixtures.push({
      id: `slate-${id}`,
      home: `${c1} FC #${id}`,
      away: `${c2} United #${id + 1}`,
      league: lg,
      odds_home: parseFloat((1.45 + ((id % 8) * 0.20)).toFixed(2)),
      odds_draw: parseFloat((3.15 + ((id % 4) * 0.10)).toFixed(2)),
      odds_away: parseFloat((2.80 + ((id % 6) * 0.35)).toFixed(2)),
    });
    id++;
  }

  res.json({ count: fixtures.length, fixtures });
});

app.post('/api/parse-slip', (req: Request, res: Response) => {
  try {
    const { text } = req.body as { text: string };
    if (!text) {
      res.status(400).json({ error: 'text is required' });
      return;
    }
    const fixtures = parseUniversalSlip(text);
    res.json({ count: fixtures.length, fixtures });
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'Failed to parse slip text' });
  }
});

app.get('/api/autopsy/:slipId', (req: Request, res: Response) => {
  const slipId = req.params.slipId.toUpperCase();
  const autopsy = HISTORICAL_SLIPS[slipId];
  if (!autopsy) {
    res.status(404).json({ error: `Slip ${slipId} not found. Available slips: JZ98TA, JDE66A` });
    return;
  }
  res.json(autopsy);
});

app.get('/api/elo/:team', (req: Request, res: Response) => {
  const team = req.params.team;
  const isIntl = req.query.international === 'true';
  const data = lookupElo(team, isIntl);
  res.json({
    team,
    elo: data.elo,
    is_real_data: data.is_real,
    tier: data.tier,
    source: data.source,
    date: new Date().toISOString().split('T')[0],
  });
});

app.post('/api/ai/match-intel', async (req: Request, res: Response) => {
  try {
    const { home, away, league } = req.body as { home: string; away: string; league?: string };
    if (!home || !away) {
      res.status(400).json({ error: 'home and away are required' });
      return;
    }
    const analysis = await researchMatchWithAi(home, away, league);
    res.json(analysis);
  } catch (err: any) {
    res.status(500).json({ error: err.message || 'AI Research failed' });
  }
});

app.get('/api/fixtures/preset', (req: Request, res: Response) => {
  res.json({
    presets: [
      {
        category: 'Premier League Clashes',
        fixtures: [
          { home: 'Arsenal', away: 'Chelsea', league: 'Premier League', odds_home: 1.85, odds_draw: 3.65, odds_away: 4.20 },
          { home: 'Manchester City', away: 'Liverpool', league: 'Premier League', odds_home: 2.15, odds_draw: 3.50, odds_away: 3.30 },
          { home: 'Aston Villa', away: 'Tottenham', league: 'Premier League', odds_home: 2.30, odds_draw: 3.60, odds_away: 2.90 },
          { home: 'Brighton', away: 'West Ham', league: 'Premier League', odds_home: 1.95, odds_draw: 3.70, odds_away: 3.80 },
        ],
      },
    ],
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Soccer Intelligence Engine v4.0 running on http://localhost:${PORT}`);
  });
}

startServer();
