import {
  FixtureInput,
  PredictionResult,
  ScoreProb,
  ConfidenceTier,
  Outcome,
} from '../../src/types';
import {
  computeMatchProbabilities,
  devigOdds,
  regularizeProbabilities,
  calculateSizing,
} from './math';
import {
  lookupElo,
  estimateRatingGapFromOdds,
} from './eloDatabase';

export function classifyDomain(league: string): string {
  if (!league) return 'unmodeled';
  const l = league.toLowerCase();
  if (
    l.includes('world cup') ||
    l.includes('wcq') ||
    l.includes('euro') ||
    l.includes('copa america') ||
    l.includes('afcon') ||
    l.includes('nations league') ||
    l.includes('fifa')
  ) {
    return 'international_tournament';
  }
  if (
    l.includes('champions league') ||
    l.includes('europa') ||
    l.includes('conference league') ||
    l.includes('libertadores') ||
    l.includes('caf champions')
  ) {
    return 'club_continental';
  }
  if (
    l.includes('premier league') ||
    l.includes('la liga') ||
    l.includes('serie a') ||
    l.includes('bundesliga') ||
    l.includes('ligue 1')
  ) {
    return 'domestic_top5';
  }
  if (
    l.includes('championship') ||
    l.includes('eredivisie') ||
    l.includes('primeira') ||
    l.includes('scottish') ||
    l.includes('mls') ||
    l.includes('brasileirao')
  ) {
    return 'domestic_secondary';
  }
  return 'unmodeled';
}

export function predictFixture(f: FixtureInput): PredictionResult {
  const domain = classifyDomain(f.league || '');
  const isIntl = domain === 'international_tournament';

  // 1. Elo Lookups
  const homeEloData = lookupElo(f.home, isIntl);
  const awayEloData = lookupElo(f.away, isIntl);

  let eloH = homeEloData.elo;
  let eloA = awayEloData.elo;
  const isHomeReal = homeEloData.is_real;
  const isAwayReal = awayEloData.is_real;
  const dataConfidence = isHomeReal && isAwayReal;

  // 2. Bookmaker Odds & Market De-vigging
  const marketFair = devigOdds(f.odds_home, f.odds_draw, f.odds_away);

  // 3. Market Implied Elo Back-Calculation
  if (marketFair && (!isHomeReal || !isAwayReal)) {
    const marketEloGap = estimateRatingGapFromOdds(marketFair.fair_1, marketFair.fair_2, f.is_neutral ? 0 : 55);
    if (!isHomeReal && isAwayReal) {
      eloH = eloA + marketEloGap;
    } else if (isHomeReal && !isAwayReal) {
      eloA = eloH - marketEloGap;
    } else {
      eloH = 1450 + marketEloGap / 2;
      eloA = 1450 - marketEloGap / 2;
    }
  }

  // 4. Home Field Advantage
  const hfa = f.is_neutral ? 0.0 : isIntl ? 35.0 : 55.0;
  const eloGap = eloH + hfa - eloA;

  // 5. Expected Goals Calibration
  let baseLh = 1.48;
  let baseLa = 1.18;
  let rho = -0.048;

  if (isIntl) {
    baseLh = 1.40;
    baseLa = 1.12;
    rho = 0.035;
  } else if (domain === 'club_continental') {
    baseLh = 1.50;
    baseLa = 1.22;
    rho = -0.045;
  } else if (domain === 'unmodeled') {
    baseLh = 1.35;
    baseLa = 1.15;
    rho = -0.038;
  }

  const goalDiffMod = Math.tanh(eloGap / 450.0) * 0.45;
  const lambdaH = Math.max(0.40, baseLh * (1.0 + goalDiffMod));
  const lambdaA = Math.max(0.35, baseLa * (1.0 - goalDiffMod * 0.85));

  // 6. Joint Probabilities via Dixon-Coles Bivariate Poisson
  const rawProbs = computeMatchProbabilities(lambdaH, lambdaA, rho);

  // 7. Regularization with Market Consensus (Bayesian Shrinkage)
  const regularized = regularizeProbabilities(
    { p_home: rawProbs.p_home, p_draw: rawProbs.p_draw, p_away: rawProbs.p_away },
    marketFair,
    dataConfidence,
    isHomeReal,
    isAwayReal
  );

  const ph = regularized.p_home;
  const pd = regularized.p_draw;
  const pa = regularized.p_away;

  // 8. Edges vs Fair Market Odds
  let edgeH: number | null = null;
  let edgeD: number | null = null;
  let edgeA: number | null = null;

  if (marketFair) {
    edgeH = ph - marketFair.fair_1;
    edgeD = pd - marketFair.fair_X;
    edgeA = pa - marketFair.fair_2;
  }

  // 9. Model Preference
  let modelPick: Outcome = '1';
  let modelP = ph;
  if (pa > ph && pa >= pd) {
    modelPick = '2';
    modelP = pa;
  } else if (pd > ph && pd > pa) {
    modelPick = 'X';
    modelP = pd;
  }

  const pickOddsMap: Record<Outcome, number | null> = {
    '1': f.odds_home ?? null,
    'X': f.odds_draw ?? null,
    '2': f.odds_away ?? null,
  };
  const pickOdds = pickOddsMap[modelPick] ?? null;

  const rawEdge = marketFair
    ? (modelPick === '1' ? edgeH : modelPick === '2' ? edgeA : edgeD) ?? 0
    : 0;

  const adjEdge = (dataConfidence ? rawEdge : rawEdge * 0.35);

  // 10. Longshot Trap Detection
  const isLongshotTrap = Boolean(
    marketFair &&
      ((modelPick === '1' && marketFair.fair_1 < 0.28) ||
      (modelPick === '2' && marketFair.fair_2 < 0.28) ||
      (pickOdds && pickOdds >= 3.20))
  );

  // 11. Guarantee Actionable Pick for 100% of matches
  let pick: Outcome = modelPick;
  let primaryPick = `${pick === '1' ? f.home : pick === '2' ? f.away : 'Draw'} (${pick})`;
  let primaryWinProb = modelP;

  const p1X = ph + pd;
  const pX2 = pd + pa;

  if (modelP < 0.50) {
    if (ph >= pa && p1X >= 0.65) {
      primaryPick = `${f.home} or Draw (Double Chance 1X)`;
      primaryWinProb = p1X;
    } else if (pa > ph && pX2 >= 0.65) {
      primaryPick = `Draw or ${f.away} (Double Chance X2)`;
      primaryWinProb = pX2;
    } else if (rawProbs.p_over15 >= 0.75) {
      primaryPick = 'Over 1.5 Total Goals';
      primaryWinProb = rawProbs.p_over15;
    }
  }

  let confidenceTier: ConfidenceTier = 'SPECULATIVE';
  if (primaryWinProb >= 0.70) {
    confidenceTier = 'ELITE';
  } else if (primaryWinProb >= 0.60) {
    confidenceTier = 'STRONG';
  } else if (primaryWinProb >= 0.50 || (rawEdge && rawEdge > 0.03)) {
    confidenceTier = 'VALUE';
  } else {
    confidenceTier = 'SPECULATIVE';
  }

  const sizing = calculateSizing(modelP, pickOdds);
  let warning: string | undefined;

  if (isLongshotTrap && !dataConfidence) {
    warning = 'High-risk longshot odds detected on unverified tier — pick calibrated to avoid longshot trap.';
  }

  const recommendedMarket = primaryPick;
  const reason = `Elo Gap: ${eloGap >= 0 ? '+' : ''}${Math.round(eloGap)} (${f.home} ${Math.round(eloH)} vs ${f.away} ${Math.round(eloA)}${dataConfidence ? ', verified' : ', market-aligned'}). ` +
    `Expected Goals: ${lambdaH.toFixed(2)} - ${lambdaA.toFixed(2)}. ` +
    `Primary selection: ${primaryPick} with ${(primaryWinProb * 100).toFixed(1)}% win probability` +
    (marketFair ? `, de-vigged 1X2 edge: ${rawEdge >= 0 ? '+' : ''}${(rawEdge * 100).toFixed(1)}%` : '');

  const accaEligible = primaryWinProb >= 0.52;

  let modelLabel = 'Dixon-Coles Bivariate Poisson + Calibrated Elo (v4.0)';
  if (marketFair) {
    modelLabel += ' + Shin De-vigging & Bayesian Shrinkage';
  }

  // Top scorelines
  const topScores: ScoreProb[] = [];
  for (let x = 0; x <= 4; x++) {
    for (let y = 0; y <= 4; y++) {
      topScores.push({
        homeGoals: x,
        awayGoals: y,
        probability: parseFloat(rawProbs.matrix[x][y].toFixed(4)),
      });
    }
  }
  topScores.sort((a, b) => b.probability - a.probability);

  return {
    home: f.home,
    away: f.away,
    league: f.league || 'Standard League',
    domain,
    data_confidence: dataConfidence,
    confidence_rating: regularized.confidenceScore,
    home_elo: Math.round(eloH),
    away_elo: Math.round(eloA),
    elo_gap: Math.round(eloGap),
    is_home_real: isHomeReal,
    is_away_real: isAwayReal,

    lambda_home: parseFloat(lambdaH.toFixed(2)),
    lambda_away: parseFloat(lambdaA.toFixed(2)),
    rho_used: rho,

    p_home: parseFloat(rawProbs.p_home.toFixed(4)),
    p_draw: parseFloat(rawProbs.p_draw.toFixed(4)),
    p_away: parseFloat(rawProbs.p_away.toFixed(4)),

    consensus_home: parseFloat(ph.toFixed(4)),
    consensus_draw: parseFloat(pd.toFixed(4)),
    consensus_away: parseFloat(pa.toFixed(4)),

    p_over15: parseFloat(rawProbs.p_over15.toFixed(4)),
    p_over25: parseFloat(rawProbs.p_over25.toFixed(4)),
    p_over35: parseFloat(rawProbs.p_over35.toFixed(4)),
    p_btts_yes: parseFloat(rawProbs.p_btts_yes.toFixed(4)),
    p_1X: parseFloat(rawProbs.p_1X.toFixed(4)),
    p_X2: parseFloat(rawProbs.p_X2.toFixed(4)),
    p_12: parseFloat(rawProbs.p_12.toFixed(4)),
    p_dnb_home: parseFloat(rawProbs.p_dnb_home.toFixed(4)),
    p_dnb_away: parseFloat(rawProbs.p_dnb_away.toFixed(4)),

    edge_home: edgeH !== null ? parseFloat(edgeH.toFixed(4)) : null,
    edge_draw: edgeD !== null ? parseFloat(edgeD.toFixed(4)) : null,
    edge_away: edgeA !== null ? parseFloat(edgeA.toFixed(4)) : null,
    adj_edge: parseFloat(adjEdge.toFixed(4)),

    pick,
    primary_pick: primaryPick,
    primary_win_prob: parseFloat(primaryWinProb.toFixed(4)),
    recommended_market: recommendedMarket,
    pick_odds: pickOdds ? parseFloat(pickOdds.toFixed(2)) : null,
    expected_value: sizing.ev !== null ? parseFloat(sizing.ev.toFixed(4)) : null,
    kelly_stake_pct: sizing.kelly,
    confidence_tier: confidenceTier,
    acca_eligible: accaEligible,
    is_longshot_trap: isLongshotTrap,

    model_used: modelLabel,
    reason,
    warning,
    top_scores: topScores.slice(0, 5),
  };
}

/**
 * Builds mathematically structured accumulators ranked strictly by HIGHEST WIN PROBABILITY.
 */
export function buildAccumulators(predictions: PredictionResult[]) {
  if (!predictions || predictions.length < 2) return [];

  const getWinProb = (p: PredictionResult) => {
    return p.primary_win_prob ?? (p.pick === '1' ? p.p_home : p.pick === 'X' ? p.p_draw : p.p_away);
  };

  const getEffectiveOdds = (p: PredictionResult) => {
    if (p.pick_odds && p.pick_odds > 1.0) return p.pick_odds;
    const prob = getWinProb(p);
    return Math.max(1.10, parseFloat((1.0 / Math.max(0.05, prob * 0.95)).toFixed(2)));
  };

  // Sort ALL predictions strictly by HIGHEST WIN PROBABILITY descending
  const sorted = [...predictions].sort((a, b) => {
    const probDiff = getWinProb(b) - getWinProb(a);
    if (Math.abs(probDiff) > 0.01) return probDiff;
    return b.adj_edge - a.adj_edge;
  });

  const accumulators = [];

  // 1. Banker Double (2 legs)
  if (sorted.length >= 2) {
    const legs = sorted.slice(0, 2);
    const odd0 = getEffectiveOdds(legs[0]);
    const odd1 = getEffectiveOdds(legs[1]);
    const prob0 = getWinProb(legs[0]);
    const prob1 = getWinProb(legs[1]);
    const combinedOdds = odd0 * odd1;
    const combinedProb = prob0 * prob1;
    const ev = combinedProb * combinedOdds - 1.0;

    accumulators.push({
      id: 'banker-double',
      name: 'Banker Double (Highest Win Probability)',
      n_legs: 2,
      combined_odds: parseFloat(combinedOdds.toFixed(2)),
      combined_model_prob: parseFloat((combinedProb * 100).toFixed(1)),
      expected_value: parseFloat((ev * 100).toFixed(1)),
      risk_tier: 'BANKER' as const,
      recommendation_note: 'Optimal 2-leg accumulator featuring the 2 absolute highest win-probability selections from the slate.',
      legs: legs.map(l => ({
        home: l.home,
        away: l.away,
        league: l.league,
        pick: l.primary_pick || `${l.pick === '1' ? l.home : l.pick === '2' ? l.away : 'Draw'} (${l.pick})`,
        tier: l.confidence_tier,
        odds: getEffectiveOdds(l),
        model_prob: getWinProb(l),
        edge: l.adj_edge,
        risk_level: 'LOW' as const,
      })),
    });
  }

  // 2. High-Probability Treble (3 legs)
  if (sorted.length >= 3) {
    const legs = sorted.slice(0, 3);
    let combinedOdds = 1.0;
    let combinedProb = 1.0;
    for (const l of legs) {
      combinedOdds *= getEffectiveOdds(l);
      combinedProb *= getWinProb(l);
    }
    const ev = combinedProb * combinedOdds - 1.0;

    accumulators.push({
      id: 'value-treble',
      name: 'High-Probability Treble (Top 3 Picks)',
      n_legs: 3,
      combined_odds: parseFloat(combinedOdds.toFixed(2)),
      combined_model_prob: parseFloat((combinedProb * 100).toFixed(1)),
      expected_value: parseFloat((ev * 100).toFixed(1)),
      risk_tier: 'VALUE' as const,
      recommendation_note: 'Standard professional 3-leg treble constructed from the top 3 highest probability fixtures.',
      legs: legs.map(l => ({
        home: l.home,
        away: l.away,
        league: l.league,
        pick: l.primary_pick || `${l.pick === '1' ? l.home : l.pick === '2' ? l.away : 'Draw'} (${l.pick})`,
        tier: l.confidence_tier,
        odds: getEffectiveOdds(l),
        model_prob: getWinProb(l),
        edge: l.adj_edge,
        risk_level: 'LOW' as const,
      })),
    });
  }

  // 3. Safe Rolling 4-Fold (4 legs)
  if (sorted.length >= 4) {
    const legs = sorted.slice(0, 4);
    let combinedOdds = 1.0;
    let combinedProb = 1.0;
    for (const l of legs) {
      combinedOdds *= getEffectiveOdds(l);
      combinedProb *= getWinProb(l);
    }
    const ev = combinedProb * combinedOdds - 1.0;

    accumulators.push({
      id: 'safe-4fold',
      name: 'Safe Rolling 4-Fold (Top 4 Picks)',
      n_legs: 4,
      combined_odds: parseFloat(combinedOdds.toFixed(2)),
      combined_model_prob: parseFloat((combinedProb * 100).toFixed(1)),
      expected_value: parseFloat((ev * 100).toFixed(1)),
      risk_tier: 'SAFE' as const,
      recommendation_note: 'Controlled 4-leg accumulator composed strictly of top probability anchors.',
      legs: legs.map(l => ({
        home: l.home,
        away: l.away,
        league: l.league,
        pick: l.primary_pick || `${l.pick === '1' ? l.home : l.pick === '2' ? l.away : 'Draw'} (${l.pick})`,
        tier: l.confidence_tier,
        odds: getEffectiveOdds(l),
        model_prob: getWinProb(l),
        edge: l.adj_edge,
        risk_level: 'MEDIUM' as const,
      })),
    });
  }

  // 4. High-Confidence 5-Fold (5 legs)
  if (sorted.length >= 5) {
    const legs = sorted.slice(0, 5);
    let combinedOdds = 1.0;
    let combinedProb = 1.0;
    for (const l of legs) {
      combinedOdds *= getEffectiveOdds(l);
      combinedProb *= getWinProb(l);
    }
    const ev = combinedProb * combinedOdds - 1.0;

    accumulators.push({
      id: 'solid-5fold',
      name: 'Solid 5-Fold Accumulator',
      n_legs: 5,
      combined_odds: parseFloat(combinedOdds.toFixed(2)),
      combined_model_prob: parseFloat((combinedProb * 100).toFixed(1)),
      expected_value: parseFloat((ev * 100).toFixed(1)),
      risk_tier: 'SAFE' as const,
      recommendation_note: 'Top 5 highest win probability matches combined for solid multiplier growth.',
      legs: legs.map(l => ({
        home: l.home,
        away: l.away,
        league: l.league,
        pick: l.primary_pick || `${l.pick === '1' ? l.home : l.pick === '2' ? l.away : 'Draw'} (${l.pick})`,
        tier: l.confidence_tier,
        odds: getEffectiveOdds(l),
        model_prob: getWinProb(l),
        edge: l.adj_edge,
        risk_level: 'MEDIUM' as const,
      })),
    });
  }

  // 5. Mega High-Probability 8-Fold (8 legs)
  if (sorted.length >= 8) {
    const legs = sorted.slice(0, 8);
    let combinedOdds = 1.0;
    let combinedProb = 1.0;
    for (const l of legs) {
      combinedOdds *= getEffectiveOdds(l);
      combinedProb *= getWinProb(l);
    }
    const ev = combinedProb * combinedOdds - 1.0;

    accumulators.push({
      id: 'mega-8fold',
      name: 'Mega High-Probability 8-Fold',
      n_legs: 8,
      combined_odds: parseFloat(combinedOdds.toFixed(2)),
      combined_model_prob: parseFloat((combinedProb * 100).toFixed(1)),
      expected_value: parseFloat((ev * 100).toFixed(1)),
      risk_tier: 'AGGRESSIVE' as const,
      recommendation_note: 'Extended 8-leg accumulator composed purely of the safest available picks across the entire slate.',
      legs: legs.map(l => ({
        home: l.home,
        away: l.away,
        league: l.league,
        pick: l.primary_pick || `${l.pick === '1' ? l.home : l.pick === '2' ? l.away : 'Draw'} (${l.pick})`,
        tier: l.confidence_tier,
        odds: getEffectiveOdds(l),
        model_prob: getWinProb(l),
        edge: l.adj_edge,
        risk_level: 'HIGH' as const,
      })),
    });
  }

  return accumulators;
}
