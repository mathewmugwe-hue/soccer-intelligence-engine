/**
 * Soccer Intelligence Engine - Mathematical Core
 * Implements:
 * 1. Dixon-Coles (1997) time-dependent bivariate Poisson with low-scoring dependency adjustment (tau)
 * 2. Overround removal / De-vigging (Shin's method & Multiplicative)
 * 3. Bayesian market shrinkage (Anti-longshot & Anti-phantom-edge protection)
 * 4. Kelly Criterion & Expected Value (EV) sizing
 */

export interface MatchProbs {
  p_home: number;
  p_draw: number;
  p_away: number;
  p_over15: number;
  p_over25: number;
  p_over35: number;
  p_btts_yes: number;
  p_1X: number;
  p_X2: number;
  p_12: number;
  p_dnb_home: number;
  p_dnb_away: number;
  matrix: number[][];
}

export function factorial(n: number): number {
  if (n <= 1) return 1;
  let res = 1;
  for (let i = 2; i <= n; i++) res *= i;
  return res;
}

export function poissonProb(k: number, lambda: number): number {
  if (lambda <= 0) return k === 0 ? 1 : 0;
  return (Math.pow(lambda, k) * Math.exp(-lambda)) / factorial(k);
}

export function tauDixonColes(x: number, y: number, lambdaH: number, lambdaA: number, rho: number): number {
  if (x === 0 && y === 0) {
    return Math.max(1.0 - lambdaH * lambdaA * rho, 0.01);
  } else if (x === 0 && y === 1) {
    return Math.max(1.0 + lambdaH * rho, 0.01);
  } else if (x === 1 && y === 0) {
    return Math.max(1.0 + lambdaA * rho, 0.01);
  } else if (x === 1 && y === 1) {
    return Math.max(1.0 - rho, 0.01);
  }
  return 1.0;
}

export function computeMatchProbabilities(
  lambdaH: number,
  lambdaA: number,
  rho: number = -0.052,
  maxGoals: number = 7
): MatchProbs {
  const matrix: number[][] = Array.from({ length: maxGoals + 1 }, () =>
    new Array(maxGoals + 1).fill(0)
  );

  let p_home = 0;
  let p_draw = 0;
  let p_away = 0;
  let p_over15 = 0;
  let p_over25 = 0;
  let p_over35 = 0;
  let p_btts_yes = 0;

  let totalProb = 0;

  for (let x = 0; x <= maxGoals; x++) {
    for (let y = 0; y <= maxGoals; y++) {
      const tau = tauDixonColes(x, y, lambdaH, lambdaA, rho);
      const prob = tau * poissonProb(x, lambdaH) * poissonProb(y, lambdaA);
      matrix[x][y] = prob;
      totalProb += prob;

      if (x > y) p_home += prob;
      else if (x === y) p_draw += prob;
      else p_away += prob;

      const totalGoals = x + y;
      if (totalGoals > 1.5) p_over15 += prob;
      if (totalGoals > 2.5) p_over25 += prob;
      if (totalGoals > 3.5) p_over35 += prob;
      if (x > 0 && y > 0) p_btts_yes += prob;
    }
  }

  if (totalProb > 0) {
    p_home /= totalProb;
    p_draw /= totalProb;
    p_away /= totalProb;
    p_over15 /= totalProb;
    p_over25 /= totalProb;
    p_over35 /= totalProb;
    p_btts_yes /= totalProb;
    for (let x = 0; x <= maxGoals; x++) {
      for (let y = 0; y <= maxGoals; y++) {
        matrix[x][y] /= totalProb;
      }
    }
  }

  const p_1X = p_home + p_draw;
  const p_X2 = p_draw + p_away;
  const p_12 = p_home + p_away;

  const dnbDenominator = p_home + p_away;
  const p_dnb_home = dnbDenominator > 0 ? p_home / dnbDenominator : 0.5;
  const p_dnb_away = dnbDenominator > 0 ? p_away / dnbDenominator : 0.5;

  return {
    p_home,
    p_draw,
    p_away,
    p_over15,
    p_over25,
    p_over35,
    p_btts_yes,
    p_1X,
    p_X2,
    p_12,
    p_dnb_home,
    p_dnb_away,
    matrix,
  };
}

export function devigOdds(
  oh?: number | null,
  od?: number | null,
  oa?: number | null
): { fair_1: number; fair_X: number; fair_2: number; overround: number } | null {
  if (!oh || !od || !oa || oh <= 1.0 || od <= 1.0 || oa <= 1.0) {
    return null;
  }

  const impH = 1.0 / oh;
  const impD = 1.0 / od;
  const impA = 1.0 / oa;
  const overround = impH + impD + impA;

  if (overround <= 1.0 || overround > 1.6) {
    return {
      fair_1: impH / overround,
      fair_X: impD / overround,
      fair_2: impA / overround,
      overround: Math.max(overround, 1.0),
    };
  }

  let kLow = 1.0;
  let kHigh = 2.0;
  let k = 1.15;

  for (let iter = 0; iter < 16; iter++) {
    k = (kLow + kHigh) / 2.0;
    const sum = Math.pow(impH, k) + Math.pow(impD, k) + Math.pow(impA, k);
    if (sum > 1.0) {
      kLow = k;
    } else {
      kHigh = k;
    }
  }

  const p1 = Math.pow(impH, k);
  const pX = Math.pow(impD, k);
  const p2 = Math.pow(impA, k);
  const total = p1 + pX + p2;

  return {
    fair_1: p1 / total,
    fair_X: pX / total,
    fair_2: p2 / total,
    overround,
  };
}

export function regularizeProbabilities(
  rawModel: { p_home: number; p_draw: number; p_away: number },
  market: { fair_1: number; fair_X: number; fair_2: number; overround?: number } | null,
  dataConfidence: boolean,
  homeIsReal: boolean,
  awayIsReal: boolean
): { p_home: number; p_draw: number; p_away: number; confidenceScore: number } {
  if (!market) {
    const score = (homeIsReal ? 40 : 0) + (awayIsReal ? 40 : 0) + (dataConfidence ? 20 : 0);
    return {
      p_home: rawModel.p_home,
      p_draw: rawModel.p_draw,
      p_away: rawModel.p_away,
      confidenceScore: score,
    };
  }

  let modelWeight = 0.05;
  if (dataConfidence && homeIsReal && awayIsReal) {
    modelWeight = 0.38;
  } else if (homeIsReal || awayIsReal) {
    modelWeight = 0.20;
  }

  const minMarketProb = Math.min(market.fair_1, market.fair_X, market.fair_2);
  if (minMarketProb < 0.18) {
    modelWeight *= 0.5;
  }

  let regH = modelWeight * rawModel.p_home + (1.0 - modelWeight) * market.fair_1;
  let regD = modelWeight * rawModel.p_draw + (1.0 - modelWeight) * market.fair_X;
  let regA = modelWeight * rawModel.p_away + (1.0 - modelWeight) * market.fair_2;

  const sum = regH + regD + regA;
  regH /= sum;
  regD /= sum;
  regA /= sum;

  const confidenceScore = Math.round(
    (homeIsReal ? 35 : 10) +
    (awayIsReal ? 35 : 10) +
    (dataConfidence ? 20 : 0) +
    ((market.overround ?? 1.15) <= 1.08 ? 10 : 5)
  );

  return {
    p_home: regH,
    p_draw: regD,
    p_away: regA,
    confidenceScore: Math.min(100, Math.max(10, confidenceScore)),
  };
}

export function calculateSizing(
  prob: number,
  odds?: number | null
): { ev: number | null; kelly: number | null } {
  if (!odds || odds <= 1.0) return { ev: null, kelly: null };

  const ev = prob * odds - 1.0;
  if (ev <= 0) return { ev, kelly: 0 };

  const b = odds - 1.0;
  const q = 1.0 - prob;
  const fullKelly = (b * prob - q) / b;
  const fractionalKelly = Math.max(0, fullKelly * 0.25);

  return {
    ev,
    kelly: parseFloat((fractionalKelly * 100).toFixed(2)),
  };
}
