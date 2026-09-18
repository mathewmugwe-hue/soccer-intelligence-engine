export type Outcome = '1' | 'X' | '2';
export type ConfidenceTier = 'ELITE' | 'STRONG' | 'VALUE' | 'SPECULATIVE' | 'NO BET';

export interface FixtureInput {
  id?: string;
  home: string;
  away: string;
  league?: string;
  match_date?: string;
  odds_home?: number | null;
  odds_draw?: number | null;
  odds_away?: number | null;
  is_neutral?: boolean;
  tournament_phase?: string;
}

export interface ScoreProb {
  homeGoals: number;
  awayGoals: number;
  probability: number;
}

export interface MarketOdds {
  raw_1?: number | null;
  raw_X?: number | null;
  raw_2?: number | null;
  fair_1: number;
  fair_X: number;
  fair_2: number;
  overround: number;
}

export interface PredictionResult {
  home: string;
  away: string;
  league: string;
  domain: string;
  data_confidence: boolean;
  confidence_rating: number;
  home_elo: number;
  away_elo: number;
  elo_gap: number;
  is_home_real: boolean;
  is_away_real: boolean;

  // Expected goals (Dixon-Coles lambda)
  lambda_home: number;
  lambda_away: number;
  rho_used: number;

  // Joint model probabilities
  p_home: number;
  p_draw: number;
  p_away: number;

  // Consensus (Bayesian regularized) probabilities
  consensus_home: number;
  consensus_draw: number;
  consensus_away: number;

  // Secondary markets
  p_over15: number;
  p_over25: number;
  p_over35: number;
  p_btts_yes: number;
  p_1X: number;
  p_X2: number;
  p_12: number;
  p_dnb_home: number;
  p_dnb_away: number;

  // Edges vs de-vigged bookmaker odds
  edge_home: number | null;
  edge_draw: number | null;
  edge_away: number | null;
  adj_edge: number;

  // Decision & sizing
  pick: Outcome | 'NO BET';
  primary_pick?: string;
  primary_win_prob?: number;
  recommended_market: string;
  pick_odds: number | null;
  expected_value: number | null;
  kelly_stake_pct: number | null;
  confidence_tier: ConfidenceTier;
  acca_eligible: boolean;
  is_longshot_trap: boolean;

  model_used: string;
  reason: string;
  warning?: string;

  top_scores: ScoreProb[];
}

export interface AccaLeg {
  home: string;
  away: string;
  league: string;
  pick: string;
  tier: ConfidenceTier;
  odds: number;
  model_prob: number;
  edge: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface Accumulator {
  id: string;
  name: string;
  n_legs: number;
  combined_odds: number;
  combined_model_prob: number;
  expected_value: number;
  risk_tier: 'BANKER' | 'VALUE' | 'SAFE' | 'AGGRESSIVE';
  legs: AccaLeg[];
  recommendation_note: string;
}

export interface BatchPredictionResponse {
  total_fixtures: number;
  actionable_picks: number;
  no_bet_count: number;
  longshot_traps_blocked: number;
  acca_eligible_count: number;
  predictions: PredictionResult[];
  accumulators: Accumulator[];
}

export interface SlipAutopsyEvent {
  home: string;
  away: string;
  score?: string;
  pick: string;
  odds: number;
  outcome: string;
  won: boolean;
  reason_failed: string;
  was_longshot_trap: boolean;
  was_draw_trap: boolean;
  was_unrated_team: boolean;
}

export interface SlipAutopsyResult {
  slip_id: string;
  timestamp: string;
  total_events: number;
  won_events: number;
  lost_events: number;
  total_odds: number;
  stake_kes: number;
  payout_kes: number;
  mathematical_win_chance_pct: number;
  primary_root_causes: string[];
  events: SlipAutopsyEvent[];
  structural_recommendations: string[];
}

export interface AiMatchAnalysis {
  match: string;
  league: string;
  timestamp: string;
  summary: string;
  key_factors: string[];
  injury_and_lineup_news: string[];
  h2h_trend: string;
  tactical_verdict: string;
  model_validation_assessment: string;
  sources?: string[];
}
