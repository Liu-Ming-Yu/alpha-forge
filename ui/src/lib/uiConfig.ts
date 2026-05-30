export const TIME = {
  secondMs: 1000,
  minuteMs: 60 * 1000,
} as const;

export const QUERY_DEFAULTS = {
  retryCount: 1,
} as const;

export const QUERY_TIMING = {
  liveStaleMs: TIME.secondMs / 2,
  relaxedStaleMs: 20 * TIME.secondMs,
  relaxedRefetchMs: 30 * TIME.secondMs,
  standardStaleMs: TIME.minuteMs,
  cacheGcMs: 5 * TIME.minuteMs,
  commandCatalogStaleMs: 5 * TIME.minuteMs,
  jobPollMs: 2 * TIME.secondMs,
  focusedJobPollMs: TIME.secondMs + TIME.secondMs / 2,
  commandValidationDebounceMs: 400,
  clipboardResetMs: TIME.secondMs + 400,
  clockTickMs: TIME.secondMs,
  immediateMs: 0,
} as const;

export const REQUEST_LIMITS = {
  strategyRuns: 20,
  signalContributions: 200,
  signalContributionAttribution: 300,
  unmatchedFills: 50,
  complianceSinceHours: 24,
  compliance: 50,
  audit: 50,
  researchCampaigns: 20,
  featureAudits: 40,
} as const;

export const DISPLAY_LIMITS = {
  alphaFamilyFeatureNames: 6,
  overviewLoadingCards: 4,
  overviewFreshnessRows: 6,
  overviewAuditRows: 10,
  overviewInstrumentIdChars: 14,
  settingsConfigFields: 8,
  researchPromotionFields: 6,
  researchReadinessFields: 7,
  researchCampaignRows: 8,
  researchAuditRows: 8,
  executionLoadingCards: 3,
  executionComplianceRows: 8,
  systemLoadingCards: 3,
  commandOptionalExpandedArgs: 6,
  backtestRangeSuggestions: 4,
  shortIdChars: 8,
  execIdChars: 10,
} as const;

export const REPLAY_TIMING = {
  backtestCurveMs: 2200,
  shimmerMs: 600,
} as const;

export const FORMATS = {
  isoDateLength: 10,
} as const;

export const BACKTEST_COMMAND_SIGNATURE = {
  requiredDests: ["contracts_file", "start", "end", "model_version"],
  preferredDests: ["campaign_top_n"],
} as const;

export function intervalLabel(ms: number): string {
  if (ms % TIME.minuteMs === 0) return `${ms / TIME.minuteMs}m`;
  if (ms % TIME.secondMs === 0) return `${ms / TIME.secondMs}s`;
  return `${ms}ms`;
}
