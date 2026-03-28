/** TypeScript interfaces matching packages/shared/schemas. */

export type CellValue = 0 | 1 | -1;
export type Player = 1 | -1;

export interface Position {
  boardSize: number;
  winLength: number;
  currentPlayer: Player;
  cells: CellValue[];
  lastMove: number;
  variant?: string;
}

export interface MoveCandidate {
  move: number;
  score: number;
  confidence?: number;
  row?: number;
  col?: number;
}

export interface EngineMeta {
  ttHitRate?: number;
  ttSize?: number;
  rawScore?: number;
}

export type EngineSource =
  | "safety_win"
  | "safety_block"
  | "safety_multi_block"
  | "vcf_win"
  | "vcf_defense"
  | "fork"
  | "alpha_beta";

// -- Analyze --

export interface AnalyzeRequest {
  position: Position;
  topK?: number;
  timeLimitMs?: number;
  includePv?: boolean;
}

export interface AnalyzeResponse {
  bestMove: number;
  value: number;
  confidence: number;
  source: EngineSource;
  depth: number;
  nodesSearched: number;
  timeMs: number;
  topMoves: MoveCandidate[];
  pvLine: number[];
  policy?: number[];
  engineMeta?: EngineMeta;
}

// -- Best move --

export interface BestMoveRequest {
  position: Position;
  timeLimitMs?: number;
}

export interface BestMoveResponse {
  move: number;
  row: number;
  col: number;
  value: number;
  source: EngineSource;
}

// -- Suggest --

export interface SuggestRequest {
  position: Position;
  topK?: number;
}

export interface SuggestResponse {
  suggestions: MoveCandidate[];
}

// -- Engine info --

export interface EngineInfo {
  version: string;
  supportedBoardSizes: number[];
  capabilities: string[];
}

// -- Training --

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type TrainPhase = "tactical" | "bootstrap" | "self_play" | "training" | "evaluating";

export interface TrainJobConfig {
  variant?: string;
  batchSize?: number;
  epochs?: number;
  lr?: number;
  selfPlayGames?: number;
  selfPlaySimulations?: number;
  resumeFromCheckpoint?: string;
}

export interface TrainJobProgress {
  phase: TrainPhase;
  epoch: number;
  totalEpochs: number;
  loss: number;
  policyAccuracy: number;
  valueMae: number;
  gamesGenerated: number;
  positionsCollected: number;
  elapsedSec: number;
}

export interface TrainJob {
  jobId: string;
  variant: string;
  status: JobStatus;
  config?: TrainJobConfig;
  progress?: TrainJobProgress;
  createdAt: string;
  updatedAt?: string;
  completedAt?: string;
  artifactId?: string;
  error?: string;
}

export interface ModelArtifact {
  artifactId: string;
  name: string;
  version: string;
  format: "onnx" | "pytorch" | "checkpoint";
  boardSizes: number[];
  inputShape?: number[];
  createdAt: string;
  metrics?: Record<string, number>;
}
