/**
 * Fetch wrapper for the Gomoku Platform FastAPI backend.
 * In dev mode Vite proxies /api/* to http://localhost:8000.
 */

import type {
  AnalyzeRequest,
  AnalyzeResponse,
  BestMoveRequest,
  BestMoveResponse,
  EngineInfo,
  SuggestRequest,
  SuggestResponse,
} from "./types";

const BASE = (import.meta.env.VITE_API_URL as string) || "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function analyze(body: AnalyzeRequest): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function bestMove(body: BestMoveRequest): Promise<BestMoveResponse> {
  return request<BestMoveResponse>("/best-move", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function suggest(body: SuggestRequest): Promise<SuggestResponse> {
  return request<SuggestResponse>("/suggest", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function engineInfo(): Promise<EngineInfo> {
  return request<EngineInfo>("/engine/info");
}
