export type EmotionName =
  | "angry"
  | "fearful"
  | "happy"
  | "neutral"
  | "playfulness"
  | "sad"
  | "surprise"
  | string;

export interface EmotionCurvePoint {
  time?: number;
  emotion?: EmotionName;
  confidence?: number;
  probability?: number;
  [key: string]: unknown;
}

export interface EvidenceItem {
  keyword?: string;
  text?: string;
  reason?: string;
  value?: string;
  [key: string]: unknown;
}

export interface AudioAnalysis {
  asr_text?: string;
  audio_prediction?: EmotionName;
  text_prediction?: EmotionName;
  fusion_prediction?: EmotionName;
  audio_confidence?: number;
  text_confidence?: number;
  fusion_confidence?: number;
  audio_probabilities?: Record<string, number>;
  text_probabilities?: Record<string, number>;
  fusion_probabilities?: Record<string, number>;
  emotion_curve?: EmotionCurvePoint[];
  emotion_change_points?: EmotionCurvePoint[];
  keywords?: EvidenceItem[];
  possible_reasons?: EvidenceItem[];
}

export interface AgentFeedback {
  main_emotion_summary?: string;
  curve_interpretation?: string;
  possible_reasons?: string;
  companion_response?: string;
  gentle_suggestion?: string;
  safety_note?: string;
  provider?: string;
}

export interface AnalysisResponse {
  ok?: boolean;
  analysis?: AudioAnalysis;
  agent_feedback?: AgentFeedback;
}

export interface HealthResponse {
  deepseek?: { configured?: boolean };
}

export interface ChatHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export interface AgentChatRequest {
  message: string;
  session_id?: string;
  history?: ChatHistoryItem[];
  latest_analysis?: AudioAnalysis;
}

export interface AgentToolCall {
  name: string;
  arguments?: Record<string, unknown>;
  result?: Record<string, unknown>;
}

export interface AgentStreamToolEvent {
  name: string;
  arguments?: Record<string, unknown>;
  result?: Record<string, unknown>;
}

export interface AgentChatResponse {
  ok: boolean;
  response: string;
  provider: string;
  session_id: string;
  tool_calls: AgentToolCall[];
  fallback_reason?: string;
  search_required?: boolean;
}
