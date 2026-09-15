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
  session_id?: string;
  analysis?: AudioAnalysis;
  agent_feedback?: AgentFeedback;
}

export interface HealthResponse {
  deepseek?: { configured?: boolean };
  memory?: { configured?: boolean; backend?: string };
}

export interface ChatHistoryItem {
  role: "user" | "assistant";
  content: string;
}

export interface AgentChatRequest {
  message: string;
  user_id?: string;
  session_id?: string;
  request_id?: string;
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
  request_id?: string;
  tool_calls: AgentToolCall[];
  fallback_reason?: string;
  search_required?: boolean;
}

export interface SessionRecord {
  session_id: string;
  user_id: string;
  title: string;
  current_emotion?: string;
  current_confidence?: number;
  latest_analysis?: AudioAnalysis;
  preview?: string;
  message_count?: number;
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  emotion?: string;
  created_at: string;
}

export interface SessionDetailResponse {
  ok: boolean;
  session: SessionRecord;
  messages: StoredMessage[];
}

export interface MemoryTrendPoint {
  date: string;
  dominant_emotion?: string;
  count: number;
  average_confidence?: number;
}

export interface MemoryTrend {
  available: boolean;
  scope: "memory_trend";
  range_days: number;
  event_count: number;
  dominant_emotion?: string;
  emotion_counts: Record<string, number>;
  common_triggers: Array<{ text: string; count: number }>;
  points: MemoryTrendPoint[];
  note: string;
}

export interface MemoryPreference {
  key: string;
  value: string;
  source: string;
  confirmed: boolean;
  updated_at: string;
}

export interface MemorySummaryResponse {
  ok: boolean;
  user_id: string;
  short_term: {
    session?: SessionRecord;
    recent_messages: StoredMessage[];
  };
  long_term: {
    trend: MemoryTrend;
    preferences: MemoryPreference[];
  };
  safety_note: string;
}
