export const emotionMeta: Record<string, { label: string; color: string; glow: string }> = {
  angry: { label: "愤怒", color: "#fb7185", glow: "rgba(251,113,133,.28)" },
  fearful: { label: "害怕", color: "#a78bfa", glow: "rgba(167,139,250,.28)" },
  happy: { label: "开心", color: "#fbbf24", glow: "rgba(251,191,36,.25)" },
  neutral: { label: "平静", color: "#60a5fa", glow: "rgba(96,165,250,.25)" },
  playfulness: { label: "轻松", color: "#34d399", glow: "rgba(52,211,153,.25)" },
  sad: { label: "难过", color: "#818cf8", glow: "rgba(129,140,248,.28)" },
  surprise: { label: "惊讶", color: "#f472b6", glow: "rgba(244,114,182,.25)" },
  unknown: { label: "待分析", color: "#94a3b8", glow: "rgba(148,163,184,.2)" }
};

export function emotionInfo(name?: string) {
  return emotionMeta[name || "unknown"] || {
    label: name || "待分析",
    color: emotionMeta.unknown.color,
    glow: emotionMeta.unknown.glow
  };
}

export function formatPercent(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) return "暂无";
  return `${(value * 100).toFixed(value >= 0.995 ? 2 : 1)}%`;
}
