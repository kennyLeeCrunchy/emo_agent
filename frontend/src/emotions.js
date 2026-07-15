export const emotionMeta = {
  angry: { label: "愤怒", color: "#d95757" },
  fearful: { label: "害怕", color: "#8067d8" },
  happy: { label: "开心", color: "#d99422" },
  neutral: { label: "中性", color: "#3f83b8" },
  playfulness: { label: "玩笑感", color: "#239b7a" },
  sad: { label: "难过", color: "#4d70ca" },
  surprise: { label: "惊讶", color: "#ce6698" },
  unknown: { label: "未知", color: "#7c94a8" }
};

export function emotionInfo(name) {
  return emotionMeta[name] || { label: name || "未知", color: emotionMeta.unknown.color };
}

export function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "暂无";
  return `${(value * 100).toFixed(value >= 0.995 ? 2 : 1)}%`;
}
