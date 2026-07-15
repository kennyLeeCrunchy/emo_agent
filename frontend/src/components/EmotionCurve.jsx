import { emotionInfo, formatPercent } from "../emotions";

function getPointValue(point, emotion) {
  if (typeof point?.confidence === "number") return point.confidence;
  if (typeof point?.probability === "number") return point.probability;
  if (typeof point?.[emotion] === "number") return point[emotion];
  return 0;
}

export default function EmotionCurve({ analysis }) {
  const curve = Array.isArray(analysis.emotion_curve) ? analysis.emotion_curve : [];
  const emotion = analysis.fusion_prediction || "unknown";
  const info = emotionInfo(emotion);

  if (curve.length < 2) {
    return (
      <section className="curve-card">
        <h2 className="section-title">情绪窗口变化</h2>
        <div className="curve-note">当前音频没有返回足够的窗口曲线数据。可以换一段更长的语音再试。</div>
      </section>
    );
  }

  const width = 720;
  const height = 220;
  const pad = 24;
  const maxIndex = Math.max(1, curve.length - 1);
  const points = curve.map((point, index) => {
    const x = pad + (index / maxIndex) * (width - pad * 2);
    const y = height - pad - Math.max(0, Math.min(1, getPointValue(point, emotion))) * (height - pad * 2);
    return { x, y, raw: point };
  });
  const line = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");
  const area = `${line} L ${points[points.length - 1].x.toFixed(1)} ${height - pad} L ${points[0].x.toFixed(1)} ${height - pad} Z`;

  return (
    <section className="curve-card">
      <h2 className="section-title">情绪窗口变化</h2>
      <div className="curve-wrap">
        <svg className="curve-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="情绪窗口曲线">
          <path d={area} className="curve-area" fill={info.color} />
          <path d={line} className="curve-line" stroke={info.color} />
          {points.map((point, index) => (
            <circle className="curve-point" key={index} cx={point.x} cy={point.y} r="5" fill={info.color} />
          ))}
        </svg>
      </div>
      <div className="window-list">
        {curve.slice(0, 5).map((point, index) => (
          <div className="window-row" key={index}>
            <span>{typeof point.time === "number" ? `${point.time.toFixed(1)}s` : `窗口 ${index + 1}`}</span>
            <strong>{emotionInfo(point.emotion || emotion).label}</strong>
            <span className="track"><span className="fill" style={{ width: `${getPointValue(point, emotion) * 100}%`, background: info.color }} /></span>
            <span>{formatPercent(getPointValue(point, emotion))}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
