import { emotionInfo, formatPercent } from "../emotions";

function ProbabilityRows({ probabilities }) {
  if (!probabilities || typeof probabilities !== "object") {
    return <p className="hint">暂无概率分布。</p>;
  }
  return (
    <div className="bar-list">
      {Object.entries(probabilities)
        .sort((a, b) => Number(b[1]) - Number(a[1]))
        .map(([name, value]) => {
          const info = emotionInfo(name);
          const numeric = Number(value) || 0;
          return (
            <div className="bar-row" style={{ "--emotion-color": info.color }} key={name}>
              <span>{info.label}</span>
              <span className="track">
                <span className="fill" style={{ width: `${Math.max(0, Math.min(100, numeric * 100))}%` }} />
              </span>
              <span>{formatPercent(numeric)}</span>
            </div>
          );
        })}
    </div>
  );
}

function PredictionCard({ title, prediction, confidence, probabilities }) {
  const info = emotionInfo(prediction || "unknown");
  return (
    <article className="emotion-card" style={{ "--emotion-color": info.color }}>
      <div className="card-label">{title}</div>
      <div className="emotion-name">
        <span>{info.label}</span>
        <span className="confidence">{formatPercent(confidence)}</span>
      </div>
      <ProbabilityRows probabilities={probabilities} />
    </article>
  );
}

export default function EmotionSummary({ analysis }) {
  return (
    <section className="summary-grid">
      <PredictionCard
        title="声学情绪"
        prediction={analysis.audio_prediction}
        confidence={analysis.audio_confidence}
        probabilities={analysis.audio_probabilities}
      />
      <PredictionCard
        title="文本情绪"
        prediction={analysis.text_prediction}
        confidence={analysis.text_confidence}
        probabilities={analysis.text_probabilities}
      />
      <PredictionCard
        title="融合情绪"
        prediction={analysis.fusion_prediction}
        confidence={analysis.fusion_confidence}
        probabilities={analysis.fusion_probabilities}
      />
    </section>
  );
}
