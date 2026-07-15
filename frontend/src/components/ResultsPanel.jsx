import EmotionCurve from "./EmotionCurve";
import EmotionSummary from "./EmotionSummary";
import Feedback from "./Feedback";
import TextEvidence from "./TextEvidence";

export default function ResultsPanel({ result, isLoading, error }) {
  if (isLoading) {
    return (
      <section className="panel results" aria-live="polite">
        <div className="loading-layout">
          <span />
          <span />
          <span />
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel results" aria-live="polite">
        <div className="empty-state error-state">
          <div>
            <strong>分析没有完成</strong>
            <span>{error}</span>
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="panel results" aria-live="polite">
        <div className="empty-state">
          <div>
            <strong>准备好后，我会把结果整理成清晰的陪伴分析</strong>
            <span>包括主要情绪、窗口变化、关键词线索和具体建议。</span>
          </div>
        </div>
      </section>
    );
  }

  const analysis = result.analysis || {};
  return (
    <section className="panel results" aria-live="polite">
      <EmotionSummary analysis={analysis} />
      <EmotionCurve analysis={analysis} />
      <Feedback feedback={result.agent_feedback} />
      <TextEvidence analysis={analysis} />
    </section>
  );
}
