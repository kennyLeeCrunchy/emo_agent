export default function Feedback({ feedback }) {
  if (!feedback || typeof feedback !== "object") return null;
  return (
    <section className="feedback-card">
      <h2 className="section-title">陪伴反馈</h2>
      <div className="feedback-highlight">{feedback.companion_response || "暂未生成陪伴回应。"}</div>
      <div className="feedback-grid">
        <div className="mini-card">
          <strong>主要情绪</strong>
          {feedback.main_emotion_summary || "暂无"}
        </div>
        <div className="mini-card">
          <strong>情绪曲线</strong>
          {feedback.curve_interpretation || "暂无"}
        </div>
        <div className="mini-card">
          <strong>可能原因</strong>
          {feedback.possible_reasons || "暂无"}
        </div>
        <div className="mini-card">
          <strong>温和建议</strong>
          {feedback.gentle_suggestion || "暂无"}
        </div>
      </div>
      <p className="hint">{feedback.safety_note || ""}</p>
    </section>
  );
}
