function chipText(item) {
  if (typeof item === "string") return item;
  return item?.keyword || item?.text || item?.reason || item?.value || JSON.stringify(item);
}

export default function TextEvidence({ analysis }) {
  const keywords = Array.isArray(analysis.keywords) ? analysis.keywords : [];
  const reasons = Array.isArray(analysis.possible_reasons) ? analysis.possible_reasons : [];
  return (
    <section className="text-card">
      <h2 className="section-title">ASR 转写文本</h2>
      <p className="asr-text">{analysis.asr_text || "未识别到明确文本。"}</p>
      <h2 className="section-title">关键词</h2>
      <div className="chips">
        {keywords.length > 0 ? keywords.map((item, index) => <span className="chip" key={index}>{chipText(item)}</span>) : <span className="chip">暂无明确关键词</span>}
      </div>
      <h2 className="section-title">可能原因线索</h2>
      <div className="chips">
        {reasons.length > 0 ? reasons.map((item, index) => <span className="chip reason" key={index}>{chipText(item)}</span>) : <span className="chip reason">暂无明确原因线索</span>}
      </div>
    </section>
  );
}
