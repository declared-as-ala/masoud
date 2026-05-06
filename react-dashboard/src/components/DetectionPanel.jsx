function DetectionPanel({ detections, humanDetected }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4 shadow-xl">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">
          Recent Detections
        </h2>
        <span
          className={`rounded-full px-2 py-1 text-xs ${
            humanDetected ? "bg-amber-500/20 text-amber-300" : "bg-slate-700 text-slate-300"
          }`}
        >
          Human: {humanDetected ? "Yes" : "No"}
        </span>
      </div>

      {detections.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-600 p-4 text-sm text-slate-400">
          No detections yet. Check camera/model status.
        </p>
      ) : (
        <div className="space-y-2">
          {detections.map((item, idx) => (
            <div
              key={`${item.class_name}-${idx}`}
              className="rounded-lg border border-slate-700 bg-slate-950/50 p-3"
            >
              <p className="text-sm font-medium text-slate-100">{item.label}</p>
              <p className="mt-1 text-xs text-slate-400">
                Confidence: {Math.round((item.confidence || 0) * 100)}%
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DetectionPanel;
