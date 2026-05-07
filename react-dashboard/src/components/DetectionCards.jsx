function ItemCard({ label, active }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <p className={`mt-2 text-xl font-semibold ${active ? "text-emerald-300" : "text-slate-300"}`}>
        {active ? "Yes" : "No"}
      </p>
    </div>
  );
}

function hasObject(objects, names) {
  return objects.some((obj) => names.includes((obj.label || "").toLowerCase()));
}

function DetectionCards({ detections }) {
  const objects = detections?.objects || [];
  const tomatoDetected = hasObject(objects, ["tomato"]);
  const pepperDetected = hasObject(objects, ["pepper", "felfel"]);
  const humanDetected = !!detections?.human?.detected;

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <ItemCard label="Tomato detected" active={tomatoDetected} />
      <ItemCard label="Pepper detected" active={pepperDetected} />
      <ItemCard label="Human detected" active={humanDetected} />
    </div>
  );
}

export default DetectionCards;
