function HumanStatus({ human }) {
  let text = "No human";
  let cls = "text-slate-300";

  if (human?.detected && human?.authorized === true) {
    text = `Authorized: ${human?.name || "Known user"}`;
    cls = "text-emerald-300";
  } else if (human?.detected && human?.authorized === false) {
    text = "Unauthorized";
    cls = "text-rose-300";
  } else if (human?.detected) {
    text = "Human detected (checking face...)";
    cls = "text-amber-300";
  }

  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-400">Human status</p>
      <p className={`mt-2 text-lg font-semibold ${cls}`}>{text}</p>
    </div>
  );
}

export default HumanStatus;
