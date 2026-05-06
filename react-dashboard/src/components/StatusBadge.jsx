function StatusBadge({ label, color = "green" }) {
  const classes =
    color === "green"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-600/50"
      : "bg-rose-500/20 text-rose-300 border-rose-600/50";

  return (
    <span className={`rounded-full border px-3 py-1 text-xs font-medium ${classes}`}>{label}</span>
  );
}

export default StatusBadge;
