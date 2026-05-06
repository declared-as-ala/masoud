function StatCard({ title, value, accent }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
      <p className="text-xs uppercase tracking-wide text-slate-400">{title}</p>
      <p className={`mt-2 text-2xl font-bold ${accent}`}>{value}</p>
    </div>
  );
}

function StatsCards({ stats }) {
  return (
    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
      <StatCard title="Total Tomatoes" value={stats.tomatoes || 0} accent="text-emerald-400" />
      <StatCard title="Total Peppers" value={stats.peppers || 0} accent="text-lime-400" />
      <StatCard title="Bad Vegetables" value={stats.bad_vegetables || 0} accent="text-rose-400" />
      <StatCard
        title="Human Detected"
        value={stats.human_detected ? "Yes" : "No"}
        accent={stats.human_detected ? "text-amber-400" : "text-slate-200"}
      />
    </div>
  );
}

export default StatsCards;
