function AlertsList({ alerts = [] }) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-4">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-300">Recent alerts</h2>
      {alerts.length === 0 ? (
        <p className="text-sm text-slate-400">No alerts yet.</p>
      ) : (
        <div className="space-y-2">
          {alerts.slice(0, 8).map((alert, index) => (
            <div key={`${alert.timestamp}-${index}`} className="rounded-lg border border-slate-700 p-3">
              <p className="text-sm font-medium text-rose-300">{alert.message}</p>
              <p className="text-xs text-slate-400">{alert.timestamp}</p>
              {alert.error ? <p className="mt-1 text-xs text-amber-300">Telegram error: {alert.error}</p> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default AlertsList;
