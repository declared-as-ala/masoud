import { useEffect, useMemo, useRef, useState } from "react";
import CameraStream from "./components/CameraStream";
import DetectionPanel from "./components/DetectionPanel";
import StatsCards from "./components/StatsCards";
import StatusBadge from "./components/StatusBadge";

const API_URL = import.meta.env.VITE_RPI_API_URL || "http://127.0.0.1:8000";

function App() {
  const [snapshot, setSnapshot] = useState({
    detections: [],
    stats: {
      tomatoes: 0,
      peppers: 0,
      bad_vegetables: 0,
      human_detected: false,
    },
    model_ready: false,
    model_note: "",
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  const recentDetections = useMemo(() => {
    return [...snapshot.detections]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8);
  }, [snapshot.detections]);

  useEffect(() => {
    let mounted = true;
    let pollId = null;
    let wsPingInterval = null;

    const fetchDetections = async () => {
      try {
        const response = await fetch(`${API_URL}/detections`);
        if (!response.ok) {
          throw new Error("detections request failed");
        }
        const data = await response.json();
        if (mounted) {
          setSnapshot(data);
          setConnected(true);
        }
      } catch (err) {
        if (mounted) {
          setConnected(false);
        }
      }
    };

    const connectWs = () => {
      const wsBase = API_URL.replace("http://", "ws://").replace("https://", "wss://");
      const ws = new WebSocket(`${wsBase}/ws/detections`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (mounted) {
            setSnapshot(data);
            setConnected(true);
          }
        } catch (err) {
          // Ignore invalid payloads and continue polling fallback.
        }
      };

      ws.onopen = () => {
        setConnected(true);
        wsPingInterval = window.setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send("ping");
          }
        }, 4000);
      };

      ws.onclose = () => {
        setConnected(false);
      };

      ws.onerror = () => {
        setConnected(false);
      };
    };

    fetchDetections();
    pollId = window.setInterval(fetchDetections, 1500);
    connectWs();

    return () => {
      mounted = false;
      if (pollId) {
        window.clearInterval(pollId);
      }
      if (wsPingInterval) {
        window.clearInterval(wsPingInterval);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 p-6 text-slate-100">
      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h1 className="text-2xl font-semibold">Raspberry Pi Vision Dashboard</h1>
            <StatusBadge
              label={connected ? "Connected" : "Disconnected"}
              color={connected ? "green" : "red"}
            />
          </div>
          <CameraStream apiUrl={API_URL} />
          <div className="mt-4 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
            <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-300">
              Model Status
            </h2>
            <p className="text-sm text-slate-200">
              {snapshot.model_note || "Waiting for backend status..."}
            </p>
          </div>
          <StatsCards stats={snapshot.stats} />
        </section>

        <aside className="lg:col-span-1">
          <DetectionPanel detections={recentDetections} humanDetected={snapshot.stats.human_detected} />
        </aside>
      </div>
    </div>
  );
}

export default App;
