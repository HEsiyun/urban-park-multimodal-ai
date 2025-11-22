import React, { useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

export default function App() {
  const [baseUrl, setBaseUrl] = useState("http://127.0.0.1:8000");
  const [activeTab, setActiveTab] = useState("agent");
  const [text, setText] = useState("");
  const [imageUri, setImageUri] = useState("");
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState("");
  const [e2eLatency, setE2eLatency] = useState(null);

  const presetCategories = [
    {
      category: "💰 Cost Analysis",
      queries: [
        "Which park had the highest total mowing labor cost in March 2025?",
        "Show mowing cost trend from January to June 2025",
        "Compare mowing costs across all parks in March 2025",
        "When was the last mowing at Cambridge Park?",
        "Which parks have the least mowing cost from June 2024 to May 2025?",
        "What is the cost of the activity in Stanley from February 2025 to March 2025?"
      ],
    },
    {
      category: "📋 Procedures & Standards",
      queries: [
        "What are the mowing steps and safety requirements?",
        "What are the dimensions for U15 soccer?",
        "Show me baseball field requirements for U13",
        "What's the pitching distance for female softball U17?",
        "What is the latest activity in Stanley?",
        "Which parks need Leaf Removal in 10 weeks?"
      ],
    },
    {
      category: "🖼️ Image Analysis",
      queries: [
        "Assess this field condition (upload image)",
        "Does this field need mowing? (upload image)",
        "Is this field suitable for soccer? (upload image)",
      ],
    },
  ];

  async function callEndpoint(path) {
    setLoading(true);
    setError("");
    setResp(null);
    setE2eLatency(null);
    
    const startTime = performance.now();
    
    try {
      const body = { text: text.trim() };
      if (imageUri) body.image_uri = imageUri;

      const r = await fetch(`${baseUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!r.ok) {
        const errorData = await r.json().catch(() => ({}));
        const detail =
          typeof errorData.detail === "string"
            ? errorData.detail
            : errorData.detail?.error || errorData.detail?.message || "";
        throw new Error(detail || `${r.status} ${r.statusText}`);
      }

      const data = await r.json();
      const endTime = performance.now();
      const latencyMs = endTime - startTime;
      
      setE2eLatency(latencyMs);
      setResp(data);
      
      console.log(`⏱️ End-to-end latency: ${latencyMs.toFixed(2)} ms (${(latencyMs/1000).toFixed(2)}s)`);
    } catch (e) {
      const endTime = performance.now();
      setE2eLatency(endTime - startTime);
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  function renderMarkdown(md) {
    if (!md) return null;

    let html = md;

    html = html.replace(/^### (.*)$/gim, '<h3 class="md-h3">$1</h3>');
    html = html.replace(/^## (.*)$/gim, '<h2 class="md-h2">$1</h2>');
    html = html.replace(/^# (.*)$/gim, '<h1 class="md-h1">$1</h1>');

    html = html.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");

    html = html.replace(/^\d+\.\s+(.*)$/gim, "<li>$1</li>");
    html = html.replace(/^- (.*)$/gim, "<li>$1</li>");

    html = html.replace(/\n\n+/g, "<br/><br/>");

    return <div className="prose" dangerouslySetInnerHTML={{ __html: html }} />;
  }

  function StatusBanner({ status, message }) {
    if (!status || status === "OK") return null;

    const map = {
      NEEDS_CLARIFICATION: {
        bg: "#fff3cd",
        bd: "#ffc107",
        emoji: "💡",
        title: "More Information Needed",
      },
      UNSUPPORTED: {
        bg: "#fde2e1",
        bd: "#f44336",
        emoji: "🚧",
        title: "Not Supported Yet",
      },
    };

    const styles = map[status] || {
      bg: "#e7f3ff",
      bd: "#2196f3",
      emoji: "ℹ️",
      title: status,
    };

    return (
      <div
        className="card"
        style={{ marginTop: 16, background: styles.bg, borderColor: styles.bd }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 8,
          }}
        >
          <span style={{ fontSize: 20 }}>{styles.emoji}</span>
          <div className="card-title" style={{ margin: 0 }}>
            {styles.title}
          </div>
        </div>
        {message && <div style={{ fontSize: 13 }}>{message}</div>}
      </div>
    );
  }

  function ClarificationsView({ clarifications }) {
    if (!clarifications || !clarifications.length) return null;
    return (
      <div
        className="card"
        style={{ marginTop: 16, background: "#fff3cd", borderColor: "#ffc107" }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 12,
          }}
        >
          <span style={{ fontSize: 24 }}>💡</span>
          <div className="card-title" style={{ margin: 0 }}>
            More Information Needed
          </div>
        </div>
        <ul className="bullets">
          {clarifications.map((c, i) => (
            <li key={i} style={{ fontSize: 14 }}>
              {c}
            </li>
          ))}
        </ul>
        <div style={{ marginTop: 12, fontSize: 13, color: "#856404" }}>
          Please provide the missing information and try again.
        </div>
      </div>
    );
  }

  function ChartsView({ charts }) {
    if (!charts || !charts.length) return null;

    return (
      <div style={{ marginTop: 16 }}>
        {charts.map((chart, idx) => (
          <div key={idx} className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">{chart.title || `Chart ${idx + 1}`}</div>
            {chart.note && (
              <div
                style={{
                  fontSize: 12,
                  color: "#666",
                  marginBottom: 8,
                  fontStyle: "italic",
                }}
              >
                {chart.note}
              </div>
            )}
            <div style={{ width: "100%", height: 400 }}>{renderChart(chart)}</div>
          </div>
        ))}
      </div>
    );
  }

  function renderChart(chart) {
    const chartType = chart?.type;
    if (chartType === "line") return renderLineChart(chart);
    if (chartType === "bar") return renderBarChart(chart);
    if (chartType === "bar_stacked") return renderStackedBarChart(chart);
    if (chartType === "timeline") return renderTimeline(chart);
    return <div className="muted">Unsupported chart type: {String(chartType)}</div>;
  }

  function renderLineChart(chart) {
    const allXValues = [
      ...new Set(chart.series.flatMap((s) => s.data.map((d) => d.x))),
    ].sort((a, b) => a - b);

    const chartData = allXValues.map((xVal) => {
      const dataPoint = { [chart.x_axis.field]: xVal };
      chart.series.forEach((series) => {
        const point = series.data.find((d) => d.x === xVal);
        dataPoint[series.name] = point ? point.y : null;
      });
      return dataPoint;
    });

    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7c7c", "#8dd1e1"];

    return (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey={chart.x_axis.field}
            label={{
              value: chart.x_axis.label,
              position: "insideBottom",
              offset: -5,
            }}
          />
          <YAxis
            label={{
              value: chart.y_axis.label,
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip />
          {chart.legend && <Legend />}
          {chart.series.map((series, idx) => (
            <Line
              key={series.name}
              type="monotone"
              dataKey={series.name}
              stroke={colors[idx % colors.length]}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }

  function renderBarChart(chart) {
    const chartData = (chart.series?.[0]?.data || []).map((d) => ({
      [chart.x_axis.field]: d.x,
      [chart.y_axis.field]: d.y,
    }));

    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey={chart.x_axis.field}
            label={{
              value: chart.x_axis.label,
              position: "insideBottom",
              offset: -5,
            }}
            angle={-15}
            textAnchor="end"
            height={80}
          />
          <YAxis
            label={{
              value: chart.y_axis.label,
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip />
          <Bar
            dataKey={chart.y_axis.field}
            fill={chart.color || "#4CAF50"}
            radius={[8, 8, 0, 0]}
          />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  function renderStackedBarChart(chart) {
    const allXValues = [
      ...new Set(chart.series.flatMap((s) => s.data.map((d) => d.x))),
    ];
    const chartData = allXValues.map((xVal) => {
      const dataPoint = { [chart.x_axis.field]: xVal };
      chart.series.forEach((series) => {
        const point = series.data.find((d) => d.x === xVal);
        dataPoint[series.name] = point ? point.y : 0;
      });
      return dataPoint;
    });

    const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7c7c", "#8dd1e1"];

    return (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey={chart.x_axis.field}
            label={{
              value: chart.x_axis.label,
              position: "insideBottom",
              offset: -5,
            }}
            angle={-15}
            textAnchor="end"
            height={80}
          />
          <YAxis
            label={{
              value: chart.y_axis.label,
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip />
          {chart.legend && <Legend />}
          {chart.series.map((series, idx) => (
            <Bar
              key={series.name}
              dataKey={series.name}
              stackId="a"
              fill={colors[idx % colors.length]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    );
  }

  function renderTimeline(chart) {
    const sortedData = [...(chart.data || [])].sort((a, b) => {
      if (chart.sort_order === "asc") return new Date(a.date) - new Date(b.date);
      return new Date(b.date) - new Date(a.date);
    });

    return (
      <div className="timeline-scroll">
        {sortedData.map((item, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              gap: 16,
              marginBottom: 24,
              paddingBottom: 24,
              borderBottom:
                idx < sortedData.length - 1 ? "1px solid #e0e0e0" : "none",
            }}
          >
            <div style={{ fontSize: 24, flexShrink: 0 }}>📅</div>
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 4 }}>
                {item.park}
              </div>
              <div style={{ color: "#666", fontSize: 14, marginBottom: 8 }}>
                {item.date
                  ? new Date(item.date).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })
                  : "—"}
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <span
                  style={{
                    display: "inline-block",
                    padding: "4px 8px",
                    background: "#e3f2fd",
                    color: "#1976d2",
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: 500,
                  }}
                >
                  {item.sessions} session{item.sessions !== 1 ? "s" : ""}
                </span>
                <span
                  style={{
                    display: "inline-block",
                    padding: "4px 8px",
                    background: "#e3f2fd",
                    color: "#1976d2",
                    borderRadius: 4,
                    fontSize: 12,
                    fontWeight: 500,
                  }}
                >
                  $
                  {typeof item.cost === "number"
                    ? item.cost.toFixed(2)
                    : "0.00"}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  function TablesView({ tables }) {
    if (!tables || !tables.length) return null;
    return (
      <div style={{ marginTop: 16 }}>
        {tables.map((t, idx) => (
          <div key={idx} className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">{t.name || `table_${idx}`}</div>
            <div className="table-wrap">
              <table className="grid-table">
                <thead>
                  <tr>
                    {(t.columns || Object.keys((t.rows && t.rows[0]) || {})).map(
                      (c) => (
                        <th key={c}>{c}</th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {(t.rows || []).map((row, rIdx) => (
                    <tr key={rIdx}>
                      {(t.columns && t.columns.length
                        ? t.columns
                        : Object.keys(row)
                      ).map((c) => (
                        <td key={c}>{String(row[c])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    );
  }

  function CitationsView({ citations }) {
    if (!citations || !citations.length) return null;
    return (
      <div className="mt">
        <div className="label">Citations</div>
        <ul className="bullets">
          {citations.map((c, i) => (
            <li key={i}>
              {c.title || "source"} —{" "}
              <span className="muted">{c.source || ""}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  function LogsView({ logs }) {
    if (!logs || !logs.length) return null;
    
    // Calculate total time and tool call time
    const totalToolTime = logs.reduce((sum, l) => sum + (l.elapsed_ms || 0), 0);
    const successCount = logs.filter(l => l.ok).length;
    const failCount = logs.filter(l => !l.ok).length;
    
    // Calculate network/overhead time
    const networkTime = e2eLatency ? e2eLatency - totalToolTime : null;
    const networkPercent = e2eLatency ? (networkTime / e2eLatency * 100) : null;
    const toolPercent = e2eLatency ? (totalToolTime / e2eLatency * 100) : null;
    
    return (
      <div className="mt">
        <div className="label">⏱️ Performance Metrics</div>
        
        {/* Summary Stats */}
        <div style={{
          display: 'flex',
          gap: 12,
          marginBottom: 16,
          flexWrap: 'wrap'
        }}>
          <div style={{
            padding: '12px 16px',
            background: '#e3f2fd',
            borderRadius: 8,
            flex: '1 1 auto',
            minWidth: 150
          }}>
            <div style={{ fontSize: 11, color: '#1565c0', fontWeight: 600, marginBottom: 4 }}>
              TOTAL TOOL TIME
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#0d47a1' }}>
              {(totalToolTime / 1000).toFixed(2)}s
            </div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
              {totalToolTime.toFixed(0)} ms
              {toolPercent && ` · ${toolPercent.toFixed(1)}% of E2E`}
            </div>
          </div>
          
          <div style={{
            padding: '12px 16px',
            background: '#f3e5f5',
            borderRadius: 8,
            flex: '1 1 auto',
            minWidth: 150
          }}>
            <div style={{ fontSize: 11, color: '#7b1fa2', fontWeight: 600, marginBottom: 4 }}>
              TOOL CALLS
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#4a148c' }}>
              {logs.length}
            </div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
              ✅ {successCount} success · ❌ {failCount} failed
            </div>
          </div>
          
          <div style={{
            padding: '12px 16px',
            background: '#fff3e0',
            borderRadius: 8,
            flex: '1 1 auto',
            minWidth: 150
          }}>
            <div style={{ fontSize: 11, color: '#e65100', fontWeight: 600, marginBottom: 4 }}>
              AVG PER TOOL
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#bf360c' }}>
              {(totalToolTime / logs.length / 1000).toFixed(2)}s
            </div>
            <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
              {(totalToolTime / logs.length).toFixed(0)} ms
            </div>
          </div>
          
          {networkTime !== null && (
            <div style={{
              padding: '12px 16px',
              background: '#e8f5e9',
              borderRadius: 8,
              flex: '1 1 auto',
              minWidth: 150
            }}>
              <div style={{ fontSize: 11, color: '#2e7d32', fontWeight: 600, marginBottom: 4 }}>
                NETWORK OVERHEAD
              </div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#1b5e20' }}>
                {(networkTime / 1000).toFixed(2)}s
              </div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                {networkTime.toFixed(0)} ms
                {networkPercent && ` · ${networkPercent.toFixed(1)}% of E2E`}
              </div>
            </div>
          )}
        </div>
        
        {/* Time Breakdown Bar */}
        {e2eLatency && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b', marginBottom: 6 }}>
              TIME BREAKDOWN
            </div>
            <div style={{ 
              display: 'flex', 
              height: 32, 
              borderRadius: 6, 
              overflow: 'hidden',
              border: '1px solid #e5e7eb'
            }}>
              <div style={{
                width: `${toolPercent}%`,
                background: 'linear-gradient(90deg, #3b82f6, #2563eb)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: 11,
                fontWeight: 600
              }}>
                {toolPercent > 15 && `Tools: ${toolPercent.toFixed(1)}%`}
              </div>
              <div style={{
                width: `${networkPercent}%`,
                background: 'linear-gradient(90deg, #10b981, #059669)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'white',
                fontSize: 11,
                fontWeight: 600
              }}>
                {networkPercent > 15 && `Network: ${networkPercent.toFixed(1)}%`}
              </div>
            </div>
          </div>
        )}
        
        {/* Detailed Logs */}
        <div className="label" style={{ marginTop: 16, marginBottom: 8 }}>Tool Execution Details</div>
        <div className="logs">
          {logs.map((l, i) => (
            <div key={i} className="log-row">
              <span className={`pill ${l.ok ? "ok" : "err"}`}>
                {l.ok ? "ok" : "err"}
              </span>
              <span className="mono">{l.tool}</span>
              <span style={{ 
                fontWeight: 600,
                color: l.elapsed_ms > 2000 ? '#ef4444' : l.elapsed_ms > 1000 ? '#f59e0b' : '#10b981'
              }}>
                {(l.elapsed_ms / 1000).toFixed(2)}s
              </span>
              <span className="muted" style={{ fontSize: 11 }}>
                ({l.elapsed_ms} ms)
              </span>
              <span className="muted">
                args:{" "}
                {Array.isArray(l.args_redacted)
                  ? l.args_redacted.join(", ")
                  : "-"}
              </span>
              {l.err && <span className="err-text">{l.err}</span>}
            </div>
          ))}
        </div>
      </div>
    );
  }

  function DebugView({ debug }) {
    if (!debug) return null;
    return (
      <div className="mt">
        <details>
          <summary
            style={{ cursor: "pointer", fontWeight: 600, marginBottom: 8 }}
          >
            🛠 Debug Information
          </summary>
        <div className="card" style={{ background: "#f8f9fa" }}>
            <pre className="json">{JSON.stringify(debug, null, 2)}</pre>
          </div>
        </details>
      </div>
    );
  }

  return (
    <div className="page">
      <div className="shell">
        <header className="header">
          <h1>Parks Maintenance Intelligence System</h1>
          <div className="row">
            <span className="muted small">API Endpoint</span>
            <input
              className="input"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:8000"
            />
          </div>
        </header>

        <div className="grid">
          <div className="col-main">
            <div className="card">
              {presetCategories.map((cat, catIdx) => (
                <div key={catIdx} style={{ marginBottom: 16 }}>
                  <div
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      marginBottom: 8,
                      color: "#64748b",
                    }}
                  >
                    {cat.category}
                  </div>
                  <div className="row wrap gap">
                    {cat.queries.map((query, qIdx) => (
                      <button
                        key={qIdx}
                        className="btn ghost"
                        onClick={() => {
                          setText(query.replace(" (upload image)", ""));
                          if (query.includes("upload image") && !imageUri) {
                            alert(
                              "💡 Heads-up: image analysis works best if you upload an image."
                            );
                          }
                        }}
                        style={{ fontSize: 12 }}
                      >
                        {query.length > 40
                          ? query.substring(0, 37) + "..."
                          : query}
                      </button>
                    ))}
                  </div>
                </div>
              ))}

              <textarea
                className="textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Type your question or select a preset above..."
              />

              <div className="row gap">
                <label className="btn file" style={{ marginRight: 8 }}>
                  <input
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (!f) {
                        setImageUri("");
                        return;
                      }
                      
                      try {
                        console.log("📸 Original file size:", (f.size / 1024 / 1024).toFixed(2), "MB");
                        
                        // ✅ Compress image before converting to base64
                        const img = new Image();
                        const reader = new FileReader();
                        
                        reader.onload = (e) => {
                          img.src = e.target.result;
                        };
                        
                        img.onload = () => {
                          // Calculate new dimensions (max 1920px width/height)
                          const MAX_SIZE = 1920;
                          let width = img.width;
                          let height = img.height;
                          
                          if (width > height) {
                            if (width > MAX_SIZE) {
                              height = (height * MAX_SIZE) / width;
                              width = MAX_SIZE;
                            }
                          } else {
                            if (height > MAX_SIZE) {
                              width = (width * MAX_SIZE) / height;
                              height = MAX_SIZE;
                            }
                          }
                          
                          // Create canvas and compress
                          const canvas = document.createElement('canvas');
                          canvas.width = width;
                          canvas.height = height;
                          const ctx = canvas.getContext('2d');
                          ctx.drawImage(img, 0, 0, width, height);
                          
                          // Convert to base64 with quality setting
                          const base64String = canvas.toDataURL('image/jpeg', 0.85);
                          const sizeInMB = (base64String.length * 0.75 / 1024 / 1024).toFixed(2);
                          
                          console.log("✅ Compressed to:", sizeInMB, "MB");
                          console.log("📐 Dimensions:", width, "x", height);
                          
                          if (base64String.length * 0.75 > 5 * 1024 * 1024) {
                            alert("⚠️ Image still too large after compression. Try a smaller image.");
                            return;
                          }
                          
                          setImageUri(base64String);
                        };
                        
                        reader.readAsDataURL(f);
                      } catch (error) {
                        console.error("❌ Error processing image:", error);
                        alert("Failed to process image: " + error.message);
                      }
                    }}
                  />
                  📷 Upload Image
                </label>
                {imageUri && (
                  <>
                    <img
                      src={imageUri}
                      alt="preview"
                      className="thumb"
                      style={{ maxHeight: 60 }}
                    />
                    <button
                      className="btn ghost"
                      onClick={() => setImageUri("")}
                      style={{ padding: "4px 8px", fontSize: 12 }}
                    >
                      ✕
                    </button>
                  </>
                )}

                <div className="spacer" />

                <div className="tabs">
                  <button
                    className={`tab ${activeTab === "agent" ? "active" : ""}`}
                    onClick={() => setActiveTab("agent")}
                  >
                    Agent Answer
                  </button>
                  <button
                    className={`tab ${activeTab === "nlu" ? "active" : ""}`}
                    onClick={() => setActiveTab("nlu")}
                  >
                    NLU Parse
                  </button>
                </div>

                <button
                  className="btn primary"
                  disabled={loading}
                  onClick={() =>
                    callEndpoint(
                      activeTab === "agent" ? "/agent/answer" : "/nlu/parse"
                    )
                  }
                >
                  {loading ? "⏳ Processing..." : "🚀 Send"}
                </button>
              </div>

              {error && <div className="error">❌ {error}</div>}
            </div>
          </div>

          <aside className="col-side">
            <div className="card">
              <div className="label">✨ System Capabilities</div>

              <div style={{ marginBottom: 16 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#64748b",
                    marginBottom: 6,
                  }}
                >
                  💰 Cost Analysis
                </div>
                <ul className="bullets" style={{ fontSize: 13 }}>
                  <li>Highest cost by park/month</li>
                  <li>Cost trends over time</li>
                  <li>Park comparisons</li>
                  <li>Last activity tracking</li>
                </ul>
              </div>

              <div style={{ marginBottom: 16 }}>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#64748b",
                    marginBottom: 6,
                  }}
                >
                  📋 Standards & Procedures
                </div>
                <ul className="bullets" style={{ fontSize: 13 }}>
                  <li>Mowing SOPs and safety</li>
                  <li>Field dimensions (all sports)</li>
                  <li>Age group requirements</li>
                  <li>Equipment specifications</li>
                </ul>
              </div>

              <div>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#64748b",
                    marginBottom: 6,
                  }}
                >
                  🖼️ Image Analysis (VLM)
                </div>
                <ul className="bullets" style={{ fontSize: 13 }}>
                  <li>Field condition assessment</li>
                  <li>Maintenance needs detection</li>
                  <li>Turf health evaluation</li>
                  <li>AI-powered recommendations</li>
                </ul>
              </div>

              <div
                style={{
                  marginTop: 16,
                  padding: 12,
                  background: "#f8f9fc",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>💡 Tips</div>
                <div style={{ color: "#64748b", lineHeight: 1.5 }}>
                  • Upload images for visual analysis
                  <br />
                  • Ask about any sport or age group
                  <br />
                  • Combine data queries with standards
                </div>
              </div>
            </div>
          </aside>
        </div>

        <section className="card response-card">
          <div className="label">Response</div>
          
          {/* E2E Latency Banner */}
          {e2eLatency && (
            <div style={{
              padding: '12px 16px',
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              borderRadius: 8,
              marginBottom: 16,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              color: 'white',
              boxShadow: '0 2px 8px rgba(102, 126, 234, 0.3)'
            }}>
              <div>
                <div style={{ fontSize: 11, fontWeight: 600, opacity: 0.9, marginBottom: 4 }}>
                  🚀 END-TO-END LATENCY
                </div>
                <div style={{ fontSize: 24, fontWeight: 700 }}>
                  {(e2eLatency / 1000).toFixed(2)}s
                </div>
              </div>
              <div style={{ textAlign: 'right', fontSize: 12, opacity: 0.9 }}>
                <div>{e2eLatency.toFixed(0)} ms</div>
                <div style={{ fontSize: 10, marginTop: 2 }}>
                  Frontend → Backend → Frontend
                </div>
              </div>
            </div>
          )}
          
          {!resp && (
            <div className="muted">No response yet. Try a query above!</div>
          )}
          {resp && (
            <div className="stack response-content">
              {activeTab === "agent" && (
                <StatusBanner status={resp.status} message={resp.message} />
              )}

              {activeTab === "agent" &&
                resp.clarifications &&
                resp.clarifications.length > 0 && (
                  <ClarificationsView clarifications={resp.clarifications} />
                )}

              {resp.answer_md && <div>{renderMarkdown(resp.answer_md)}</div>}
              {activeTab === "agent" && <ChartsView charts={resp.charts} />}
              {activeTab === "agent" && <TablesView tables={resp.tables} />}
              {activeTab === "agent" && (
                <CitationsView citations={resp.citations} />
              )}
              {activeTab === "agent" && <LogsView logs={resp.logs} />}
              {activeTab === "agent" && <DebugView debug={resp.debug} />}

              {activeTab === "nlu" && (
                <div className="stack">
                  <div className="card">
                    <div className="label">Intent</div>
                    <pre className="json">
                      {JSON.stringify(resp.intent, null, 2)}
                    </pre>
                  </div>
                  <div className="card">
                    <div className="label">Confidence</div>
                    <pre className="json">
                      {JSON.stringify(resp.confidence, null, 2)}
                    </pre>
                  </div>
                  <div className="card">
                    <div className="label">Slots</div>
                    <pre className="json">
                      {JSON.stringify(resp.slots, null, 2)}
                    </pre>
                  </div>
                  <div className="card">
                    <div className="label">Raw Query</div>
                    <pre className="json">
                      {JSON.stringify(resp.raw_query, null, 2)}
                    </pre>
                  </div>
                  <div className="card">
                    <div className="label">Full Response</div>
                    <pre className="json">{JSON.stringify(resp, null, 2)}</pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}