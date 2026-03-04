// ─── LearningCurves.jsx ──────────────────────────────────────────────────────
// Collapsible learning curve chart for race-based and runner-based models.
// Props:
//   info      — trainingInfo object from model_full.json
//   variant   — 'race' | 'runner'
// ─────────────────────────────────────────────────────────────────────────────

import React, { useState, useMemo } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, Legend, ResponsiveContainer,
} from 'recharts';

// ── Metric configs per model variant ────────────────────────────────────────

const RACE_METRICS = [
    { key: 'loss',     label: 'loss (train)',     color: '#e74c3c', dash: '4 2' },
    { key: 'val_loss', label: 'val loss',         color: '#ff8c69', dash: '' },
    { key: 'val_logloss_race', label: 'val logloss (per-race)', color: '#ffb199', dash: '2 2' },
    { key: 'val_acc',  label: 'val acc',          color: '#4a90d9', dash: '' },
    { key: 'val_r3',   label: 'val recall@3',     color: '#2ecc71', dash: '' },
    { key: 'val_p3',   label: 'val prec@3',       color: '#a8e063', dash: '3 2' },
    { key: 'val_ndcg3',label: 'val NDCG@3',       color: '#ffd166', dash: '' },
    { key: 'val_hit1', label: 'val Hit@1',        color: '#06d6a0', dash: '5 3' },
    { key: 'val_ap_macro', label: 'val AP (macro per-race)', color: '#a78bfa', dash: '' },
    { key: 'val_auc',  label: 'val AUC (ROC)',    color: '#f0a500', dash: '' },
];

const RUNNER_METRICS = [
    { key: 'loss',     label: 'loss (train)', color: '#e74c3c', dash: '4 2' },
    { key: 'val_loss', label: 'val loss',     color: '#ff8c69', dash: '' },
    { key: 'val_logloss_race', label: 'val logloss (per-race)', color: '#ffb199', dash: '2 2' },
    { key: 'val_acc',  label: 'val acc',      color: '#4a90d9', dash: '' },
    { key: 'val_auc',  label: 'val AUC',      color: '#f0a500', dash: '' },
    { key: 'val_ap',   label: 'val AP',       color: '#a78bfa', dash: '' },
    { key: 'val_ndcg3',label: 'val NDCG@3',   color: '#ffd166', dash: '' },
    { key: 'val_hit1', label: 'val Hit@1',    color: '#06d6a0', dash: '5 3' },
    { key: 'val_p05',  label: 'val prec@0.5', color: '#2ecc71', dash: '3 2' },
];

// ── Custom tooltip ────────────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label, metrics }) {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: '#0d0f14',
            border: '1px solid #1e2330',
            borderRadius: 4,
            padding: '10px 14px',
            fontFamily: "'IBM Plex Mono','Courier New',monospace",
            fontSize: 11,
            minWidth: 160,
        }}>
            <div style={{ color: '#4a90d9', marginBottom: 6, letterSpacing: 1 }}>
                EPOCH {label}
            </div>
            {payload.map(p => (
                <div key={p.dataKey} style={{ color: p.color, marginBottom: 2, display: 'flex', justifyContent: 'space-between', gap: 12 }}>
                    <span style={{ color: '#666' }}>{metrics.find(m => m.key === p.dataKey)?.label ?? p.dataKey}</span>
                    <span>{typeof p.value === 'number' ? p.value.toFixed(4) : '—'}</span>
                </div>
            ))}
        </div>
    );
}

// ── Custom legend ─────────────────────────────────────────────────────────────

function CustomLegend({ metrics, hidden, onToggle }) {
    return (
        <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '6px 14px',
            marginTop: 10,
            paddingTop: 10,
            borderTop: '1px solid #1a1d26',
        }}>
            {metrics.map(m => {
                const isHidden = hidden.has(m.key);
                return (
                    <button
                        key={m.key}
                        onClick={() => onToggle(m.key)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 6,
                            background: 'none',
                            border: 'none',
                            cursor: 'pointer',
                            padding: '2px 4px',
                            borderRadius: 3,
                            opacity: isHidden ? 0.3 : 1,
                            transition: 'opacity 0.15s',
                            fontFamily: "'IBM Plex Mono','Courier New',monospace",
                            fontSize: 10,
                            color: isHidden ? '#555' : '#aaa',
                            letterSpacing: 0.3,
                        }}
                    >
                        <span style={{
                            display: 'inline-block',
                            width: 20,
                            height: 2,
                            background: isHidden ? '#333' : m.color,
                            borderRadius: 1,
                            boxShadow: isHidden ? 'none' : `0 0 4px ${m.color}55`,
                            ...(m.dash ? {
                                background: 'none',
                                borderTop: `2px dashed ${isHidden ? '#333' : m.color}`,
                            } : {}),
                        }} />
                        {m.label}
                    </button>
                );
            })}
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function LearningCurves({ info, variant }) {
    const [open, setOpen] = useState(false);
    const [hidden, setHidden] = useState(new Set());

    const metrics = variant === 'runner' ? RUNNER_METRICS : RACE_METRICS;

    const history = useMemo(() => {
        if (!info?.history?.length) return [];
        return info.history;
    }, [info]);

    // Per-metric Y domains (min/max with a bit of padding)
    const domains = useMemo(() => {
        if (!history.length) return {};
        const out = {};
        for (const m of metrics) {
            const vals = history.map(e => e[m.key]).filter(v => v != null && isFinite(v));
            if (!vals.length) continue;
            const lo = Math.min(...vals);
            const hi = Math.max(...vals);
            const pad = (hi - lo) * 0.08 || 0.01;
            out[m.key] = [lo - pad, hi + pad];
        }
        return out;
    }, [history, metrics]);

    // Single shared Y domain across all visible metrics
    const yDomain = useMemo(() => {
        const allMins = [], allMaxs = [];
        for (const m of metrics) {
            if (hidden.has(m.key)) continue;
            const d = domains[m.key];
            if (d) { allMins.push(d[0]); allMaxs.push(d[1]); }
        }
        if (!allMins.length) return ['auto', 'auto'];
        return [Math.min(...allMins), Math.max(...allMaxs)];
    }, [domains, metrics, hidden]);

    const toggleMetric = (key) => {
        setHidden(prev => {
            const next = new Set(prev);
            next.has(key) ? next.delete(key) : next.add(key);
            return next;
        });
    };

    const hasData = history.length > 0;

    return (
        <div style={{ marginBottom: 20 }}>
            {/* Toggle button */}
            <button
                onClick={() => setOpen(v => !v)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    background: 'none',
                    border: 'none',
                    cursor: hasData ? 'pointer' : 'default',
                    padding: 0,
                    fontFamily: "'IBM Plex Mono','Courier New',monospace",
                    fontSize: 11,
                    letterSpacing: 2,
                    color: hasData ? '#4a90d9' : '#333',
                    textTransform: 'uppercase',
                    opacity: hasData ? 1 : 0.4,
                }}
                disabled={!hasData}
            >
                <span style={{
                    display: 'inline-block',
                    width: 12,
                    height: 12,
                    border: '1px solid currentColor',
                    borderRadius: 2,
                    position: 'relative',
                    flexShrink: 0,
                }}>
                    <span style={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: `translate(-50%, -50%) rotate(${open ? 45 : 0}deg)`,
                        transition: 'transform 0.2s',
                        fontSize: 10,
                        lineHeight: 1,
                        color: 'inherit',
                    }}>+</span>
                </span>
                Training and Validation Learning Curves
                {hasData && (
                    <span style={{ color: '#2a3a4a', fontSize: 10 }}>
                        ({history.length} epoch{history.length !== 1 ? 's' : ''})
                    </span>
                )}
            </button>

            {/* Chart panel */}
            {open && hasData && (
                <div style={{
                    marginTop: 12,
                    background: '#080b10',
                    border: '1px solid #1a1d26',
                    borderRadius: 6,
                    padding: '16px 12px 12px',
                }}>
                    <ResponsiveContainer width="100%" height={260}>
                        <LineChart
                            data={history}
                            margin={{ top: 4, right: 16, left: -10, bottom: 0 }}
                        >
                            <CartesianGrid
                                strokeDasharray="2 4"
                                stroke="#151820"
                                vertical={false}
                            />
                            <XAxis
                                dataKey="epoch"
                                tick={{ fontSize: 10, fill: '#3a4a5a', fontFamily: 'IBM Plex Mono, monospace' }}
                                tickLine={false}
                                axisLine={{ stroke: '#1a1d26' }}
                                label={{ value: 'epoch', position: 'insideBottomRight', offset: -4, fontSize: 9, fill: '#2a3a4a', fontFamily: 'IBM Plex Mono, monospace' }}
                            />
                            <YAxis
                                domain={yDomain}
                                tick={{ fontSize: 10, fill: '#3a4a5a', fontFamily: 'IBM Plex Mono, monospace' }}
                                tickLine={false}
                                axisLine={{ stroke: '#1a1d26' }}
                                tickFormatter={v => v.toFixed(2)}
                                width={44}
                            />
                            <Tooltip
                                content={<CustomTooltip metrics={metrics} />}
                                cursor={{ stroke: '#1e2330', strokeWidth: 1 }}
                            />
                            {metrics.map(m => (
                                <Line
                                    key={m.key}
                                    dataKey={m.key}
                                    stroke={m.color}
                                    strokeWidth={hidden.has(m.key) ? 0 : 1.5}
                                    strokeDasharray={m.dash || ''}
                                    dot={{ r: 2.5, fill: m.color, strokeWidth: 0 }}
                                    activeDot={{ r: 4, fill: m.color, strokeWidth: 0 }}
                                    hide={hidden.has(m.key)}
                                    isAnimationActive={false}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                    <CustomLegend metrics={metrics} hidden={hidden} onToggle={toggleMetric} />
                </div>
            )}
        </div>
    );
}
