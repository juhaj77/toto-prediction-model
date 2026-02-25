import React, { useEffect, useState } from 'react';

// ─── COLUMNS ──────────────────────────────────────────────────────────────────
// Static (per runner) and historical (per prior start) column definitions.
// Päivä / Current_Start_Date removed — it is always today and adds no value.

const STATIC_COLS = [
    { key: 'number',           label: 'No',          w: 42  },
    { key: 'name',             label: 'Horse',        w: 175 },
    { key: 'coach',            label: 'Coach',        w: 150 },
    { key: 'driver',           label: 'Driver',       w: 150 },
    { key: 'age',              label: 'Age',          w: 36  },
    { key: 'gender',           label: 'Sex',          w: 30  },
    { key: 'isColdBlood',      label: 'CB',           w: 30  },
    { key: 'frontShoes',       label: 'F.Shoe',       w: 68  },
    { key: 'rearShoes',        label: 'R.Shoe',       w: 68  },
    { key: 'frontShoeChg',     label: 'F↑',           w: 40  },
    { key: 'rearShoeChg',      label: 'R↑',           w: 46  },
    { key: 'specialCart',      label: 'Cart',         w: 46  },
    { key: 'distance',         label: 'Dist',         w: 56  },
    { key: 'isCarStart',       label: 'Car',          w: 40  },
    { key: 'bettingPct',       label: 'Bet%',         w: 60  },
    { key: 'winPct',           label: 'Win%',         w: 66  },
    { key: 'record',           label: 'Record',       w: 70  },
    { key: 'isAutoRecord',     label: 'AutoRec',      w: 60  },
    { key: 'prevIndexNorm',    label: 'PodiumIdx',    w: 72  },
];

const HIST_COLS = [
    { key: 'histDate',         label: 'Date',         w: 86  },
    { key: 'histDriver',        label: 'H.Driver',     w: 140 },
    { key: 'track',            label: 'Track',        w: 46  },
    { key: 'histDistance',     label: 'Dist',         w: 56  },
    { key: 'startPos',         label: 'StartPos',     w: 60  },
    { key: 'kmTime',           label: 'km/min',       w: 72  },
    { key: 'histFrontShoes',   label: 'F.Shoe',       w: 58  },
    { key: 'histRearShoes',    label: 'R.Shoe',       w: 58  },
    { key: 'histSpecialCart',  label: 'Cart',         w: 46  },
    { key: 'histIsCarStart',   label: 'Car',          w: 40  },
    { key: 'break',            label: 'Break',        w: 50  },
    { key: 'disqualified',     label: 'Disq',         w: 40  },
    { key: 'DNF',              label: 'DNF',          w: 46  },
    { key: 'trackCondition',   label: 'Track cond.',  w: 100 },
    { key: 'position',         label: 'Pos',          w: 40  },
    { key: 'odd',              label: 'Odd',          w: 66  },
    { key: 'prize',            label: 'Prize €',      w: 86  },
];

// ─── FORMATTERS ───────────────────────────────────────────────────────────────

const fmtShoes = v => {
    const s = String(v || '').toUpperCase();
    if (s === 'HAS_SHOES') return '✓';
    if (s === 'NO_SHOES')  return '—';
    return '?';
};

const fmtCart = v => {
    const s = String(v || '').toUpperCase();
    if (s === 'YES') return '✓';
    if (s === 'NO')  return '—';
    return s || '?';
};

const fmtPosition = n => {
    if (n === 20) return 'Disq';
    if (n === 21) return 'DNF';
    if (!n || n === 10) return '—';
    return String(n);
};

// ─── PARSERS — must match scraper.js and ravimalli.js exactly ─────────────────

function parseRecord(runner) {
    for (const key of ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord']) {
        const val = runner[key];
        if (!val) continue;
        const n = parseFloat(String(val).replace(',', '.'));
        if (!isNaN(n) && n > 0)
            return { record: n, isAutoRecord: key === 'mobileStartRecord' };
    }
    return { record: null, isAutoRecord: false };
}

function parseBettingPct(runner) {
    return parseFloat(runner.betPercentages?.KAK?.percentage ?? 0) / 100;
}

function parseWinPct(runner) {
    const total = runner.stats?.total;
    if (!total) return 0;
    if (total.winningPercent != null) return parseFloat(total.winningPercent);
    if (total.starts > 0) return Math.round((total.position1 / total.starts) * 10000) / 100;
    return 0;
}

function encodeGender(g) {
    if (g === 'TAMMA') return 'M';   // Mare
    if (g === 'ORI')   return 'S';   // Stallion
    return 'G';                       // Gelding
}

// Parse raw km-time string: "15,5a" → { kmNum: 15.5, isCarStart: true, isBreak: false }
function parseKmTime(raw) {
    if (!raw) return { kmNum: 0, isCarStart: false, isBreak: false, display: '' };
    const s          = String(raw);
    const isCarStart = s.includes('a');
    const isBreak    = s.includes('x');
    const numStr     = s.match(/[\d,]+/)?.[0] || '';
    const kmNum      = parseFloat(numStr.replace(',', '.')) || 0;
    const suffix     = isBreak ? 'x' : isCarStart ? 'a' : '';
    return { kmNum, isCarStart, isBreak, display: kmNum > 0 ? kmNum.toFixed(2) + suffix : '' };
}

// Parse finishing position: "4" → 4, "hyl" → 20, "k" → 21
function parsePosition(result) {
    if (!result) return { position: null, disqualified: false, DNF: false };
    const s            = String(result).toLowerCase();
    const disqualified = /[hdp]/.test(s);
    const DNF          = s.includes('k');
    const m            = s.match(/^\d+/);
    let position       = null;
    if (m)             position = parseInt(m[0]);
    else if (disqualified) position = 20;
    else if (DNF)      position = 21;
    return { position, disqualified, DNF };
}

// ─── ROW BUILDER ──────────────────────────────────────────────────────────────
// Builds a flat row array from the Veikkaus API runner objects.
// One row per prior start; runners with no history get one placeholder row.

function buildRows(runnersArr, raceInfo) {
    const rows       = [];
    const raceDistStr = raceInfo?.distance   || '';
    const isCarStart  = raceInfo?.startType === 'CAR_START';

    for (const r of runnersArr) {
        const { record, isAutoRecord } = parseRecord(r);

        // Weighted podium index normalised by ALL starts (disq/DNF count as 0 pts).
        // Matches the prevIndexNorm feature in ravimalli.js / korrelaatioanalyysi_v2.m.
        let prevIndexNorm = 0;
        const allPrev = r.prevStarts || [];
        if (allPrev.length > 0) {
            let score = 0, count = 0;
            for (const ps of allPrev) {
                const { position } = parsePosition(ps.result);
                if (position === null) continue;
                count++;                          // disq/DNF → 0 pts, still counted
                if      (position === 1) score += 1.00;
                else if (position === 2) score += 0.50;
                else if (position === 3) score += 0.33;
            }
            prevIndexNorm = count > 0 ? score / count : 0;
        }

        // Static fields shared across all rows for this runner
        const base = {
            number:        r.startNumber              || '',
            name:          r.horseName                || '',
            coach:         r.coachName                || '',
            driver:        r.driverName               || '',
            age:           r.horseAge                 || '',
            gender:        encodeGender(r.gender),
            isColdBlood:   (r.breed === 'K' || r.breed === 'FINNHORSE') ? '✓' : '',
            frontShoes:    fmtShoes(r.frontShoes),
            rearShoes:     fmtShoes(r.rearShoes),
            frontShoeChg:  r.frontShoesChanged ? '↑' : '',
            rearShoeChg:   r.rearShoesChanged  ? '↑' : '',
            specialCart:   fmtCart(r.specialCart),
            distance:      raceDistStr,
            isCarStart:    isCarStart ? '✓' : '',
            bettingPct:    parseBettingPct(r) > 0 ? parseBettingPct(r).toFixed(2) + '%' : '—',
            winPct:        parseWinPct(r)     > 0 ? parseWinPct(r).toFixed(1)     + '%' : '—',
            record:        record             != null ? record.toFixed(2)              : '—',
            isAutoRecord:  isAutoRecord ? '✓' : '',
            prevIndexNorm: prevIndexNorm > 0 ? prevIndexNorm.toFixed(3) : 0,
        };

        if (allPrev.length === 0) {
            rows.push({
                ...base,
                // History fields empty
                histDate: '', histDriver: '', track: '',
                histDistance: '', startPos: '', kmTime: '',
                histFrontShoes: '', histRearShoes: '', histSpecialCart: '', histIsCarStart: '',
                break: '', disqualified: '', DNF: '',
                trackCondition: '', position: '', odd: '', prize: '',
                // Internal sort/colour helpers
                _position: null, _isBreak: false, _isDisq: false, _palkinto: 0,
            });
        } else {
            for (const ps of allPrev) {
                const { kmNum, isCarStart: psIsCarStart, isBreak, display } = parseKmTime(ps.kmTime);
                const { position, disqualified, DNF }                        = parsePosition(ps.result);

                // winOdd is ×10 in Veikkaus API (e.g. 152 = 15.2)
                const odd    = parseFloat(ps.winOdd    || 0) / 10;
                // firstPrize is in cents×10 (e.g. 1 000 000 = 100 €)
                const prize  = parseFloat(ps.firstPrize || 0) / 10000;

                rows.push({
                    // Static fields
                    ...base,
                    // History fields (override keys that exist in both)
                    histDate:     ps.shortMeetDate || '',
                    histDriver:    ps.driverFullName || ps.driver || '',
                    track:         ps.trackCode     || '',
                    histDistance: ps.distance      || '',
                    startPos:      ps.startTrack    || '',
                    kmTime:        display,
                    histFrontShoes: fmtShoes(ps.frontShoes),
                    histRearShoes:  fmtShoes(ps.rearShoes),
                    histSpecialCart: fmtCart(ps.specialCart),
                    histIsCarStart: psIsCarStart ? '✓' : '',
                    break:         isBreak      ? '✗' : '',
                    disqualified:  disqualified ? '✗' : '',
                    DNF:           DNF          ? '✗' : '',
                    trackCondition: ps.trackCondition || '',
                    position:      fmtPosition(position),
                    odd:           odd   > 0 ? odd.toFixed(1)                          : '—',
                    prize:         prize > 0 ? Math.round(prize).toLocaleString('fi-FI') : '—',
                    // Internal helpers
                    _position:     position,
                    _isBreak:      isBreak,
                    _isDisq:       disqualified,
                    _palkinto:     prize,
                });
            }
        }
    }
    return rows;
}

// ─── CELL COLOUR ──────────────────────────────────────────────────────────────

function cellColor(colKey, row) {
    if (colKey === 'name')   return '#e8eaf0';
    if (colKey === 'number') return '#c8a040';
    if (colKey === 'position') {
        if (row._position === 1)               return '#f0a500';
        if (row._position === 2 || row._position === 3) return '#4a90d9';
        if (row._position >= 20)               return '#e74c3c';
    }
    if (['break', 'disqualified', 'DNF'].includes(colKey) && row[colKey] === '✗')
        return '#e74c3c';
    if (colKey === 'kmTime' && (row._isBreak || row._isDisq))
        return '#e74c3c';
    if (colKey === 'prize' && row._palkinto > 0)
        return '#2ecc71';
    if (HIST_COLS.find(c => c.key === colKey))
        return '#6a7a8a';
    return '#7a8a9a';
}

// ─── COMPONENT ────────────────────────────────────────────────────────────────

export default function RunnerModal({ raceId, race, raceLabel, preloadedData, onClose }) {
    const [rows,    setRows]    = useState([]);
    const [loading, setLoading] = useState(true);
    const [error,   setError]   = useState('');

    // Close on Escape
    useEffect(() => {
        const handler = e => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [onClose]);

    useEffect(() => {
        if (!raceId) return;

        // Reuse data already fetched by App.jsx for the prediction — avoids a duplicate request
        if (preloadedData?.raw) {
            try {
                const raw = preloadedData.raw;
                const arr = Array.isArray(raw)
                    ? raw
                    : (raw.collection || raw.runners || raw.data || Object.values(raw));
                const starters = arr.filter(r => r.scratched !== true);
                console.log('[modal] preloaded', starters.length, 'runners');
                setRows(buildRows(starters, preloadedData.race || race));
            } catch (e) { setError(e.message); }
            finally     { setLoading(false);   }
            return;
        }

        // Fallback: fetch independently (modal opened before running prediction)
        setLoading(true); setError('');
        fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`)
            .then(r => r.json())
            .then(raw => {
                const arr = Array.isArray(raw)
                    ? raw
                    : (raw.collection || raw.runners || raw.data || Object.values(raw));
                const starters = arr.filter(r => r.scratched !== true);
                console.log('[modal] fetched', starters.length, 'runners');
                setRows(buildRows(starters, race));
            })
            .catch(e  => setError(e.message))
            .finally(  () => setLoading(false));
    }, [raceId, preloadedData]);

    // Assign alternating background per horse (zebra striping across history rows)
    const horseOrder = [];
    const seen = {};
    rows.forEach(r => { if (!seen[r.name]) { seen[r.name] = true; horseOrder.push(r.name); } });
    const horseIndex = Object.fromEntries(horseOrder.map((n, i) => [n, i]));

    const bgStatic = ['#0d1018', '#0f1320'];
    const bgHist   = ['#08090f', '#090c15'];

    const thStyle = isHist => ({
        position: 'sticky', top: 0, zIndex: 2,
        background: isHist ? '#06070c' : '#07090e',
        color: '#3a6090', fontWeight: 700, fontSize: 10.5,
        letterSpacing: 1.2, textTransform: 'uppercase',
        padding: '6px 9px', whiteSpace: 'nowrap',
        borderBottom: '2px solid #151c28',
        fontFamily: "'IBM Plex Mono','Courier New',monospace",
    });

    // Resolve display value.
    // History columns use _hist-prefixed keys to avoid collision with static
    // columns that share the same name (driver, distance, frontShoes, etc.).
    const resolveCell = (colKey, row) => String(row[colKey] ?? '');

    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 1000,
            background: 'rgba(0,0,0,0.94)',
            display: 'flex', flexDirection: 'column',
            fontFamily: "'IBM Plex Mono','Courier New',monospace",
        }}>
            {/* Top bar */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0,
                padding: '8px 14px', background: '#07090e',
                borderBottom: '1px solid #151c28',
            }}>
                <span style={{ fontSize: 10, letterSpacing: 3, color: '#3a6090', textTransform: 'uppercase' }}>
                    Race details
                </span>
                <span style={{ fontSize: 13, fontWeight: 700, color: '#e8eaf0' }}>{raceLabel}</span>
                <span style={{ marginLeft: 'auto', fontSize: 10, color: '#334' }}>ESC to close</span>
                <button onClick={onClose} style={{
                    background: 'none', border: '1px solid #1e2c40', color: '#ccd',
                    borderRadius: 4, padding: '2px 11px', fontFamily: 'inherit',
                    fontSize: 15, cursor: 'pointer', marginLeft: 8,
                }}>✕</button>
            </div>

            {/* Table */}
            <div style={{ flex: 1, overflow: 'auto' }}>
                {loading && <div style={{ padding: 40, color: '#4a90d9', fontSize: 13 }}>Loading…</div>}
                {error   && <div style={{ padding: 40, color: '#e74c3c', fontSize: 13 }}>✗ {error}</div>}
                {!loading && !error && rows.length === 0 && (
                    <div style={{ padding: 40, color: '#664', fontSize: 13 }}>No data</div>
                )}

                {!loading && !error && rows.length > 0 && (
                    <table style={{ borderCollapse: 'collapse', minWidth: '100%' }}>
                        <thead>
                        <tr>
                            {STATIC_COLS.map(c => (
                                <th key={c.key} style={{ ...thStyle(false), minWidth: c.w }}>{c.label}</th>
                            ))}
                            <th style={{ ...thStyle(true), color: '#111', padding: '6px 2px', minWidth: 6 }}>│</th>
                            {HIST_COLS.map(c => (
                                <th key={c.key} style={{ ...thStyle(true), minWidth: c.w }}>{c.label}</th>
                            ))}
                        </tr>
                        </thead>
                        <tbody>
                        {rows.map((row, i) => {
                            const hi       = horseIndex[row.name] ?? 0;
                            const newHorse = i === 0 || rows[i - 1]?.name !== row.name;
                            const tdS = {
                                padding: '4px 9px', whiteSpace: 'nowrap', fontSize: 11.5,
                                fontFamily: "'IBM Plex Mono','Courier New',monospace",
                                borderBottom: '1px solid #0a0c12',
                                background: bgStatic[hi % 2],
                            };
                            const tdH = { ...tdS, background: bgHist[hi % 2] };

                            return (
                                <tr key={i} style={{ borderTop: newHorse ? '2px solid #141c28' : 'none' }}>
                                    {STATIC_COLS.map(c => (
                                        <td key={c.key} style={{
                                            ...tdS, minWidth: c.w,
                                            color:      cellColor(c.key, row),
                                            fontWeight: c.key === 'name' ? 600 : 400,
                                        }}>
                                            {resolveCell(c.key, row)}
                                        </td>
                                    ))}
                                    <td style={{ ...tdH, color: '#111', padding: '4px 2px' }}>│</td>
                                    {HIST_COLS.map(c => (
                                        <td key={c.key} style={{
                                            ...tdH, minWidth: c.w,
                                            color: cellColor(c.key, row),
                                        }}>
                                            {resolveCell(c.key, row)}
                                        </td>
                                    ))}
                                </tr>
                            );
                        })}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Footer */}
            {!loading && !error && (
                <div style={{
                    padding: '5px 14px', background: '#07090e',
                    borderTop: '1px solid #151c28',
                    fontSize: 10, color: '#2a3a4a', flexShrink: 0,
                }}>
                    {rows.length} rows · {horseOrder.length} runners
                </div>
            )}
        </div>
    );
}
