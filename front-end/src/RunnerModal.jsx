import React, { useEffect, useState } from 'react';

// ─────────────────────────────────────────────────────────────────────────────
// SARAKKEET — täsmälleen CSV:n järjestys (ilman RaceID, Scratched, SIJOITUS)
// ─────────────────────────────────────────────────────────────────────────────
const STATIC_COLS = [
    { key: 'Nro',                 label: 'Nro',         w: 42 },
    { key: 'Nimi',                label: 'Nimi',         w: 175 },
    { key: 'Valmentaja',          label: 'Valmentaja',   w: 150 },
    { key: 'Current_Ohjastaja',   label: 'Ohj.',         w: 150 },
    { key: 'Ika',                 label: 'Ikä',          w: 36 },
    { key: 'Sukupuoli',           label: 'Sp',           w: 30 },
    { key: 'Is_Suomenhevonen',    label: 'SH',           w: 30 },
    { key: 'Kengat_Etu',          label: 'K.Etu',        w: 68 },
    { key: 'Kengat_Taka',         label: 'K.Taka',       w: 68 },
    { key: 'Kengat_etu_changed',  label: 'Etu↑',         w: 40 },
    { key: 'Kengat_taka_changed', label: 'Taka↑',        w: 46 },
    { key: 'Current_Special_Cart',label: 'Cart',         w: 46 },
    { key: 'Current_Distance',    label: 'Matka',        w: 56 },
    { key: 'Current_Is_Auto',     label: 'Auto',         w: 40 },
    { key: 'Current_Start_Date',  label: 'Päivä',        w: 90 },
    { key: 'Peli_pros',           label: 'Peli%',        w: 60 },
    { key: 'Voitto_pros',         label: 'Voitto%',      w: 66 },
    { key: 'Ennatys_nro',         label: 'Ennätys',      w: 70 },
    { key: 'Is_Auto_Record',      label: 'AutoRec',      w: 60 },
    { key: 'PrevIndeksiNorm',     label: 'Podium%',      w: 68 },
];
const HIST_COLS = [
    { key: 'Hist_PVM',            label: 'Hist.PVM',     w: 86 },
    { key: 'Ohjastaja',           label: 'Hist.Ohj',     w: 140 },
    { key: 'Rata',                label: 'Rata',         w: 46 },
    { key: 'Matka',               label: 'Matka',        w: 56 },
    { key: 'RataNro',             label: 'RataNro',      w: 60 },
    { key: 'Km_aika',             label: 'Km-aika',      w: 72 },
    { key: 'Hist_kengat_etu',     label: 'K.Etu',        w: 58 },
    { key: 'Hist_kengat_taka',    label: 'K.Taka',       w: 58 },
    { key: 'Hist_Special_Cart',   label: 'Cart',         w: 46 },
    { key: 'Hist_Is_Auto',        label: 'Auto',         w: 40 },
    { key: 'Laukka',              label: 'Laukka',       w: 50 },
    { key: 'Hylatty',             label: 'Hyl.',         w: 40 },
    { key: 'Keskeytys',           label: 'Kesk.',        w: 46 },
    { key: 'Track_Condition',     label: 'Rata kunto',   w: 100 },
    { key: 'Hist_Sij',            label: 'Sij.',         w: 40 },
    { key: 'Kerroin',             label: 'Kerroin',      w: 66 },
    { key: 'Palkinto',            label: 'Palkinto €',   w: 86 },
];

// ─────────────────────────────────────────────────────────────────────────────
// PARSINTA — täsmälleen MATLAB-scraperin logiikka JSON-kentillä
// ─────────────────────────────────────────────────────────────────────────────
function fmtKengat(v) {
    const s = String(v || '').toUpperCase();
    if (s === 'HAS_SHOES') return '✓';
    if (s === 'NO_SHOES')  return '—';
    return '?';
}
function fmtCart(v) {
    const s = String(v || '').toUpperCase();
    if (s === 'YES') return '✓';
    if (s === 'NO')  return '—';
    return s || '?';
}

function parseEnnatys(r) {
    // MATLAB: eMatch = regexp(block, '"(handicapRaceRecord|mobileStartRecord|vaultStartRecord)":"([\d,]+)[a-z]*"')
    for (const key of ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord']) {
        const val = r[key];
        if (val) {
            const numStr = String(val).match(/[\d,]+/)?.[0] || '';
            const n = parseFloat(numStr.replace(',', '.'));
            if (!isNaN(n) && n > 0)
                return { ennatys: n, isAutoRecord: key === 'mobileStartRecord' ? 1 : 0 };
        }
    }
    return { ennatys: 0, isAutoRecord: 0 };
}

function parsePeliP(r) {
    // MATLAB: peliP = str2double(pMatch{1}) / 100
    // JSON: betPercentages.KAK.percentage = 1148 → /100 = 11.48%
    const pct = r.betPercentages?.KAK?.percentage ?? 0;
    return parseFloat(pct) / 100;
}

function parseVoittoP(r) {
    // MATLAB: statBlock → total.starts / total.position1
    // App.jsx käyttää myös total.winningPercent suoraan jos saatavilla
    const total = r.stats?.total;
    if (!total) return 0;
    if (total.winningPercent != null) return parseFloat(total.winningPercent);
    if (total.starts > 0) return Math.round((total.position1 / total.starts) * 10000) / 100;
    return 0;
}

function parseSukupuoli(g) {
    if (g === 'TAMMA') return 1;
    if (g === 'ORI')   return 3;
    return 2; // RUUNA tai tuntematon
}

function parseKmTime(kmRaw) {
    // MATLAB: histIsAuto = ~isempty(strfind(kmRaw,'a')), isLaukka = ~isempty(strfind(kmRaw,'x'))
    // kmNum = str2double(strrep(regexp(kmRaw,'[\d,]+','match','once'),',','.'))
    if (!kmRaw) return { kmNum: 0, histIsAuto: 0, isLaukka: 0, display: '' };
    const s        = String(kmRaw);
    const histIsAuto = s.includes('a') ? 1 : 0;
    const isLaukka   = s.includes('x') ? 1 : 0;
    const numStr   = s.match(/[\d,]+/)?.[0] || '';
    const kmNum    = parseFloat(numStr.replace(',', '.')) || 0;
    const suffix   = isLaukka ? 'x' : histIsAuto ? 'a' : '';
    const display  = kmNum > 0 ? kmNum.toFixed(2) + suffix : '';
    return { kmNum, histIsAuto, isLaukka, display };
}

function parseSijoitus(result) {
    // MATLAB: isHyl=regexp(sijRaw,'[hdp]'), isKesk=strfind(sijRaw,'k')
    if (!result) return { sijNum: 10, isHyl: 0, isKesk: 0 };
    const s      = String(result).toLowerCase();
    const isHyl  = /[hdp]/.test(s) ? 1 : 0;
    const isKesk = s.includes('k') ? 1 : 0;
    const m      = s.match(/^\d+/);
    let sijNum = 10;
    if (m)           sijNum = parseInt(m[0]);
    else if (isHyl)  sijNum = 20;
    else if (isKesk) sijNum = 21;
    return { sijNum, isHyl, isKesk };
}

function fmtSij(n) {
    if (n === 20) return 'Hyl';
    if (n === 21) return 'Kesk';
    if (n === 10 || n === 0) return '—';
    return String(n);
}

// ─────────────────────────────────────────────────────────────────────────────
// buildRows — rakentaa flat CSV-tyyliset rivit JSON-datasta
// Vastaa täsmälleen MATLAB-scraperin KaikkiData-taulukkoa
// ─────────────────────────────────────────────────────────────────────────────
function buildRows(runnersArr, raceInfo) {
    const rows = [];

    // raceInfo tulee races-endpointista (sama kuin App.jsx:n lahdot-array)
    const currDist   = raceInfo?.distance   || '';
    const currIsAuto = (raceInfo?.startType === 'CAR_START') ? 1 : 0;
    const currDate   = ''; // meetDate tulee card-endpointista — jätetään tyhjäksi modalissa

    for (const r of runnersArr) {
        const { ennatys, isAutoRecord } = parseEnnatys(r);

        // prevIndeksiNorm
        const _prevAll = r.prevStarts || r.priorStarts || [];
        let _prevIndeksiNorm = 0;
        { let ind = 0, n = 0;
            _prevAll.forEach(h => {
                const s = String(h.result || ''), m = s.match(/^\d+/);
                if (!m) return; const sij = parseInt(m[0]);
                if (sij < 1 || sij > 16) return; n++;
                if (sij===1) ind+=1.00; else if (sij===2) ind+=0.50; else if (sij===3) ind+=0.33;
            });
            _prevIndeksiNorm = n > 0 ? ind / n : 0;
        }

        const base = {
            Nro:                  r.startNumber             || '',
            Nimi:                 r.horseName               || '',
            Valmentaja:           r.coachName               || '',
            Current_Ohjastaja:    r.driverName              || '',
            Ika:                  r.horseAge                || '',
            Sukupuoli:            (['T','1',1].includes(parseSukupuoli(r.gender)) ? 'T' :
                parseSukupuoli(r.gender) === 3 ? 'O' : 'R'),
            Is_Suomenhevonen:     (r.breed === 'K' || r.breed === 'FINNHORSE') ? '✓' : '',
            Kengat_Etu:           fmtKengat(r.frontShoes),
            Kengat_Taka:          fmtKengat(r.rearShoes),
            Kengat_etu_changed:   r.frontShoesChanged  ? '↑' : '',
            Kengat_taka_changed:  r.rearShoesChanged   ? '↑' : '',
            Current_Special_Cart: fmtCart(r.specialCart),
            Current_Distance:     currDist,
            Current_Is_Auto:      currIsAuto ? '✓' : '',
            Current_Start_Date:   currDate,
            Peli_pros:            parsePeliP(r) > 0 ? parsePeliP(r).toFixed(2) + '%' : '—',
            Voitto_pros:          parseVoittoP(r) > 0 ? parseVoittoP(r).toFixed(1) + '%' : '—',
            Ennatys_nro:          ennatys > 0 ? ennatys.toFixed(2) : '—',
            Is_Auto_Record:       isAutoRecord ? '✓' : '',
            PrevIndeksiNorm:      _prevIndeksiNorm > 0 ? _prevIndeksiNorm.toFixed(3) : '—',
        };

        // KRIITTINEN: JSON:issa historia on "prevStarts" — EI "priorStarts"!
        const prevStarts = r.prevStarts || r.priorStarts || [];

        if (prevStarts.length === 0) {
            rows.push({ ...base,
                Hist_PVM: '', Ohjastaja: '', Rata: '', Matka: '', RataNro: '',
                Km_aika: '', Hist_kengat_etu: '', Hist_kengat_taka: '',
                Hist_Special_Cart: '', Hist_Is_Auto: '', Laukka: '',
                Hylatty: '', Keskeytys: '', Track_Condition: '',
                Hist_Sij: '', _sijNum: 10, _isLaukka: 0, _isHyl: 0,
                Kerroin: '—', Palkinto: '—', _palkinto: 0,
            });
        } else {
            for (const h of prevStarts) {
                const { kmNum, histIsAuto, isLaukka, display } = parseKmTime(h.kmTime);
                const { sijNum, isHyl, isKesk }                = parseSijoitus(h.result);

                // MATLAB: winOdd / 10  ("103" → 10.3)
                const kerroin  = parseFloat(h.winOdd || 0) / 10;
                // MATLAB: firstPrize / 100  — mutta JSON:issa arvo on senttejä×10
                // JSON firstPrize=1000000, CSV palkinto=100 → jaetaan /10000
                const palkinto = parseFloat(h.firstPrize || 0) / 10000;

                rows.push({ ...base,
                    Hist_PVM:          h.shortMeetDate        || '',
                    Ohjastaja:         h.driverFullName || h.driver || '',
                    Rata:              h.trackCode             || '',
                    Matka:             h.distance              || '',
                    RataNro:           h.startTrack            || '',
                    Km_aika:           display,
                    Hist_kengat_etu:   fmtKengat(h.frontShoes),
                    Hist_kengat_taka:  fmtKengat(h.rearShoes),
                    Hist_Special_Cart: fmtCart(h.specialCart),
                    Hist_Is_Auto:      histIsAuto ? '✓' : '',
                    Laukka:            isLaukka   ? '✗' : '',
                    Hylatty:           isHyl      ? '✗' : '',
                    Keskeytys:         isKesk     ? '✗' : '',
                    // trackCondition EI ole tässä runners-endpointissa
                    Track_Condition:   h.trackCondition || '',
                    Hist_Sij:          fmtSij(sijNum),
                    _sijNum:           sijNum,
                    _isLaukka:         isLaukka,
                    _isHyl:            isHyl,
                    Kerroin:           kerroin  > 0 ? kerroin.toFixed(1)                       : '—',
                    Palkinto:          palkinto > 0 ? Math.round(palkinto).toLocaleString('fi-FI') : '—',
                    _palkinto:         palkinto,
                });
            }
        }
    }
    return rows;
}

// ─────────────────────────────────────────────────────────────────────────────
// Värikoodaus
// ─────────────────────────────────────────────────────────────────────────────
function getColor(colKey, row) {
    if (colKey === 'Nimi')    return '#e8eaf0';
    if (colKey === 'Nro')     return '#c8a040';
    if (colKey === 'Hist_Sij') {
        if (row._sijNum === 1)               return '#f0a500';
        if (row._sijNum === 2 || row._sijNum === 3) return '#4a90d9';
        if (row._sijNum >= 20)               return '#e74c3c';
    }
    if ((colKey === 'Laukka' || colKey === 'Hylatty' || colKey === 'Keskeytys') && row[colKey] === '✗')
        return '#e74c3c';
    if (colKey === 'Km_aika' && (row._isLaukka || row._isHyl))
        return '#e74c3c';
    if (colKey === 'Palkinto' && row._palkinto > 0)
        return '#2ecc71';
    if (HIST_COLS.find(c => c.key === colKey))
        return '#6a7a8a';
    return '#7a8a9a';
}

// ─────────────────────────────────────────────────────────────────────────────
// Komponentti
// ─────────────────────────────────────────────────────────────────────────────
export default function RunnerModal({ raceId, race, raceLabel, preloadedData, onClose }) {
    const [rows,    setRows]    = useState([]);
    const [loading, setLoading] = useState(true);
    const [error,   setError]   = useState('');

    // ESC sulkee
    useEffect(() => {
        const h = e => { if (e.key === 'Escape') onClose(); };
        window.addEventListener('keydown', h);
        return () => window.removeEventListener('keydown', h);
    }, [onClose]);

    useEffect(() => {
        if (!raceId) return;

        // Jos App.jsx on jo hakenut datan ennustuksen yhteydessä — käytetään sitä
        if (preloadedData?.raw) {
            try {
                const raw = preloadedData.raw;
                const arr = Array.isArray(raw)
                    ? raw
                    : (raw.collection || raw.runners || raw.data || Object.values(raw));
                const filtered = arr.filter(r => r.scratched !== true);
                console.log('[modal] preloaded', filtered.length, 'hevosta | prevStarts[0]:', filtered[0]?.prevStarts?.length);
                setRows(buildRows(filtered, preloadedData.race || race));
            } catch(e) { setError(e.message); }
            finally { setLoading(false); }
            return;
        }

        // Fallback: hae itse (käytetään kun modal avataan ilman ennustusta)
        setLoading(true); setError('');
        fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`)
            .then(r => r.json())
            .then(raw => {
                const arr = Array.isArray(raw)
                    ? raw
                    : (raw.collection || raw.runners || raw.data || Object.values(raw));
                const filtered = arr.filter(r => r.scratched !== true);
                console.log('[modal] fetched', filtered.length, 'hevosta | prevStarts[0]:', filtered[0]?.prevStarts?.length);
                setRows(buildRows(filtered, race));
            })
            .catch(e => setError(e.message))
            .finally(() => setLoading(false));
    }, [raceId, preloadedData]);

    // Hevosittain vuorottelevat taustavärit
    const horseOrder = [];
    const seen = {};
    rows.forEach(r => { if (!seen[r.Nimi]) { seen[r.Nimi] = true; horseOrder.push(r.Nimi); } });
    const horseIdx = {};
    horseOrder.forEach((n, i) => horseIdx[n] = i);

    const bgStatic = ['#0d1018', '#0f1320'];
    const bgHist   = ['#08090f', '#090c15'];

    const TH = (hist) => ({
        position: 'sticky', top: 0, zIndex: 2,
        background: hist ? '#06070c' : '#07090e',
        color: '#3a6090', fontWeight: 700, fontSize: 10.5,
        letterSpacing: 1.2, textTransform: 'uppercase',
        padding: '6px 9px', whiteSpace: 'nowrap',
        borderBottom: '2px solid #151c28',
        fontFamily: "'IBM Plex Mono','Courier New',monospace",
    });

    const uniq = horseOrder.length;

    return (
        <div style={{
            position: 'fixed', inset: 0, zIndex: 1000,
            background: 'rgba(0,0,0,0.94)',
            display: 'flex', flexDirection: 'column',
            fontFamily: "'IBM Plex Mono','Courier New',monospace",
        }}>
            {/* Topbar */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: 14, flexShrink: 0,
                padding: '8px 14px', background: '#07090e',
                borderBottom: '1px solid #151c28',
            }}>
        <span style={{ fontSize: 10, letterSpacing: 3, color: '#3a6090', textTransform: 'uppercase' }}>
          LÄHTÖTIEDOT
        </span>
                <span style={{ fontSize: 13, fontWeight: 700, color: '#e8eaf0' }}>{raceLabel}</span>
                <span style={{ marginLeft: 'auto', fontSize: 10, color: '#334' }}>ESC sulkee</span>
                <button onClick={onClose} style={{
                    background: 'none', border: '1px solid #1e2c40', color: '#ccd',
                    borderRadius: 4, padding: '2px 11px', fontFamily: 'inherit',
                    fontSize: 15, cursor: 'pointer', marginLeft: 8,
                }}>✕</button>
            </div>

            {/* Taulukko */}
            <div style={{ flex: 1, overflow: 'auto' }}>
                {loading && <div style={{ padding: 40, color: '#4a90d9', fontSize: 13 }}>Ladataan…</div>}
                {error   && <div style={{ padding: 40, color: '#e74c3c', fontSize: 13 }}>✗ {error}</div>}

                {!loading && !error && rows.length === 0 && (
                    <div style={{ padding: 40, color: '#664', fontSize: 13 }}>Ei dataa</div>
                )}

                {!loading && !error && rows.length > 0 && (
                    <table style={{ borderCollapse: 'collapse', minWidth: '100%' }}>
                        <thead>
                        <tr>
                            {STATIC_COLS.map(c => (
                                <th key={c.key} style={{ ...TH(false), minWidth: c.w }}>{c.label}</th>
                            ))}
                            {/* Erotin */}
                            <th style={{ ...TH(true), color: '#111', padding: '6px 2px', minWidth: 6 }}>│</th>
                            {HIST_COLS.map(c => (
                                <th key={c.key} style={{ ...TH(true), minWidth: c.w }}>{c.label}</th>
                            ))}
                        </tr>
                        </thead>
                        <tbody>
                        {rows.map((row, i) => {
                            const hi      = horseIdx[row.Nimi] ?? 0;
                            const newHorse = i === 0 || rows[i-1]?.Nimi !== row.Nimi;
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
                                            color: getColor(c.key, row),
                                            fontWeight: c.key === 'Nimi' ? 600 : 400,
                                        }}>
                                            {String(row[c.key] ?? '')}
                                        </td>
                                    ))}
                                    <td style={{ ...tdH, color: '#111', padding: '4px 2px' }}>│</td>
                                    {HIST_COLS.map(c => (
                                        <td key={c.key} style={{
                                            ...tdH, minWidth: c.w,
                                            color: getColor(c.key, row),
                                        }}>
                                            {String(row[c.key] ?? '')}
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
                    {rows.length} riviä · {uniq} hevosta
                </div>
            )}
        </div>
    );
}
