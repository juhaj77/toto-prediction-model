import React, { useState, useEffect, useCallback } from 'react';
import RunnerModal from './RunnerModal.jsx';
import * as tf from '@tensorflow/tfjs';

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const MAX_HISTORY = 8;

// ─── FEATURE HELPERS — must match ravimalli.js and scraper.js exactly ─────────

// Transliterate Finnish characters and strip non-ASCII (mirrors scraper.js puhdista)
function sanitize(str) {
    if (!str) return '';
    return str
        .replace(/ä/g, 'a').replace(/ö/g, 'o').replace(/å/g, 'a')
        .replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Å/g, 'A')
        .replace(/[^a-zA-Z0-9\s\-\.\:]/g, '')
        .trim();
}

// Parse raw km-time string from Veikkaus API.
// Format examples: "15,5"  "15,5a" (car start)  "15,5x" (gait fault / break)
function parseKmTime(raw) {
    if (!raw) return { kmNum: 0, isCarStart: false, isBreak: false };
    const s          = String(raw);
    const isCarStart = s.includes('a');
    const isBreak    = s.includes('x');
    const match      = s.match(/[\d,]+/);
    const kmNum      = match ? parseFloat(match[0].replace(',', '.')) : 0;
    return { kmNum: isNaN(kmNum) ? 0 : kmNum, isCarStart, isBreak };
}

// Normalise km time to a 2100 m base distance so times from different
// distances are comparable. Correlation with position: r = 0.031 → 0.090.
function normaliseKmTime(km, distance) {
    if (!km || isNaN(km) || km <= 0) return km;
    return km + (2100 - (distance || 2100)) / 2000;
}

// Parse finishing position from API result string.
// Returns numeric position (1–16), 20 for disqualification, 21 for DNF.
function parsePosition(result) {
    if (!result) return { position: null, disqualified: false, DNF: false };
    const s            = String(result).toLowerCase();
    const disqualified = /[hdp]/.test(s);
    const DNF          = s.includes('k');
    const numMatch     = s.match(/^\d+/);
    let position       = null;
    if (numMatch)          position = parseInt(numMatch[0]);
    else if (disqualified) position = 20;
    else if (DNF)          position = 21;
    return { position, disqualified, DNF };
}

// Gender code → numeric (1 = mare, 2 = gelding, 3 = stallion)
function encodeGender(g) {
    if (g === 'TAMMA') return 1;
    if (g === 'ORI')   return 3;
    return 2;
}

// Select the best available race record for this runner.
// Priority depends on start type: car starts prefer mobileStartRecord.
function parseRecord(runner, isCarStart) {
    const order = isCarStart
        ? ['mobileStartRecord', 'handicapRaceRecord', 'vaultStartRecord']
        : ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord'];
    for (const key of order) {
        const val = runner[key];
        if (!val) continue;
        const num = parseFloat(String(val).replace(',', '.'));
        if (!isNaN(num) && num > 0)
            return { record: num, isAutoRecord: key === 'mobileStartRecord' };
    }
    return { record: null, isAutoRecord: false };
}

// Shoe value normalisation — accept multiple API formats
function encodeShoes(val) {
    if (!val) return 'UNKNOWN';
    const s = String(val).toUpperCase();
    if (['HAS_SHOES', 'TRUE', 'YES', '1'].includes(s)) return 'HAS_SHOES';
    if (['NO_SHOES',  'FALSE', 'NO', '0'].includes(s)) return 'NO_SHOES';
    return 'UNKNOWN';
}

// Betting percentage — betPercentages.KAK.percentage is ×100 (e.g. 1148 = 11.48%)
function parseBettingPct(runner) {
    return parseFloat(runner.betPercentages?.KAK?.percentage ?? 0) / 100;
}

// Historical win percentage from runner stats
function parseWinPct(runner) {
    try {
        const total = runner.stats?.total;
        if (total?.winningPercent != null) return parseFloat(total.winningPercent);
        if (total?.starts > 0) return Math.round((total.position1 / total.starts) * 10000) / 100;
    } catch (_) {}
    return 0;
}

// Track condition → 0–1 scale (0 = heaviest, 1 = lightest)
function encodeTrackCondition(condition) {
    switch ((condition || '').toLowerCase().trim()) {
        case 'heavy track':       return 0.00;
        case 'quite heavy track': return 0.25;
        case 'winter track':      return 0.75;
        case 'light track':       return 1.00;
        default:                  return 0.50;
    }
}

// Use surname only so "J Mäkinen" and "Juhani Mäkinen" resolve to the same ID
function extractSurname(fullName) {
    if (!fullName) return 'unknown';
    return fullName.trim().toLowerCase().split(' ').at(-1);
}

function parseDate(str) {
    if (!str || str === '0' || str === 'NaT') return null;
    if (str.includes('-')) return new Date(str);
    if (str.includes('.')) {
        const [d, m, y] = str.split('.');
        const year = parseInt(y) < 100 ? parseInt(y) + 2000 : parseInt(y);
        return new Date(year, parseInt(m) - 1, parseInt(d));
    }
    return null;
}

// ─── FEATURE BUILDER — must produce identical tensors to ravimalli.js ─────────

function buildFeatures(runners, maps, raceDate, raceDistance, isColdBlood, isCarStart) {
    const breed = isColdBlood ? 'SH' : 'LV';
    const means = { SH: { record: 28.0, km: 29.0 }, LV: { record: 15.0, km: 16.0 } };

    // Look up a stable integer ID from the training-time name maps.
    // Unknown names fall back to 0 (same behaviour as ravimalli.js prediction mode).
    const getID = (map, name, type) => {
        if (!name || name === 'Unknown' || name === '') return 0;
        const key = (type === 'driver') ? extractSurname(name) : name.trim().toLowerCase();
        return map[key] || 0;
    };

    // Per-race relative rank is more informative than the raw percentage value
    const starters      = runners.filter(r => !r.scratched);
    const hasBet        = starters.some(r => (r.bettingPct || 0) > 0);
    const hasWin        = starters.some(r => (r.winPct     || 0) > 0);
    const neutral       = (starters.length + 1) / 2;

    const byBet = [...starters].sort((a, b) => (b.bettingPct || 0) - (a.bettingPct || 0));
    const byWin = [...starters].sort((a, b) => (b.winPct     || 0) - (a.winPct     || 0));

    const X_hist   = [];
    const X_static = [];
    const metadata = [];

    for (const runner of starters) {
        const betKnown  = hasBet && (runner.bettingPct || 0) > 0 ? 1 : 0;
        const winKnown  = hasWin && (runner.winPct     || 0) > 0 ? 1 : 0;
        const betRank   = betKnown ? byBet.findIndex(r => r.number === runner.number) + 1 : neutral;
        const winRank   = winKnown ? byWin.findIndex(r => r.number === runner.number) + 1 : neutral;

        // Shoe encoding
        const frontStr    = (runner.frontShoes || '').toUpperCase();
        const rearStr     = (runner.rearShoes  || '').toUpperCase();
        const frontActive = frontStr === 'HAS_SHOES' ? 1 : 0;
        const frontKnown  = (frontStr === 'HAS_SHOES' || frontStr === 'NO_SHOES') ? 1 : 0;
        const rearActive  = rearStr  === 'HAS_SHOES' ? 1 : 0;
        const rearKnown   = (rearStr  === 'HAS_SHOES' || rearStr  === 'NO_SHOES') ? 1 : 0;

        // Special cart
        const cartStr    = (runner.specialCart || '').toUpperCase();
        const cartActive = cartStr === 'YES' ? 1 : 0;
        const cartKnown  = (cartStr === 'YES' || cartStr === 'NO') ? 1 : 0;

        // Weighted podium index normalised by ALL starts (disq/DNF = 0 pts, still counted).
        // This penalises horses with many non-finishes. Correlation with top-3: r = 0.161.
        // The Veikkaus API returns position as a raw result string (e.g. "1", "hyl", "k").
        const prevStarts  = runner.prevStarts || [];
        const validPrev   = prevStarts.filter(ps => {
            const date   = (ps.shortMeetDate || ps.meetDate || '').trim();
            const driver = (ps.driverFullName || ps.driver  || '').trim();
            return date !== '' && driver !== '';
        });
        const histKnown   = validPrev.length > 0 ? 1 : 0;
        let prevIndexNorm = 0;

        if (validPrev.length > 0) {
            let score = 0, count = 0;
            for (const ps of validPrev) {
                const { position, disqualified, DNF } = parsePosition(ps.result);
                // All starts count toward the denominator — including disq and DNF.
                // Only podium finishes add to the numerator.
                if (position === null) continue;
                count++;
                if      (position === 1) score += 1.00;
                else if (position === 2) score += 0.50;
                else if (position === 3) score += 0.33;
                // position > 3, disqualified (20), DNF (21) → 0 pts
            }
            prevIndexNorm = count > 0 ? score / count : 0;
        }

        // ── Static features — 27 total, same order as ravimalli.js ──────────
        const staticFeats = [
            (runner.number || 1) / 20,                                        // [0]  start number
            getID(maps.coaches, runner.coach,  'coach')  / 2000,              // [1]  coach ID
            (runner.record || means[breed].record) / 50,                      // [2]  race record
            getID(maps.drivers, runner.driver, 'driver') / 3000,              // [3]  driver ID
            (runner.age || 5) / 15,                                           // [4]  age
            (runner.gender || 2) / 3,                                         // [5]  gender
            isColdBlood ? 1 : 0,                                              // [6]  cold blood breed
            frontActive, frontKnown,                                          // [7-8]   front shoes
            rearActive,  rearKnown,                                           // [9-10]  rear shoes
            runner.frontShoesChanged ? 1 : 0,                                 // [11] front shoes changed
            runner.rearShoesChanged  ? 1 : 0,                                 // [12] rear shoes changed
            (raceDistance || 2100) / 3100,                                    // [13] race distance
            isCarStart ? 1 : 0,                                               // [14] car start
            (runner.bettingPct || 0) / 100,                                   // [15] raw betting %
            (runner.winPct     || 0) / 100,                                   // [16] historical win %
            (runner.winPct     || 0) > 0 ? 1 : 0,                             // [17] win % known
            runner.isAutoRecord ? 1 : 0,                                      // [18] record from car start
            cartActive, cartKnown,                                            // [19-20] special cart
            betRank / 20, betKnown,                                           // [21-22] betting rank
            winRank / 20, winKnown,                                           // [23-24] win rank
            prevIndexNorm,                                                    // [25] podium index / n_starts
            histKnown,                                                        // [26] has history
        ];

        // ── History sequence — MAX_HISTORY × 25, same order as ravimalli.js ─
        const histSeq = [];

        for (let i = 0; i < MAX_HISTORY; i++) {
            const ps = validPrev[i];
            if (!ps) { histSeq.push(new Array(25).fill(-1)); continue; }

            const raceDateObj  = parseDate(raceDate);
            const startDateObj = parseDate(ps.shortMeetDate || ps.meetDate);
            const daysSince    = (raceDateObj && startDateObj)
                ? Math.min(365, (raceDateObj - startDateObj) / 86400000)
                : 30;

            const { kmNum, isCarStart: psIsCarStart, isBreak } = parseKmTime(ps.kmTime);
            const kmNorm  = normaliseKmTime(kmNum, ps.distance);
            const kmKnown = kmNorm > 0 ? 1 : 0;
            const kmFinal = kmKnown ? kmNorm / 100 : means[breed].km / 100;

            const distKnown  = (ps.distance || 0) > 0 ? 1 : 0;
            const distFinal  = distKnown ? ps.distance / 3100 : 0.67;

            // firstPrize in Veikkaus API is in cents (e.g. 1000000 = 100 €)
            const prize      = (parseFloat(ps.firstPrize) || 0) / 10000;
            const prizeKnown = prize > 0 ? 1 : 0;
            const prizeFinal = prizeKnown ? Math.log1p(prize) / 10 : 0.55;

            // winOdd is ×10 (e.g. 152 = 15.2)
            const odd      = (parseFloat(ps.winOdd) || 0) / 10;
            const oddKnown = odd > 0 ? 1 : 0;
            const oddFinal = oddKnown ? Math.log1p(odd) / 5 : 0.50;

            const { position, disqualified, DNF } = parsePosition(ps.result);
            const posKnown = position != null && position > 0 ? 1 : 0;
            const posFinal = posKnown ? position / 20 : 0.5;

            const psfront = (ps.frontShoes  || '').toUpperCase();
            const psrear  = (ps.rearShoes   || '').toUpperCase();
            const pscart  = (ps.specialCart || '').toUpperCase();

            histSeq.push([
                kmFinal,    kmKnown,                                           // [0-1]   km time (normalised)
                distFinal,  distKnown,                                         // [2-3]   distance
                daysSince / 365,                                               // [4]     days since start
                posFinal,   posKnown,                                          // [5-6]   finishing position
                prizeFinal, prizeKnown,                                        // [7-8]   prize money (log)
                oddFinal,   oddKnown,                                          // [9-10]  win odds (log)
                psIsCarStart ? 1 : 0,                                          // [11]    car start
                isBreak      ? 1 : 0,                                          // [12]    gait fault
                (ps.startTrack ?? 1) / 30,                                     // [13]    start position
                getID(maps.drivers, ps.driverFullName || ps.driver, 'driver') / 3000, // [14] driver ID
                getID(maps.tracks,  ps.trackCode || ps.track, 'track') / 500,  // [15]    track ID
                disqualified ? 1 : 0,                                          // [16]    disqualified
                DNF          ? 1 : 0,                                          // [17]    did not finish
                psfront === 'HAS_SHOES' ? 1 : 0,                               // [18]    front shoes on
                (psfront === 'HAS_SHOES' || psfront === 'NO_SHOES') ? 1 : 0,   // [19]    front shoes known
                psrear  === 'HAS_SHOES' ? 1 : 0,                               // [20]    rear shoes on
                (psrear  === 'HAS_SHOES' || psrear  === 'NO_SHOES') ? 1 : 0,   // [21]    rear shoes known
                pscart  === 'YES' ? 1 : 0,                                     // [22]    special cart on
                (pscart  === 'YES' || pscart  === 'NO') ? 1 : 0,               // [23]    special cart known
                encodeTrackCondition(ps.trackCondition),                       // [24]    track condition
            ]);
        }

        X_hist.push(histSeq);
        X_static.push(staticFeats);
        metadata.push({ number: runner.number, name: runner.name, driver: runner.driver });
    }

    return { X_hist, X_static, metadata };
}

// ─── JSON FETCH — with explicit error for missing public/ files ────────────────

async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok)
        throw new Error(`HTTP ${res.status} — file not found: ${path}\nCopy the file to public/.`);
    const text = await res.text();
    if (text.trimStart().startsWith('<'))
        throw new Error(`Server returned HTML instead of JSON for: ${path}\nFile is missing from public/.`);
    return JSON.parse(text);
}

// ─── APP ──────────────────────────────────────────────────────────────────────

export default function App() {
    const [cards,           setCards]           = useState([]);
    const [races,           setRaces]           = useState([]);
    const [selectedCard,    setSelectedCard]    = useState('');
    const [selectedRace,    setSelectedRace]    = useState('');
    const [loadingCards,    setLoadingCards]    = useState(false);
    const [loadingRaces,    setLoadingRaces]    = useState(false);
    const [loadingPred,     setLoadingPred]     = useState(false);
    const [predictions,     setPredictions]     = useState([]);
    const [error,           setError]           = useState('');
    const [model,           setModel]           = useState(null);
    const [maps,            setMaps]            = useState(null);
    const [modelInfo,       setModelInfo]       = useState(null);
    const [modelStatus,     setModelStatus]     = useState('idle');
    const [modalOpen,       setModalOpen]       = useState(false);
    const [modalRaceId,     setModalRaceId]     = useState('');
    const [modalRaceLabel,  setModalRaceLabel]  = useState('');
    const [modalRace,       setModalRace]       = useState(null);
    const [modalRunners,    setModalRunners]    = useState(null);

    // Load model + mappings once on mount
    useEffect(() => {
        (async () => {
            setModelStatus('loading');
            try {
                const mappingsData = await fetchJSON('/mappings.json');
                setMaps(mappingsData);

                const modelData = await fetchJSON('/ravimalli-mixed/model_full.json');
                if (modelData.trainingInfo) setModelInfo(modelData.trainingInfo);

                const binary  = atob(modelData.weightData);
                const bytes   = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

                const loaded = await tf.loadLayersModel(tf.io.fromMemory({
                    modelTopology: modelData.modelTopology,
                    weightSpecs:   modelData.weightSpecs,
                    weightData:    bytes.buffer,
                }));
                setModel(loaded);
                setModelStatus('ready');
            } catch (e) {
                console.error('Model load failed:', e);
                setError(e.message);
                setModelStatus('error');
            }
        })();
    }, []);

    // Fetch today's race cards
    useEffect(() => {
        setLoadingCards(true);
        fetch('/api-veikkaus/api/toto-info/v1/cards/today')
            .then(r => r.json())
            .then(d => setCards(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingCards(false));
    }, []);

    // Fetch races when a card is selected
    useEffect(() => {
        if (!selectedCard) { setRaces([]); return; }
        setLoadingRaces(true);
        setSelectedRace('');
        setPredictions([]);
        fetch(`/api-veikkaus/api/toto-info/v1/card/${selectedCard}/races`)
            .then(r => r.json())
            .then(d => setRaces(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingRaces(false));
    }, [selectedCard]);

    // Run prediction for the selected race
    const runPrediction = useCallback(async () => {
        if (!selectedCard || !selectedRace || !model || !maps) return;
        setLoadingPred(true);
        setError('');
        setPredictions([]);

        try {
            const cardData   = await fetch(`/api-veikkaus/api/toto-info/v1/card/${selectedCard}`).then(r => r.json());
            const raceDate   = sanitize(cardData.meetDate || cardData.date || '');

            const raceInfo   = races.find(r => String(r.number) === String(selectedRace));
            if (!raceInfo) throw new Error('Race not found');

            const raceId      = String(raceInfo.raceId || raceInfo.id);
            const raceDistance = parseInt(raceInfo.distance || 2100);
            const isColdBlood  = (raceInfo.breed === 'K' || raceInfo.breed === 'FINNHORSE');
            const isCarStart   = (raceInfo.startType === 'CAR_START' || raceInfo.startType === 'AUTO');

            const runnersRaw = await fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`).then(r => r.json());
            setModalRunners({ raw: runnersRaw, race: raceInfo });

            const runnersArr = Array.isArray(runnersRaw)
                ? runnersRaw
                : (runnersRaw.collection || runnersRaw.runners || runnersRaw.data || Object.values(runnersRaw));

            console.log('[runners] API response keys:', Object.keys(runnersRaw));
            console.log('[runners] Found', runnersArr.length, 'entries before filter');

            // Normalise runner objects to internal shape
            const runners = runnersArr
                .filter(r => r.scratched !== true)
                .map(r => {
                    const { record, isAutoRecord } = parseRecord(r, isCarStart);
                    return {
                        number:            parseInt(r.startNumber || r.number || 0),
                        name:              sanitize(r.horseName || r.name || ''),
                        coach:             sanitize(r.coachName || r.trainerName || ''),
                        driver:            sanitize(r.driverName || ''),
                        age:               parseInt(r.horseAge || r.age || 0),
                        gender:            encodeGender(r.gender),
                        frontShoes:        encodeShoes(r.frontShoes),
                        rearShoes:         encodeShoes(r.rearShoes),
                        frontShoesChanged: r.frontShoesChanged === true,
                        rearShoesChanged:  r.rearShoesChanged  === true,
                        specialCart:       r.specialCart || 'UNKNOWN',
                        scratched:         r.scratched   === true,
                        record,
                        isAutoRecord,
                        bettingPct:        parseBettingPct(r),
                        winPct:            parseWinPct(r),
                        prevStarts:        r.prevStarts || [],
                    };
                });

            console.log('[runners] After filter:', runners.length);

            if (runners.length === 0)
                throw new Error(
                    `No runners found in race.\nAPI keys: [${Object.keys(runnersRaw).join(', ')}]\n` +
                    `Entries before filter: ${runnersArr.length}\nSee DevTools console for details.`
                );

            const { X_hist, X_static, metadata } = buildFeatures(
                runners, maps, raceDate, raceDistance, isColdBlood, isCarStart
            );

            const histTensor   = tf.tensor3d(X_hist);
            const staticTensor = tf.tensor2d(X_static);
            const pred         = model.predict([histTensor, staticTensor]);
            const scores       = await pred.data();
            histTensor.dispose();
            staticTensor.dispose();
            pred.dispose();

            setPredictions(
                metadata
                    .map((m, i) => ({ number: m.number, name: m.name, driver: m.driver, prob: scores[i] }))
                    .sort((a, b) => b.prob - a.prob)
            );
        } catch (e) {
            console.error(e);
            setError(e.message || 'Unknown error');
        } finally {
            setLoadingPred(false);
        }
    }, [selectedCard, selectedRace, model, maps, races]);

    // Reset body margin (override Vite defaults)
    useEffect(() => {
        document.body.style.margin  = '0';
        document.body.style.padding = '0';
        document.body.style.width   = '100%';
        document.documentElement.style.width = '100%';
    }, []);

    const statusColor  = { idle: '#888', loading: '#f0a500', ready: '#2ecc71', error: '#e74c3c' };
    const statusLabel  = { idle: 'Idle', loading: 'Loading model…', ready: 'Model ready', error: 'Load error' };
    const canRun       = selectedRace && modelStatus === 'ready' && !loadingPred;

    return (
        <div style={{ minHeight: '100vh', background: '#1a1a1a', boxSizing: 'border-box', padding: '40px 24px' }}>
            <div style={{ background: '#0d0f14', width: 'fit-content', minWidth: 700, margin: '0 auto',
                color: '#e8eaf0', fontFamily: "'IBM Plex Mono','Courier New',monospace",
                padding: '32px 40px', borderRadius: 8 }}>

                {/* ── Header ── */}
                <div style={{ marginBottom: 28, borderBottom: '1px solid #1e2330', paddingBottom: 20 }}>
                    <div style={{ fontSize: 11, letterSpacing: 4, color: '#4a90d9', textTransform: 'uppercase', marginBottom: 6 }}>
                        Toto Prediction System
                    </div>
                    <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: '#fff', letterSpacing: -0.5 }}>
                        RaviMalli v5
                    </h1>

                    <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', fontSize: 12 }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
                                background: statusColor[modelStatus], boxShadow: `0 0 6px ${statusColor[modelStatus]}` }} />
                            <span style={{ color: statusColor[modelStatus] }}>{statusLabel[modelStatus]}</span>
                        </span>

                        {modelInfo && (
                            <span style={{ color: '#556', fontSize: 11, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                                <span>Epoch <b style={{ color: '#aaa' }}>{modelInfo.epoch}</b></span>
                                <span>val_loss <b style={{ color: '#aaa' }}>{modelInfo.val_loss}</b></span>
                                <span>val_acc <b style={{ color: '#aaa' }}>{modelInfo.val_acc != null ? (modelInfo.val_acc * 100).toFixed(1) + '%' : '—'}</b></span>
                                <span>LR <b style={{ color: '#aaa' }}>{modelInfo.learningRate}</b></span>
                                <span style={{ borderLeft: '1px solid #222', paddingLeft: 14 }}>
                                    Learning Data <b style={{ color: '#aaa' }}>{modelInfo.dataStartDate} → {modelInfo.dataEndDate}</b>
                                </span>
                                <span>races <b style={{ color: '#aaa' }}>{modelInfo.totalRaces?.toLocaleString('fi-FI')}</b></span>
                                <span style={{ color: '#333' }}>·</span>
                                <span>runners <b style={{ color: '#aaa' }}>{modelInfo.totalRunners?.toLocaleString('fi-FI')}</b></span>
                            </span>
                        )}
                    </div>
                </div>

                {/* ── Model load error ── */}
                {modelStatus === 'error' && error && (
                    <div style={{ padding: '12px 16px', background: '#1a0a0a', border: '1px solid #e74c3c',
                        borderRadius: 4, color: '#e74c3c', fontSize: 12, marginBottom: 24, whiteSpace: 'pre-wrap' }}>
                        ✗ {error}
                    </div>
                )}

                {/* ── Controls ── */}
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 32 }}>
                    <div>
                        <label style={labelStyle}>Track</label>
                        <select value={selectedCard} onChange={e => setSelectedCard(e.target.value)}
                                disabled={loadingCards} style={selectStyle}>
                            <option value="">{loadingCards ? 'Loading…' : '— Select track —'}</option>
                            {cards.map(c => (
                                <option key={c.cardId} value={c.cardId}>{c.country} · {c.trackName}</option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <label style={labelStyle}>Race</label>
                        <select value={selectedRace} onChange={e => setSelectedRace(e.target.value)}
                                disabled={!selectedCard || loadingRaces} style={{ ...selectStyle, minWidth: 160 }}>
                            <option value="">{loadingRaces ? '…' : '—'}</option>
                            {races.map(r => (
                                <option key={r.raceId} value={r.number}>Race {r.number} · {r.distance}m</option>
                            ))}
                        </select>
                    </div>

                    <button onClick={runPrediction} disabled={!canRun} style={{
                        padding: '10px 24px',
                        background: canRun ? '#4a90d9' : '#1e2330',
                        color:      canRun ? '#fff'    : '#444',
                        border: '1px solid #2a3040', borderRadius: 4,
                        fontFamily: 'inherit', fontSize: 13, letterSpacing: 1,
                        cursor: canRun ? 'pointer' : 'not-allowed', transition: 'all 0.2s',
                    }}>
                        {loadingPred ? 'Running…' : '▶ Run prediction'}
                    </button>

                    {selectedRace && (
                        <button onClick={() => {
                            const race = races.find(r => String(r.number) === String(selectedRace));
                            if (!race) return;
                            setModalRaceId(String(race.raceId || race.id));
                            setModalRaceLabel(`Race ${selectedRace} · ${race.distance}m`);
                            setModalRace(race);
                            setModalOpen(true);
                        }} style={{
                            padding: '10px 20px', background: '#0f1520', color: '#4a90d9',
                            border: '1px solid #2a4060', borderRadius: 4,
                            fontFamily: 'inherit', fontSize: 13, letterSpacing: 1,
                            cursor: 'pointer', transition: 'all 0.2s',
                        }}>
                            ⊞ Race details
                        </button>
                    )}
                </div>

                {/* ── Prediction error ── */}
                {error && modelStatus !== 'error' && (
                    <div style={{ padding: '12px 16px', background: '#1a0a0a', border: '1px solid #e74c3c',
                        borderRadius: 4, color: '#e74c3c', fontSize: 13, marginBottom: 24 }}>
                        ✗ {error}
                    </div>
                )}

                {/* ── Results table ── */}
                {predictions.length > 0 && (
                    <div>
                        <div style={{ fontSize: 11, letterSpacing: 2, color: '#4a90d9',
                            textTransform: 'uppercase', marginBottom: 12 }}>
                            Prediction · Race {selectedRace}
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                                <thead>
                                <tr style={{ borderBottom: '1px solid #1e2330' }}>
                                    {['#', 'No', 'Horse', 'Driver', 'Probability', 'Implied odds', 'Signal'].map(h => (
                                        <th key={h} style={{ padding: '8px 12px', textAlign: 'left',
                                            color: '#4a90d9', fontWeight: 500, fontSize: 11,
                                            letterSpacing: 1, textTransform: 'uppercase' }}>
                                            {h}
                                        </th>
                                    ))}
                                </tr>
                                </thead>
                                <tbody>
                                {predictions.map((p, i) => (
                                    <tr key={p.name} style={{ borderBottom: '1px solid #12151c',
                                        background: i === 0 ? '#0d1a2a' : i % 2 === 0 ? '#0f1118' : 'transparent' }}>
                                        <td style={{ padding: '10px 12px', color: i < 3 ? '#f0a500' : '#444' }}>
                                            {i < 3 ? ['①', '②', '③'][i] : `#${i + 1}`}
                                        </td>
                                        <td style={{ padding: '10px 12px', color: '#666' }}>{p.number}</td>
                                        <td style={{ padding: '10px 12px', fontWeight: 600 }}>{p.name}</td>
                                        <td style={{ padding: '10px 12px', color: '#aaa' }}>{p.driver || '—'}</td>
                                        <td style={{ padding: '10px 12px' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                <div style={{ height: 4, minWidth: 4, width: `${Math.round(p.prob * 120)}px`,
                                                    background: p.prob > 0.5 ? '#2ecc71' : p.prob > 0.3 ? '#f0a500' : '#333',
                                                    borderRadius: 2 }} />
                                                <span>{(p.prob * 100).toFixed(1)}%</span>
                                            </div>
                                        </td>
                                        <td style={{ padding: '10px 12px', color: '#aaa', fontVariantNumeric: 'tabular-nums' }}>
                                            {(1 / p.prob).toFixed(2)}
                                        </td>
                                        <td style={{ padding: '10px 12px' }}>
                                                <span style={{ padding: '2px 10px', borderRadius: 3, fontSize: 11, letterSpacing: 1,
                                                    background: p.prob > 0.5 ? '#0d2a1a' : '#111',
                                                    color:      p.prob > 0.5 ? '#2ecc71'  : '#444',
                                                    border: `1px solid ${p.prob > 0.5 ? '#2ecc71' : '#222'}` }}>
                                                    {p.prob > 0.5 ? 'BET' : 'SKIP'}
                                                </span>
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* ── Runner detail modal ── */}
                {modalOpen && (
                    <RunnerModal
                        raceId={modalRaceId}
                        race={modalRace}
                        raceLabel={modalRaceLabel}
                        preloadedData={modalRunners}
                        onClose={() => setModalOpen(false)}
                    />
                )}
            </div>
        </div>
    );
}

const labelStyle = {
    display: 'block', fontSize: 11, letterSpacing: 2,
    color: '#4a90d9', marginBottom: 6, textTransform: 'uppercase',
};
const selectStyle = {
    padding: '10px 12px', background: '#0f1118',
    border: '1px solid #2a3040', borderRadius: 4,
    color: '#e8eaf0', fontFamily: "'IBM Plex Mono','Courier New',monospace",
    fontSize: 13, minWidth: 220, outline: 'none',
};
