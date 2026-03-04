import React, { useState, useEffect, useCallback } from 'react';
import RunnerModal from './RunnerModal.jsx';
import * as tf from '@tensorflow/tfjs';
import { registerMultiHeadAttention } from './MultiHeadAttention.js';
import LearningCurves from './LearningCurves.jsx';
import {
    sanitize, detectColdBlood, normaliseRunners,
    buildRunnerBasedFeatures, buildRaceBasedFeatures,
    fetchJSON, MAX_RUNNERS,
} from './util.js';

// TensorFlow.js backend initialisation is asynchronous. We must wait until the
// desired backend is ready BEFORE loading models or running inference; otherwise
// different refreshes may end up on different backends (e.g. WebGL vs CPU),
// which can cause small numeric differences to amplify in the race-based model.
// We'll gate the app’s ML actions behind a `tfReady` flag.

// ─── MODEL REGISTRY ───────────────────────────────────────────────────────────
// Runner-based uses mappings.json (model_runner.js writes it).
// Race-based uses mappings_race.json (model_race.js writes it).

const MODEL_VARIANTS = {
    runner: {
        id:           'runner',
        label:        'Runner-based',
        description:  'LSTM + Dense  ·  27 static features  ·  tensor: [n_runners, 8, 25]',
        modelPath:    '/model-runner/model.json',
        mappingsPath: '/mappings_runner.json',
        buildFeatures: buildRunnerBasedFeatures,
        extractScores: (scores, metadata) =>
            metadata.map((m, i) => ({ ...m, prob: scores[i] })),
    },
    race: {
        id:           'race',
        label:        'Race-based',
        description:  'TimeDistributed LSTM + Attention  ·  25 static features  ·  tensor: [1, 18, 8, 25]',
        modelPath:    '/model-race/model.json',
        mappingsPath: '/mappings_race.json',
        buildFeatures: buildRaceBasedFeatures,
        // Output shape [1, MAX_RUNNERS, 1] — index by runner slot
        extractScores: (scores, metadata) =>
            metadata.map(m => ({ ...m, prob: scores[m.slot] })),
    },
};

// ─── APP ──────────────────────────────────────────────────────────────────────

export default function App() {
    const [tfReady, setTfReady] = useState(false);
    const [cards,          setCards]          = useState([]);
    const [races,          setRaces]          = useState([]);
    const [selectedCard,   setSelectedCard]   = useState('');
    const [selectedRace,   setSelectedRace]   = useState('');
    const [loadingCards,   setLoadingCards]   = useState(false);
    const [loadingRaces,   setLoadingRaces]   = useState(false);
    const [loadingPred,    setLoadingPred]    = useState(false);
    const [predictions,    setPredictions]    = useState([]);
    const [error,          setError]          = useState('');
    const [modalOpen,      setModalOpen]      = useState(false);
    const [modalRaceId,    setModalRaceId]    = useState('');
    const [modalRaceLabel, setModalRaceLabel] = useState('');
    const [modalRace,      setModalRace]      = useState(null);
    const [modalRunners,   setModalRunners]   = useState(null);
    const [activeVariant,  setActiveVariant]  = useState('runner');

    // Per-variant state: model weights + mappings + status
    const [models, setModels] = useState({
        runner: { model: null, maps: null, info: null, status: 'idle' },
        race:   { model: null, maps: null, info: null, status: 'idle' },
    });

    // Initialise TF backend deterministically (CPU) and register custom layers
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                await tf.ready();
                const desired = 'cpu';
                if (tf.getBackend() !== desired) {
                    await tf.setBackend(desired);
                }
                await tf.ready();
                if (cancelled) return;
                console.info('[tf] backend ready:', tf.getBackend());
                // Register the custom layer AFTER TF is ready
                registerMultiHeadAttention();
                setTfReady(true);
            } catch (e) {
                console.error('[tf] init failed:', e);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    // ── Load a model variant (model weights + its own mappings file) ──────────
    const loadVariant = useCallback(async (variantId) => {
        if (!tfReady) return; // wait until backend ready
        if (['ready', 'loading'].includes(models[variantId].status)) return;

        const variant = MODEL_VARIANTS[variantId];
        setModels(prev => ({ ...prev, [variantId]: { ...prev[variantId], status: 'loading' } }));

        try {
            // Load model weights and mappings in parallel
            const [modelData, mapsData] = await Promise.all([
                fetchJSON(variant.modelPath),
                fetchJSON(variant.mappingsPath),
            ]);

            const info   = modelData.trainingInfo || null;
            const binary = atob(modelData.weightData);
            const bytes  = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

            const loaded = await tf.loadLayersModel(tf.io.fromMemory({
                modelTopology: modelData.modelTopology,
                weightSpecs:   modelData.weightSpecs,
                weightData:    bytes.buffer,
            }));
            const allWeights = loaded.getWeights(false); // false = include non-trainable
            const wSum = tf.tidy(() => tf.addN(allWeights.map(w => w.abs().sum())));
            wSum.data().then(d => {
                console.log(`[${variantId}] weight checksum (all layers):`, d[0].toFixed(4), `(${allWeights.length} weight tensors)`);
                wSum.dispose();
            });
            console.log('[tf] backend during load:', tf.getBackend());
            setModels(prev => ({
                ...prev,
                [variantId]: { model: loaded, maps: mapsData, info, status: 'ready' },
            }));
        } catch (e) {
            console.error(`Model load failed (${variantId}):`, e);
            setModels(prev => ({
                ...prev,
                [variantId]: { ...prev[variantId], status: 'error' },
            }));
            setError(e.message);
        }
    }, [models, tfReady]);

    // Load default variant once TF is ready
    useEffect(() => {
        if (tfReady && models.runner.status === 'idle') {
            loadVariant('runner');
        }
    }, [tfReady, loadVariant, models.runner.status]);

    // ── Switch active variant ─────────────────────────────────────────────────
    const handleVariantChange = useCallback((variantId) => {
        setActiveVariant(variantId);
        setPredictions([]);
        setError('');
        loadVariant(variantId); // loadVariant is TF-ready guarded
    }, [loadVariant]);

    // ── Fetch today's race cards ───────────────────────────────────────────────
    // Endpoint: /api/toto-info/v1/cards/today → collection[i]
    useEffect(() => {
        setLoadingCards(true);
        fetch('/api-veikkaus/api/toto-info/v1/cards/today')
            .then(r => r.json())
            .then(d => setCards(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingCards(false));
    }, []);

    // ── Fetch races when card selected ────────────────────────────────────────
    // Endpoint: /api/toto-info/v1/card/{cardId}/races → collection[i]
    // Fields used: collection[i].raceId, .distance, .breed, .startType, .startTime
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

    // ── Run prediction ────────────────────────────────────────────────────────
    const runPrediction = useCallback(async () => {
        if (!tfReady) { console.warn('[tf] not ready yet'); return; }
        const variant    = MODEL_VARIANTS[activeVariant];
        const modelState = models[activeVariant];
        if (!selectedRace || modelState.status !== 'ready') return;

        setLoadingPred(true);
        setError('');
        setPredictions([]);

        try {
            // raceInfo comes from /card/{cardId}/races collection[i]
            const raceInfo = races.find(r => String(r.raceId) === String(selectedRace));
            if (!raceInfo) throw new Error('Race not found in card data');

            // collection[i].raceId → used for runners endpoint
            const raceId = String(raceInfo.raceId);

            // collection[i].distance
            const raceDistance = parseInt(raceInfo.distance || 2100);

            // collection[i].startType: 'CAR_START' → true
            const isCarStart = raceInfo.startType === 'CAR_START';

            // collection[i].breed: 'K' → Finnhorse (cold blood)
            const isColdBlood = detectColdBlood(raceInfo);

            // raceDate: from startTime (epoch ms) or fallback to today.
            // Use local date parts to avoid UTC off-by-one on Finnish evenings.
            const _rd   = raceInfo.startTime ? new Date(raceInfo.startTime) : new Date();
            const raceDate = `${_rd.getFullYear()}-${String(_rd.getMonth()+1).padStart(2,'0')}-${String(_rd.getDate()).padStart(2,'0')}`;

            // Runners: /race/{raceId}/runners → collection[j] or array
            const runnersRaw = await fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`).then(r => r.json());
            const runnersArr = Array.isArray(runnersRaw)
                ? runnersRaw
                : (runnersRaw.collection || runnersRaw.runners || Object.values(runnersRaw));

            console.log('[runners] raw keys:', Object.keys(runnersRaw));
            console.log('[runners] count before filter:', runnersArr.length);
            console.log(runnersRaw+'\n************************')
            // Store raw data for modal
            setModalRunners({ raw: runnersRaw, race: raceInfo, isColdBlood });

            // Normalise runners: maps live API field names → schema field names
            // Specifically handles:
            //   r.startNumber (primary) / r.number (fallback)
            //   r.specialChart (API typo, not r.specialCart)
            //   r.gender: 'ORI'/'TAMMA'/'RUUNA' → 1/2/3
            //   r.prevStarts[k]: shortMeetDate/driver/trackCode/startTrack/
            //                    winOdd/firstprice → schema equivalents
            const runners = normaliseRunners(runnersArr, isCarStart);

            console.log('[runners] after normalise (non-scratched):', runners.length);
            if (runners.length === 0)
                throw new Error(
                    `No valid runners found.\nAPI response keys: [${Object.keys(runnersRaw).join(', ')}]`
                );

            // Build feature tensors using the variant's own mappings
            const { X_hist, X_static, X_mask, metadata } = variant.buildFeatures(
                runners, modelState.maps, raceDate, raceDistance, isColdBlood, isCarStart
            );

            // DATA CHECKSUM FOR DEBUGGING
            if (activeVariant === 'runner') {
                const histT = tf.tensor3d(X_hist);      // [n, 8, 25]
                const staticT = tf.tensor2d(X_static);  // [n, 27]
                const xx = tf.tidy(() => tf.add(histT.abs().sum(), staticT.abs().sum()));
                const xHash = (await xx.data())[0];
                console.log('[X checksum runner]', xHash.toFixed(6), 'shapes:', histT.shape, staticT.shape);
                histT.dispose(); staticT.dispose(); xx.dispose();
            } else {
                const histT = tf.tensor4d(X_hist);      // [1, MAX_RUNNERS, 8, 25]
                const staticT = tf.tensor3d(X_static);  // [1, MAX_RUNNERS, 25]
                const maskT = tf.tensor3d(X_mask);      // [1, MAX_RUNNERS, 1]
                const xx = tf.tidy(() =>
                    tf.addN([histT.abs().sum(), staticT.abs().sum(), maskT.abs().sum()])
                );
                const xHash = (await xx.data())[0];
                console.log('[X checksum race]', xHash.toFixed(6), 'shapes:', histT.shape, staticT.shape, maskT.shape);
                histT.dispose(); staticT.dispose(); maskT.dispose(); xx.dispose();
            }

            // Run inference
            let scores;
            if (activeVariant === 'runner') {
                const histT   = tf.tensor3d(X_hist);
                const staticT = tf.tensor2d(X_static);
                const pred    = modelState.model.predict([histT, staticT]);
                scores        = await pred.data();
                histT.dispose(); staticT.dispose(); pred.dispose();
            } else {
                // X_hist: [1, MAX_RUNNERS, 8, 25]   X_static: [1, MAX_RUNNERS, 25]   X_mask: [1, MAX_RUNNERS, 1]
                const histT   = tf.tensor4d(X_hist);
                const staticT = tf.tensor3d(X_static);
                const maskT   = tf.tensor3d(X_mask);
                const pred    = modelState.model.predict([histT, staticT, maskT]);
                scores        = await pred.data();   // flat: MAX_RUNNERS values
                histT.dispose(); staticT.dispose(); maskT.dispose(); pred.dispose();
            }

            setPredictions(
                variant.extractScores(scores, metadata)
                    .sort((a, b) => b.prob - a.prob)
            );
        } catch (e) {
            console.error(e);
            setError(e.message || 'Unknown error during prediction');
        } finally {
            setLoadingPred(false);
        }
    }, [selectedCard, selectedRace, models, activeVariant, races]);

    // ── Derived ───────────────────────────────────────────────────────────────
    const activeModelState = models[activeVariant];
    const canRun = tfReady && selectedRace && activeModelState.status === 'ready' && !loadingPred;

    const statusColor = { idle: '#888', loading: '#f0a500', ready: '#2ecc71', error: '#e74c3c' };
    const statusLabel = { idle: 'Idle', loading: 'Loading…', ready: 'Model ready', error: 'Load error' };

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
                        TotoModels v2
                    </h1>

                    {/* ── Model selector ── */}
                    <div style={{ marginTop: 14, display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
                        {Object.values(MODEL_VARIANTS).map(v => {
                            const vState   = models[v.id];
                            const isActive = activeVariant === v.id;
                            return (
                                <label key={v.id} style={{ display: 'flex', alignItems: 'flex-start', gap: 8,
                                    cursor: 'pointer', opacity: isActive ? 1 : 0.5, transition: 'opacity 0.2s' }}>
                                    <input type="checkbox" checked={isActive}
                                           onChange={() => handleVariantChange(v.id)}
                                           style={{ marginTop: 2, accentColor: '#4a90d9' }} />
                                    <div>
                                        <div style={{ fontSize: 13, fontWeight: 600, color: isActive ? '#e8eaf0' : '#a3a3d8' }}>
                                            {v.label}
                                            {vState.status === 'loading' && <span style={{ color: '#f0a500', marginLeft: 8, fontSize: 11 }}>loading…</span>}
                                            {vState.status === 'error'   && <span style={{ color: '#e74c3c', marginLeft: 8, fontSize: 11 }}>load error</span>}
                                        </div>
                                        <div style={{ fontSize: 10, color: '#6a6a9e', letterSpacing: 0.3, marginTop: 1 }}>
                                            {v.description}
                                        </div>
                                    </div>
                                </label>
                            );
                        })}
                    </div>

                    {/* ── Model info ── */}
                    <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', fontSize: 12 }}>
                        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
                                background: statusColor[activeModelState.status],
                                boxShadow: `0 0 6px ${statusColor[activeModelState.status]}` }} />
                            <span style={{ color: statusColor[activeModelState.status] }}>
                                {statusLabel[activeModelState.status]}
                            </span>
                        </span>
                        {activeModelState.info && (() => {
                            const info = activeModelState.info;
                            return (
                                <span style={{ color: '#556', fontSize: 11, display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                                    <span>Epoch <b style={{ color: '#aaa' }}>{info.epoch}</b></span>
                                    <span>val_loss <b style={{ color: '#aaa' }}>{info.val_loss}</b></span>
                                    <span>val_acc <b style={{ color: '#aaa' }}>{info.val_acc != null ? (info.val_acc * 100).toFixed(1) + '%' : '—'}</b></span>
                                    <span>val_auc <b style={{ color: '#aaa' }}>{info.val_auc != null ? (info.val_auc * 100).toFixed(1) + '%' : '—'}</b></span>
                                    <span style={{ borderLeft: '1px solid #222', paddingLeft: 14 }}>
                                        Data <b style={{ color: '#aaa' }}>{info.dataStartDate} → {info.dataEndDate}</b>
                                    </span>
                                    <span>races <b style={{ color: '#aaa' }}>{info.totalRaces?.toLocaleString('fi-FI')}</b></span>
                                    <span style={{ color: '#333' }}>·</span>
                                    <span>runners <b style={{ color: '#aaa' }}>{info.totalRunners?.toLocaleString('fi-FI')}</b></span>
                                </span>
                            );
                        })()}
                    </div>
                </div>

                {/* ── Learning curves ── */}
                <LearningCurves
                    info={activeModelState.info}
                    variant={activeVariant}
                />

                {/* ── Error box ── */}
                {error && (
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
                                disabled={!selectedCard || loadingRaces} style={{ ...selectStyle, minWidth: 180 }}>
                            <option value="">{loadingRaces ? '…' : '— Select race —'}</option>
                            {races.map(r => (
                                // value is raceId (not race number) — used to find raceInfo later
                                <option key={r.raceId} value={r.raceId}>
                                    Race {r.number} · {r.distance}m · {r.startType === 'CAR_START' ? 'Auto' : 'Voltti'}
                                </option>
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
                            const race = races.find(r => String(r.raceId) === String(selectedRace));
                            if (!race) return;
                            setModalRaceId(String(race.raceId));
                            setModalRaceLabel(`Race ${race.number} · ${race.distance}m`);
                            setModalRace(race);
                            setModalOpen(true);
                        }} style={{
                            padding: '10px 20px', background: '#0f1520', color: '#4a90d9',
                            border: '1px solid #2a4060', borderRadius: 4,
                            fontFamily: 'inherit', fontSize: 13, letterSpacing: 1,
                            cursor: 'pointer',
                        }}>
                            ⊞ Race details
                        </button>
                    )}
                </div>

                {/* ── Results ── */}
                {predictions.length > 0 && (
                    <div>
                        <div style={{ fontSize: 11, letterSpacing: 2, color: '#4a90d9',
                            textTransform: 'uppercase', marginBottom: 12 }}>
                            Prediction · {MODEL_VARIANTS[activeVariant].label}
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
                                    <tr key={`${p.number}-${i}`} style={{ borderBottom: '1px solid #12151c',
                                        background: i === 0 ? '#0d1a2a' : i % 2 === 0 ? '#0f1118' : 'transparent' }}>
                                        <td style={{ padding: '10px 12px', color: i < 3 ? '#f0a500' : '#444' }}>
                                            {i < 3
                                                ? <span style={{ fontSize: 26, lineHeight: 1 }}>{['①','②','③'][i]}</span>
                                                : `#${i + 1}`}
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
                                                color:      p.prob > 0.5 ? '#2ecc71' : '#444',
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
