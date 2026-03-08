import React, { useState, useEffect, useCallback } from 'react';
import RunnerModal from './RunnerModal.jsx';
import * as tf from '@tensorflow/tfjs';
import { registerMultiHeadAttention } from './MultiHeadAttention.js';
import LearningCurves from './LearningCurves.jsx';
import {
    sanitize, detectColdBlood, normaliseRunners,
    buildRunnerBasedFeatures, buildRaceBasedFeatures,
    fetchJSON, MAX_RUNNERS, filterValidPrev,
} from './util.js';

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
        extractScores: (scores, metadata) =>
            metadata.map(m => ({ ...m, prob: scores[m.slot] })),
    },
    mixed: {
        id:           'mixed',
        label:        'Mixed (Runner + Race)',
        description:  'Runner encoder + Race attention  ·  tensor: [1, 18, 8, 25] + mask',
        modelPath:    '/model-mixed/model.json',
        mappingsPath: '/mappings_mixed.json',
        buildFeatures: buildRaceBasedFeatures,
        extractScores: (scores, metadata) =>
            metadata.map(m => ({ ...m, prob: scores[m.slot] })),
    },
};

// ── Helper: Build simple pseudocode summary for a model variant ─────────────
function buildModelPseudocode(variantId, info) {
    const mono = {
        fontFamily: "'IBM Plex Mono','Courier New',monospace",
        whiteSpace: 'pre',
        color: '#76eaa3',
        fontSize: 11,
        lineHeight: 1.35,
    };
    const box = {
        position: 'absolute',
        top: '100%',
        left: 22,
        marginTop: 8,
        background: '#0d0f14',
        border: '1px solid #1e2330',
        boxShadow: '0 6px 24px rgba(0,0,0,0.35)',
        borderRadius: 6,
        padding: '10px 12px',
        color: '#e8eaf0',
        zIndex: 20,
        minWidth: 300,
        maxWidth: 520,
    };

    const dims = (k, d) => (info && info[k] != null ? info[k] : d);

    let lines = [];
    if (variantId === 'mixed') {
        const attnHeads = dims('attnHeads', 12);
        const embedDim  = dims('embedDim', 96);
        const ffnDim    = dims('ffnDim', 192);
        const rL2       = dims('runnerLstm2Units', 48);
        const rProj     = dims('runnerProjDim', 48);
        const outUnits  = dims('outUnits', 24);
        lines = [
            'MixedRunnerRace(inputs: history[R,T,HF], static[R,SF], mask[R,1]) {',
            `  // Runner history encoder (per runner)`,
            `  h = TD(Masking(-1) → LSTM(64, seq) → LSTM(${rL2}) → Dense(${rProj}, relu) → Dropout(0.1))`,
            `  s = TD(Dense(48, relu) → BN/LN → Dense(32, relu) → Dropout(0.2))`,
            `  x = TD(Dense(${embedDim}, relu) → BN/LN)(Concat(h, s))`,
            `  // Self-attention over runners`,
            `  a = MHA(heads=${attnHeads}, dim=${embedDim})(LN(x))`,
            `  y = LN(TD(Dense(${embedDim}, linear))(Concat(x, a)))`,
            `  y = LN(Add(y, TD(Dense(${ffnDim}, relu) → Dropout → Dense(${embedDim}, linear))(y)))`,
            `  // Per runner output`,
            `  z = TD(Dropout(0.25) → Dense(${outUnits}, relu) → Dense(1, sigmoid))`,
            `  return z ⊙ mask`,
            '}',
            '',
            '// Loss: BCE + aux SoftTopK(K=3, w≈0.5) | Selection: val_ndcg@3',
        ];
    } else if (variantId === 'race') {
        const attnHeads = dims('attnHeads', 8);
        const embedDim  = dims('embedDim', 64);
        const ffnDim    = dims('ffnDim', 128);
        lines = [
            'RaceModel(inputs: history[R,T,HF], static[R,SF], mask[R,1]) {',
            '  x = TD(LSTM(64, seq) → LSTM(32) → Dense(32, relu))',
            `  x = TD(Dense(${embedDim}, relu) → BN/LN)(Concat(x, TD(Dense(32, relu))(static)))`,
            `  a = MHA(heads=${attnHeads}, dim=${embedDim})(LN(x))`,
            `  y = LN(TD(Dense(${embedDim}, linear))(Concat(x, a)))`,
            `  y = LN(Add(y, TD(Dense(${ffnDim}, relu) → Dropout → Dense(${embedDim}, linear))(y)))`,
            '  z = TD(Dense(1, sigmoid))',
            '  return z ⊙ mask',
            '}',
            '',
            '// Loss: BCE (+ optional SoftTopK) | Selection: val_ndcg@3',
        ];
    } else if (variantId === 'runner') {
        lines = [
            'RunnerModel(inputs: runnerHistory[T,HF], runnerStatic[SF]) {',
            '  h = LSTM(64, seq) → LSTM(32) → Dense(32, relu) → Dropout',
            '  s = Dense(48, relu) → BN/LN → Dense(32, relu) → Dropout',
            '  x = Dense(64, relu)(Concat(h, s))',
            '  y = Dense(24, relu) → Dense(1, sigmoid)',
            '  return y',
            '}',
            '',
            '// Loss: BCE | Metrics: AUC/AP',
        ];
    }

    return { box, mono, text: lines.join('\n') };
}

export default function App() {
    const [hoveredVariant, setHoveredVariant] = React.useState(null);
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

    const [histStatus, setHistStatus] = useState({ loading: false, status: null, withHist: 0, total: 0, error: null });

    const [models, setModels] = useState({
        runner: { model: null, maps: null, info: null, status: 'idle' },
        race:   { model: null, maps: null, info: null, status: 'idle' },
        mixed:  { model: null, maps: null, info: null, status: 'idle' },
    });

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
                registerMultiHeadAttention();
                setTfReady(true);
            } catch (e) {
                console.error('[tf] init failed:', e);
            }
        })();
        return () => { cancelled = true; };
    }, []);

    const loadVariant = useCallback(async (variantId) => {
        if (!tfReady) return;
        if (['ready', 'loading'].includes(models[variantId].status)) return;

        const variant = MODEL_VARIANTS[variantId];
        setModels(prev => ({ ...prev, [variantId]: { ...prev[variantId], status: 'loading' } }));

        try {
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
            const allWeights = loaded.getWeights(false);
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

    useEffect(() => {
        if (tfReady && models.runner.status === 'idle') {
            loadVariant('runner');
        }
    }, [tfReady, loadVariant, models.runner.status]);

    const handleVariantChange = useCallback((variantId) => {
        setActiveVariant(variantId);
        setPredictions([]);
        setError('');
        loadVariant(variantId);
    }, [loadVariant]);

    useEffect(() => {
        setLoadingCards(true);
        fetch('/api-veikkaus/api/toto-info/v1/cards/today')
            .then(r => r.json())
            .then(d => setCards(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingCards(false));
    }, []);

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

    useEffect(() => {
        setHistStatus({ loading: false, status: null, withHist: 0, total: 0, error: null });
        if (!selectedRace) return;
        const raceInfo = races.find(r => String(r.raceId) === String(selectedRace));
        if (!raceInfo) return;
        const raceId = String(raceInfo.raceId);
        const isCarStart = raceInfo.startType === 'CAR_START';
        let cancelled = false;
        (async () => {
            try {
                setHistStatus(prev => ({ ...prev, loading: true, error: null }));
                const runnersRaw = await fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`).then(r => r.json());
                const runnersArr = Array.isArray(runnersRaw)
                    ? runnersRaw
                    : (runnersRaw.collection || runnersRaw.runners || Object.values(runnersRaw));
                const runners = normaliseRunners(runnersArr, isCarStart);
                const starters = runners.filter(r => !r.scratched);
                const total = starters.length;
                let withHist = 0;
                for (const r of starters) {
                    const validPrev = filterValidPrev(r.prevStarts);
                    if (validPrev.length > 0) withHist++;
                }
                const status = total === 0 ? null : (withHist === 0 ? 'none' : (withHist < total ? 'partial' : 'complete'));
                if (!cancelled) setHistStatus({ loading: false, status, withHist, total, error: null });
            } catch (e) {
                console.error('[history-check] failed', e);
                if (!cancelled) setHistStatus({ loading: false, status: null, withHist: 0, total: 0, error: String(e) });
            }
        })();
        return () => { cancelled = true; };
    }, [selectedRace, races]);

    const runPrediction = useCallback(async () => {
        if (!tfReady) { console.warn('[tf] not ready yet'); return; }
        const variant    = MODEL_VARIANTS[activeVariant];
        const modelState = models[activeVariant];
        if (!selectedRace || modelState.status !== 'ready') return;

        setLoadingPred(true);
        setError('');
        setPredictions([]);

        try {
            const raceInfo = races.find(r => String(r.raceId) === String(selectedRace));
            if (!raceInfo) throw new Error('Race not found in card data');

            const raceId = String(raceInfo.raceId);
            const raceDistance = parseInt(raceInfo.distance || 2100);
            const isCarStart = raceInfo.startType === 'CAR_START';
            const isColdBlood = detectColdBlood(raceInfo);

            const _rd   = raceInfo.startTime ? new Date(raceInfo.startTime) : new Date();
            const raceDate = `${_rd.getFullYear()}-${String(_rd.getMonth()+1).padStart(2,'0')}-${String(_rd.getDate()).padStart(2,'0')}`;

            const runnersRaw = await fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`).then(r => r.json());
            const runnersArr = Array.isArray(runnersRaw)
                ? runnersRaw
                : (runnersRaw.collection || runnersRaw.runners || Object.values(runnersRaw));

            console.log('[runners] raw keys:', Object.keys(runnersRaw));
            console.log('[runners] count before filter:', runnersArr.length);
            console.log(runnersRaw+'\n************************')
            setModalRunners({ raw: runnersRaw, race: raceInfo, isColdBlood });

            const runners = normaliseRunners(runnersArr, isCarStart);

            console.log('[runners] after normalise (non-scratched):', runners.length);
            if (runners.length === 0)
                throw new Error(
                    `No valid runners found.\nAPI response keys: [${Object.keys(runnersRaw).join(', ')}]`
                );

            const { X_hist, X_static, X_mask, metadata } = variant.buildFeatures(
                runners, modelState.maps, raceDate, raceDistance, isColdBlood, isCarStart
            );

            if (activeVariant === 'runner') {
                const histT = tf.tensor3d(X_hist);
                const staticT = tf.tensor2d(X_static);
                const xx = tf.tidy(() => tf.add(histT.abs().sum(), staticT.abs().sum()));
                const xHash = (await xx.data())[0];
                console.log('[X checksum runner]', xHash.toFixed(6), 'shapes:', histT.shape, staticT.shape);
                histT.dispose(); staticT.dispose(); xx.dispose();
            } else {
                const histT = tf.tensor4d(X_hist);
                const staticT = tf.tensor3d(X_static);
                const maskT = tf.tensor3d(X_mask);
                const xx = tf.tidy(() =>
                    tf.addN([histT.abs().sum(), staticT.abs().sum(), maskT.abs().sum()])
                );
                const xHash = (await xx.data())[0];
                console.log('[X checksum race]', xHash.toFixed(6), 'shapes:', histT.shape, staticT.shape, maskT.shape);
                histT.dispose(); staticT.dispose(); maskT.dispose(); xx.dispose();
            }

            let scores;
            if (activeVariant === 'runner') {
                const histT   = tf.tensor3d(X_hist);
                const staticT = tf.tensor2d(X_static);
                const pred    = modelState.model.predict([histT, staticT]);
                scores        = Array.from(await pred.data());
                histT.dispose(); staticT.dispose(); pred.dispose();
            } else {
                const histT   = tf.tensor4d(X_hist);
                const staticT = tf.tensor3d(X_static);
                const maskT   = tf.tensor3d(X_mask);
                const pred    = modelState.model.predict([histT, staticT, maskT]);
                scores        = Array.from(await pred.data());
                histT.dispose(); staticT.dispose(); maskT.dispose(); pred.dispose();
            }

            // Optional temperature calibration (mixed and race models)
            const info = modelState.info || {};
            if ((activeVariant === 'mixed' || activeVariant === 'race') && info.calibration && info.calibration.type === 'temperature' && typeof info.calibration.T === 'number') {
                const T = info.calibration.T;
                const eps = 1e-7;
                for (let i=0;i<scores.length;i++){
                    const p = Math.min(1-eps, Math.max(eps, scores[i]));
                    const logit = Math.log(p/(1-p));
                    const pCal = 1/(1+Math.exp(-(logit/Math.max(0.2, Math.min(5.0, T)))));
                    scores[i] = pCal;
                }
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
                    <div style={{ display: 'flex', flexDirection:'row', justifyContent: 'space-between' }}>
                        <div style={{ fontSize: 11, letterSpacing: 4, color: '#4a90d9', textTransform: 'uppercase', marginBottom: 6 }}>
                            Toto Prediction System
                        </div>
                        <a href="https://github.com/juhaj77/toto-prediction-model"
                           target="_blank" rel="noopener noreferrer"
                           style={{
                               display: 'flex', alignItems: 'center', gap: 6,
                               color: '#4a90d9', fontFamily: "'IBM Plex Mono','Courier New',monospace",
                               fontSize: 11, letterSpacing: 1, textDecoration: 'none',
                               opacity: 0.7, transition: 'opacity 0.2s',
                           }}
                           onMouseEnter={e => e.currentTarget.style.opacity = 1}
                           onMouseLeave={e => e.currentTarget.style.opacity = 0.7}
                        >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                            </svg>
                            GitHub
                        </a>
                    </div>
                    <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: '#fff', letterSpacing: -0.5 }}>
                        TotoModels v2
                    </h1>

                    <div style={{ marginTop: 14, display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap' }}>
                        {Object.values(MODEL_VARIANTS).map(v => {
                            const vState   = models[v.id];
                            const isActive = activeVariant === v.id;
                            return (
                                <label
                                    key={v.id}
                                    tabIndex={0}
                                    onMouseEnter={() => setHoveredVariant(v.id)}
                                    onMouseLeave={() => setHoveredVariant(null)}
                                    onFocus={() => setHoveredVariant(v.id)}
                                    onBlur={() => setHoveredVariant(null)}
                                    style={{
                                        display: 'flex', alignItems: 'flex-start', gap: 8,
                                        cursor: 'pointer', opacity: isActive ? 1 : 0.5, transition: 'opacity 0.2s',
                                        position: 'relative'
                                    }}>
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
                                        { /*hoveredVariant === v.id && (() => {
                                            const pcs = buildModelPseudocode(v.id, vState.info);
                                            return (
                                                <div style={pcs.box} role="tooltip" aria-label={`${v.label} model pseudocode`}>
                                                    <div style={{ color: '#4a90d9', fontSize: 10, letterSpacing: 1, marginBottom: 6, textTransform: 'uppercase' }}>
                                                        Model summary
                                                    </div>
                                                    <div style={pcs.mono}>{pcs.text}</div>
                                                </div>
                                            );
                                        })()*/}
                                    </div>
                                </label>
                            );
                        })}
                    </div>

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
                                    <span>val_ndcg3 <b style={{ color: '#aaa' }}>{info.val_ndcg3 != null ? Number(info.val_ndcg3 * 100).toFixed(1) + '%' : '—'}</b></span>
                                    <span>val_hit1 <b style={{ color: '#aaa' }}>{info.val_hit1 != null ? (info.val_hit1 * 100).toFixed(1) + '%' : '—'}</b></span>
                                    <span>val_auc <b style={{ color: '#e87979' }}>{info.val_auc != null ? (info.val_auc * 100).toFixed(1) + '%' : '—'}</b></span>
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

                    {/* ── Review input data nappi — ENNEN history-varoitusta ── */}
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
                            ⊞ Review input data
                        </button>
                    )}


                </div>
                {/* ── History warning — omalla rivillään flexBasis:100% ── */}
                {selectedRace && (
                    histStatus.loading || histStatus.status === 'none' || histStatus.status === 'partial'
                ) && (
                    <div style={{marginTop: 6, marginBottom: '2em', width: 'fit-content', fontSize: 12, display: 'block', alignItems: 'center', gap: 8 }}>
                        {(() => {
                            let color = '#888';
                            let bg    = '#0f1118';
                            let border= '#1e2330';
                            let text  = 'Checking history data…';
                            if (!histStatus.loading) {
                                if (histStatus.status === 'none') {
                                    color = '#e74c3c'; bg = '#1a0a0a'; border = '#4a1e1e';
                                    text = 'No history data found for this race. Predictions may be unreliable.';
                                } else if (histStatus.status === 'partial') {
                                    color = '#f0a500'; bg = '#1a1505'; border = '#3a2e10';
                                    text = `History available for ${histStatus.withHist}/${histStatus.total} runners. Predictions may be less reliable.`;
                                }
                            }
                            return (
                                <div title="Model performance improves with more history per runner."
                                     style={{ padding: '6px 10px', background: bg, border: `1px solid ${border}`, borderRadius: 4, color }}>
                                    {text}
                                </div>
                            );
                        })()}
                    </div>
                )}
                {/* ── Results ── */}
                {predictions.length > 0 && (
                    <div>
                        <div style={{ display: 'flex', flexDirection:'row', gap:'2em',alignItems: 'center', marginBottom: 12 }}>
                            <div style={{ fontSize: 11, letterSpacing: 2, color: '#4a90d9',
                                textTransform: 'uppercase' }}>
                                Prediction · {MODEL_VARIANTS[activeVariant].label}
                            </div>
                            <span style={{fontSize: '11px', color: '#2e5883'}}>The probability refers to the likelihood that the competitor finishes in the top three.</span>
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
                                        <td style={{ padding: '10px 12px', color: '#666', fontWeight: 'bold', fontSize: '1.1em' }}>{p.number}</td>
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