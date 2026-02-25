// ═══════════════════════════════════════════════════════════════════════════════
// TOTO PREDICTION MODEL — model.js
//
// Reads:  ravit_opetusdata.json  (training)
// Writes: ravimalli-mixed/model_full.json  (trained model weights)
//         mappings.json                    (name → integer ID maps)
//
// Architecture: Mixed LSTM (history branch) + Dense (static branch)
// Target:       binary — top-3 finish (position 1–3) vs. not
// ═══════════════════════════════════════════════════════════════════════════════

'use strict';

const tf = require('@tensorflow/tfjs');
const fs = require('fs');

const TRAINING_DATA  = './ravit_opetusdata.json';
const PREDICTION_DATA = '';
const MAPPINGS_FILE  = './mappings.json';
const MODEL_FOLDER   = './ravimalli-mixed';
const MAX_HISTORY    = 8;

// ─── MODEL PERSISTENCE ────────────────────────────────────────────────────────

async function saveModel(model, meta = {}) {
    if (!fs.existsSync(MODEL_FOLDER)) fs.mkdirSync(MODEL_FOLDER);
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        const payload = {
            modelTopology: artifacts.modelTopology,
            weightSpecs:   artifacts.weightSpecs,
            weightData:    Buffer.from(artifacts.weightData).toString('base64'),
            trainingInfo: {
                savedAt:       new Date().toISOString(),
                epoch:         meta.epoch         ?? null,
                loss:          meta.loss          ?? null,
                val_loss:      meta.val_loss       ?? null,
                val_acc:       meta.val_acc        ?? null,
                learningRate:  meta.learningRate   ?? null,
                dataStartDate: meta.dataStartDate  ?? null,
                dataEndDate:   meta.dataEndDate    ?? null,
                totalRaces:    meta.totalRaces     ?? null,
                totalRunners:  meta.totalRunners   ?? null,
            },
        };
        fs.writeFileSync(`${MODEL_FOLDER}/model_full.json`, JSON.stringify(payload));
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
}

async function loadModel() {
    const filePath = `${MODEL_FOLDER}/model_full.json`;
    if (!fs.existsSync(filePath)) return null;
    const saved  = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const buffer = Buffer.from(saved.weightData, 'base64');
    return tf.loadLayersModel(tf.io.fromMemory({
        modelTopology: saved.modelTopology,
        weightSpecs:   saved.weightSpecs,
        weightData:    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
    }));
}

// ─── FEATURE HELPERS ──────────────────────────────────────────────────────────

// Map track condition string → 0–1 scale (0 = heaviest, 1 = lightest)
function encodeTrackCondition(condition) {
    switch ((String(condition ?? '')).toLowerCase().trim()) {
        case 'heavy track':       return 0.00;
        case 'quite heavy track': return 0.25;
        case 'winter track':      return 0.75;
        case 'light track':       return 1.00;
        default:                  return 0.50;
    }
}

// Use surname only so "J Mäkinen" and "Juhani Mäkinen" map to the same ID
function extractSurname(fullName) {
    if (!fullName) return 'unknown';
    return fullName.trim().toLowerCase().split(' ').at(-1);
}

function parseDate(str) {
    if (!str) return null;
    if (str.includes('-')) return new Date(str);
    if (str.includes('.')) {
        const [d, m, y] = str.split('.');
        const year = parseInt(y) < 100 ? parseInt(y) + 2000 : parseInt(y);
        return new Date(year, parseInt(m) - 1, parseInt(d));
    }
    return null;
}

// Normalise km time to a 2100 m base distance.
// Correlation with finishing position: r = 0.031 → 0.090 after normalisation.
function normaliseKmTime(km, distance) {
    if (km === null || km === undefined || isNaN(km) || km <= 0) return null;
    const dist = (distance !== null && distance !== undefined && distance > 0) ? distance : 2100;
    return km + (2100 - dist) / 2000;
}

// ─── DATA LOADING ─────────────────────────────────────────────────────────────

function loadData(filePath, isTraining = true) {
    // Name → integer ID maps. Persisted to mappings.json so prediction uses
    // the same IDs as training.
    let maps = {
        coaches: {}, drivers: {}, tracks: {},
        counts: { c: 1, d: 1, t: 1 },
    };

    if (!isTraining && fs.existsSync(MAPPINGS_FILE)) {
        maps = JSON.parse(fs.readFileSync(MAPPINGS_FILE, 'utf8'));
        console.log(
            `  Loaded mappings: ${Object.keys(maps.coaches).length} coaches, ` +
            `${Object.keys(maps.drivers).length} drivers, ` +
            `${Object.keys(maps.tracks).length} tracks`
        );
    }

    // Assign a stable integer ID for each unique name.
    // In prediction mode unknown names fall back to 0.
    function getID(map, name, type) {
        if (!name || name === 'Unknown' || name === '') return 0;
        const key = (type === 'driver') ? extractSurname(name) : name.trim().toLowerCase();
        if (isTraining) {
            if (!map[key]) { map[key] = maps.counts[type[0]]; maps.counts[type[0]]++; }
            return map[key];
        }
        return map[key] || 0;
    }

    // ── Parse JSON ────────────────────────────────────────────────────────────
    const raw   = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const races = raw.races || [];

    // ── Per-breed mean imputation values ─────────────────────────────────────
    // Used when a runner's record or a historical km time is missing.
    const breedStats = { SH: { records: [], kmTimes: [] }, LV: { records: [], kmTimes: [] } };
    for (const race of races) {
        const breed = race.isColdBlood ? 'SH' : 'LV';
        for (const runner of (race.runners || [])) {
            if (runner.record > 0) breedStats[breed].records.push(runner.record);
            for (const ps of (runner.prevStarts || [])) {
                const km = normaliseKmTime(ps.kmTime, ps.distance);
                if (km > 0) breedStats[breed].kmTimes.push(km);
            }
        }
    }
    const avg = arr => arr.length ? arr.reduce((a, b) => a + b) / arr.length : null;
    const means = {
        SH: { record: avg(breedStats.SH.records) ?? 28.0, km: avg(breedStats.SH.kmTimes) ?? 29.0 },
        LV: { record: avg(breedStats.LV.records) ?? 15.0, km: avg(breedStats.LV.kmTimes) ?? 16.0 },
    };

    // ── Per-race betting/win ranking maps ─────────────────────────────────────
    // Relative rank within a race is more informative than raw percentage.
    const betRankMap  = {};   // raceId_startNumber → rank (1 = favourite)
    const winRankMap  = {};
    const rankMetaMap = {};   // → { betKnown, winKnown }

    for (const race of races) {
        const starters = (race.runners || []).filter(r => !r.scratched);
        const hasBet   = starters.some(r => (r.bettingPercentage || 0) > 0);
        const hasWin   = starters.some(r => (r.winPercentage     || 0) > 0);
        const neutral  = (starters.length + 1) / 2;

        const byBet = [...starters].sort((a, b) => (b.bettingPercentage || 0) - (a.bettingPercentage || 0));
        const byWin = [...starters].sort((a, b) => (b.winPercentage     || 0) - (a.winPercentage     || 0));

        for (const runner of starters) {
            const key      = `${race.trackID}_${runner.number}`;
            const betKnown = hasBet && (runner.bettingPercentage || 0) > 0 ? 1 : 0;
            const winKnown = hasWin && (runner.winPercentage     || 0) > 0 ? 1 : 0;
            betRankMap[key]  = betKnown ? byBet.findIndex(r => r.number === runner.number) + 1 : neutral;
            winRankMap[key]  = winKnown ? byWin.findIndex(r => r.number === runner.number) + 1 : neutral;
            rankMetaMap[key] = { betKnown, winKnown };
        }
    }

    // ── Build feature tensors ─────────────────────────────────────────────────
    const X_hist   = [];
    const X_static = [];
    const Y        = [];
    const metadata = [];

    for (const race of races) {
        const breed     = race.isColdBlood ? 'SH' : 'LV';
        const starters  = (race.runners || []).filter(r => !r.scratched);
        const neutral   = (starters.length + 1) / 2;

        for (const runner of starters) {
            const rankKey  = `${race.trackID}_${runner.number}`;
            const betRank  = betRankMap[rankKey]  ?? neutral;
            const winRank  = winRankMap[rankKey]  ?? neutral;
            const rankMeta = rankMetaMap[rankKey] ?? { betKnown: 0, winKnown: 0 };

            // ── Shoe encoding ─────────────────────────────────────────────────
            const frontStr    = (runner.frontShoes  || '').toUpperCase();
            const rearStr     = (runner.rearShoes   || '').toUpperCase();
            const frontActive = frontStr === 'HAS_SHOES' ? 1 : 0;
            const frontKnown  = (frontStr === 'HAS_SHOES' || frontStr === 'NO_SHOES') ? 1 : 0;
            const rearActive  = rearStr  === 'HAS_SHOES' ? 1 : 0;
            const rearKnown   = (rearStr  === 'HAS_SHOES' || rearStr  === 'NO_SHOES') ? 1 : 0;

            // ── Special cart encoding ─────────────────────────────────────────
            const cartStr    = (runner.specialCart || '').toUpperCase();
            const cartActive = cartStr === 'YES' ? 1 : 0;
            const cartKnown  = (cartStr === 'YES' || cartStr === 'NO') ? 1 : 0;

            // ── prevIndexNorm ─────────────────────────────────────────────────
            // Weighted podium score normalised by ALL starts, including
            // disqualifications (pos=20) and DNFs (pos=21) which score 0 pts.
            // This penalises horses with many non-finishes. r = 0.161 vs top-3.
            const prevStarts  = runner.prevStarts || [];
            const validPrev   = prevStarts.filter(ps => ps.date && ps.driver);
            const histKnown   = validPrev.length > 0 ? 1 : 0;
            let prevIndexNorm = 0;

            if (validPrev.length > 0) {
                let score = 0, count = 0;
                for (const ps of validPrev) {
                    const pos = ps.position;
                    if (pos == null) continue;        // null (MATLAB []) or undefined → skip
                    count++;                          // disq/DNF → 0 pts, still counted
                    if      (pos === 1) score += 1.00;
                    else if (pos === 2) score += 0.50;
                    else if (pos === 3) score += 0.33;
                }
                prevIndexNorm = count > 0 ? score / count : 0;
            }

            // ── Static features — 27 total ────────────────────────────────────
            const staticFeats = [
                (runner.number || 1) / 20,                                        // [0]  start number
                getID(maps.coaches, runner.coach,  'coach')  / 2000,              // [1]  coach ID
                (runner.record || means[breed].record) / 50,                      // [2]  race record (imputed if missing)
                getID(maps.drivers, runner.driver, 'driver') / 3000,              // [3]  current driver ID
                (runner.age || 5) / 15,                                           // [4]  age
                (runner.gender || 2) / 3,                                         // [5]  gender (1=mare 2=gelding 3=stallion)
                race.isColdBlood ? 1 : 0,                                         // [6]  cold blood breed
                frontActive, frontKnown,                                          // [7-8]   front shoes on / known
                rearActive,  rearKnown,                                           // [9-10]  rear shoes on / known
                runner.frontShoesChanged ? 1 : 0,                                 // [11] front shoes changed this race
                runner.rearShoesChanged  ? 1 : 0,                                 // [12] rear shoes changed this race
                (race.distance || 2100) / 3100,                                   // [13] race distance
                race.isCarStart ? 1 : 0,                                          // [14] car start (vs. line start)
                (runner.bettingPercentage || 0) / 100,                            // [15] raw betting %
                (runner.winPercentage     || 0) / 100,                            // [16] historical win %
                (runner.winPercentage     || 0) > 0 ? 1 : 0,                      // [17] win % known flag
                runner.isAutoRecord ? 1 : 0,                                      // [18] record set from car start
                cartActive, cartKnown,                                            // [19-20] special cart on / known
                betRank / 20, rankMeta.betKnown,                                  // [21-22] betting rank / known
                winRank / 20, rankMeta.winKnown,                                  // [23-24] win rank / known
                prevIndexNorm,                                                    // [25] podium index / n_all_starts
                histKnown,                                                        // [26] has prior start history
            ];

            // ── History sequence — MAX_HISTORY × 25 features ─────────────────
            const histSeq = [];

            for (let i = 0; i < MAX_HISTORY; i++) {
                const ps = validPrev[i];

                // Pad missing history slots with -1 (LSTM masking layer ignores these)
                if (!ps) { histSeq.push(new Array(25).fill(-1)); continue; }

                const raceDate  = parseDate(race.date);
                const startDate = parseDate(ps.date);
                const daysSince = (raceDate && startDate)
                    ? Math.min(365, (raceDate - startDate) / 86400000)
                    : 30;

                const kmNorm  = normaliseKmTime(ps.kmTime, ps.distance);
                const kmKnown = kmNorm > 0 ? 1 : 0;
                const kmFinal = kmKnown ? kmNorm / 100 : means[breed].km / 100;

                const distKnown  = (ps.distance   || 0) > 0 ? 1 : 0;
                const distFinal  = distKnown  ? (ps.distance   / 3100) : 0.67;

                const prizeKnown = (ps.firstPrice || 0) > 0 ? 1 : 0;
                const prizeFinal = prizeKnown ? Math.log1p(ps.firstPrice) / 10  : 0.55;

                const oddKnown   = (ps.odd        || 0) > 0 ? 1 : 0;
                const oddFinal   = oddKnown   ? Math.log1p(ps.odd)        / 5   : 0.50;

                const posKnown   = ps.position != null && ps.position > 0 ? 1 : 0;
                const posFinal   = posKnown ? ps.position / 20 : 0.5;

                const psfront = (ps.frontShoes  || '').toUpperCase();
                const psrear  = (ps.rearShoes   || '').toUpperCase();
                const pscart  = (ps.specialCart || '').toUpperCase();

                histSeq.push([
                    kmFinal,    kmKnown,                                           // [0-1]   km time normalised to 2100 m
                    distFinal,  distKnown,                                         // [2-3]   distance
                    daysSince / 365,                                               // [4]     days since this start
                    posFinal,   posKnown,                                          // [5-6]   finishing position
                    prizeFinal, prizeKnown,                                        // [7-8]   prize money (log-scaled)
                    oddFinal,   oddKnown,                                          // [9-10]  win odds (log-scaled)
                    ps.isCarStart   ? 1 : 0,                                       // [11]    car start
                    ps.break        ? 1 : 0,                                       // [12]    gait fault (break)
                    (ps.number ?? 1) / 30,                                         // [13]    start position
                    getID(maps.drivers, ps.driver, 'driver') / 3000,               // [14]    driver ID
                    getID(maps.tracks,  ps.track,  'track')  / 500,                // [15]    track ID
                    ps.disqualified ? 1 : 0,                                       // [16]    disqualified
                    ps.DNF          ? 1 : 0,                                       // [17]    did not finish
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

            if (isTraining) {
                Y.push((runner.position != null && runner.position >= 1 && runner.position <= 3) ? 1 : 0);
            } else {
                metadata.push({
                    number: runner.number,
                    name:   runner.name,
                    driver: runner.driver,
                });
            }
        }
    }

    // Persist maps after training so prediction uses the same IDs
    if (isTraining) {
        fs.writeFileSync(MAPPINGS_FILE, JSON.stringify(maps, null, 2));
        console.log(
            `  Mappings saved: ${Object.keys(maps.coaches).length} coaches, ` +
            `${Object.keys(maps.drivers).length} drivers, ` +
            `${Object.keys(maps.tracks).length} tracks`
        );
    }

    const allDates = races.map(r => r.date).filter(Boolean).sort();
    const dataMeta = {
        totalRaces:    races.length,
        totalRunners:  X_hist.length,
        dataStartDate: allDates[0]                   ?? null,
        dataEndDate:   allDates[allDates.length - 1] ?? null,
    };

    if (isTraining)
        console.log(
            `  Data: ${dataMeta.totalRunners} runners across ${dataMeta.totalRaces} races, ` +
            `${dataMeta.dataStartDate} → ${dataMeta.dataEndDate}`
        );

    return {
        hist:               tf.tensor3d(X_hist),
        static:             tf.tensor2d(X_static),
        y:                  isTraining ? tf.tensor2d(Y, [Y.length, 1]) : null,
        metadata,
        histFeatureCount:   25,
        staticFeatureCount: 27,
        dataMeta,
    };
}

// ─── MODEL ARCHITECTURE ───────────────────────────────────────────────────────
//
// History branch:  Masking → LSTM(64) → LSTM(32) → BN → Dropout(0.3)
// Static branch:   Dense(48) → BN → Dense(32) → Dropout(0.3)
// Combined head:   Dense(48) → BN → Dropout(0.25) → Dense(24) → Dense(1, sigmoid)

function buildModel(timeSteps, histFeatures, staticFeatures) {
    const histInput   = tf.input({ shape: [timeSteps, histFeatures], name: 'history_input' });
    const staticInput = tf.input({ shape: [staticFeatures],          name: 'static_input'  });

    // History branch
    let h = tf.layers.masking({ maskValue: -1 }).apply(histInput);
    h = tf.layers.lstm({ units: 64, returnSequences: true,  recurrentDropout: 0.1,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }).apply(h);
    h = tf.layers.lstm({ units: 32, returnSequences: false, recurrentDropout: 0.1,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }).apply(h);
    h = tf.layers.batchNormalization().apply(h);
    h = tf.layers.dropout({ rate: 0.3 }).apply(h);

    // Static branch
    let s = tf.layers.dense({ units: 48, activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }).apply(staticInput);
    s = tf.layers.batchNormalization().apply(s);
    s = tf.layers.dense({ units: 32, activation: 'relu' }).apply(s);
    s = tf.layers.dropout({ rate: 0.3 }).apply(s);

    // Combined head
    let out = tf.layers.concatenate().apply([h, s]);
    out = tf.layers.dense({ units: 48, activation: 'relu' }).apply(out);
    out = tf.layers.batchNormalization().apply(out);
    out = tf.layers.dropout({ rate: 0.25 }).apply(out);
    out = tf.layers.dense({ units: 24, activation: 'relu' }).apply(out);
    out = tf.layers.dense({ units: 1,  activation: 'sigmoid' }).apply(out);

    const model = tf.model({ inputs: [histInput, staticInput], outputs: out });
    model.compile({
        optimizer: tf.train.adam(0.0003),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });
    return model;
}

// ─── TRAINING ─────────────────────────────────────────────────────────────────

async function runTraining() {
    console.log('--- TRAINING ---');
    const data  = loadData(TRAINING_DATA, true);
    const model = buildModel(MAX_HISTORY, data.histFeatureCount, data.staticFeatureCount);

    let bestValLoss     = Infinity;
    let patienceCounter = 0;
    let lrPatienceCount = 0;
    const EARLY_STOP_PATIENCE = 16;
    const LR_REDUCE_PATIENCE  = 5;
    let epochStart;

    await model.fit([data.hist, data.static], data.y, {
        epochs:          50,
        batchSize:       64,
        validationSplit: 0.1,
        shuffle:         true,
        classWeight:     { 0: 1.0, 1: 1.3 },
        callbacks: {
            onEpochBegin: async () => { epochStart = Date.now(); },
            onEpochEnd: async (epoch, logs) => {
                const ms  = Date.now() - epochStart;
                const lr  = model.optimizer.learningRate;
                const acc = (logs.val_acc || logs.val_accuracy || 0).toFixed(4);

                console.log(
                    `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} | ` +
                    `val_loss=${logs.val_loss.toFixed(4)} | val_acc=${acc} | ` +
                    `lr=${lr.toFixed(6)} | ` +
                    `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`
                );

                if (logs.val_loss < bestValLoss) {
                    bestValLoss = logs.val_loss;
                    patienceCounter = 0;
                    lrPatienceCount = 0;
                    console.log('   New best val_loss — saving...');
                    await saveModel(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(logs.val_loss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        learningRate:  lr,
                        ...data.dataMeta,
                    });
                } else {
                    patienceCounter++;
                    lrPatienceCount++;
                    if (lrPatienceCount >= LR_REDUCE_PATIENCE) {
                        model.optimizer.learningRate = lr * 0.5;
                        console.log(`   Reducing LR: ${(lr * 0.5).toFixed(6)}`);
                        lrPatienceCount = 0;
                    }
                    if (patienceCounter >= EARLY_STOP_PATIENCE) {
                        console.log('--- Early stopping ---');
                        model.stopTraining = true;
                    }
                }
            },
        },
    });

    data.hist.dispose();
    data.static.dispose();
    data.y.dispose();
}

// ─── PREDICTION ───────────────────────────────────────────────────────────────
/*
This should return the data for a single race in JSON format
(structured the same way as the training data).
This function is a leftover from the prototyping phase,
when the front-end was not yet handling the prediction logic.

async function runPrediction(predictionFile) {
    console.log('\n--- PREDICTION ---');
    const model = await loadModel();
    if (!model) { console.log('No saved model found. Run training first.'); return; }

    const data = loadData(predictionFile || PREDICTION_DATA, false);
    if (data.hist.shape[0] === 0) { console.log('No prediction data found.'); return; }

    const scores = await model.predict([data.hist, data.static]).data();

    const results = data.metadata
        .map((m, i) => ({
            No:          m.number,
            Horse:       m.name,
            Driver:      m.driver,
            Probability: (scores[i] * 100).toFixed(1) + '%',
            ImpliedOdds: (1 / scores[i]).toFixed(2),
            Signal:      scores[i] > 0.5 ? 'BET' : 'SKIP',
        }))
        .sort((a, b) => parseFloat(b.Probability) - parseFloat(a.Probability));

    console.table(results);

    data.hist.dispose();
    data.static.dispose();
}
*/
// ─── ENTRY POINT ──────────────────────────────────────────────────────────────
/*
(async () => {
    await runTraining();
    await runPrediction();
})();
*/
 runTraining();
