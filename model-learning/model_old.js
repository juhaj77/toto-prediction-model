const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const csv = require('csv-parser');
const TRAINING_DATA = './Ravit_Opetus_Data_Yhdistetty_v4.csv';
const PREDICTION_DATA = './Ravit_443077258_lahto1.csv';
const MAPPINGS_FILE = './mappings2.json';
const MODEL_FOLDER = './ravimalli-mixed-3';
const MAX_HISTORY = 8;

// --- TALLENNUS- JA LATAUSAPUVÄLINEET (BASE64) ---
const saveModelBase64 = async (model, trainingMeta = {}) => {
    if (!fs.existsSync(MODEL_FOLDER)) fs.mkdirSync(MODEL_FOLDER);
    await model.save(tf.io.withSaveHandler(async (artifacts) => {
        const weightsBase64 = Buffer.from(artifacts.weightData).toString('base64');
        const modelData = {
            modelTopology: artifacts.modelTopology,
            weightSpecs: artifacts.weightSpecs,
            weightData: weightsBase64,
            // --- Mallin metatiedot front-endiä varten ---
            trainingInfo: {
                savedAt:        new Date().toISOString(),
                epoch:          trainingMeta.epoch       ?? null,
                loss:           trainingMeta.loss        ?? null,
                val_loss:       trainingMeta.val_loss    ?? null,
                val_acc:        trainingMeta.val_acc     ?? null,
                learningRate:   trainingMeta.lr          ?? null,
                dataStartDate:  trainingMeta.dataStartDate ?? null,   // vanhin Current_Start_Date opetusdatassa
                dataEndDate:    trainingMeta.dataEndDate   ?? null,   // uusin Current_Start_Date opetusdatassa
                totalRows:      trainingMeta.totalRows     ?? null,   // CSV-rivien kokonaismäärä (historia+current)
                totalStarts:    trainingMeta.totalStarts   ?? null,   // uniikkeja hevonen+lähtö -pareja
            }
        };
        fs.writeFileSync(`${MODEL_FOLDER}/model_full.json`, JSON.stringify(modelData));
        return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
};

const loadModelBase64 = async () => {
    const filePath = `${MODEL_FOLDER}/model_full.json`;
    if (!fs.existsSync(filePath)) return null;
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const weightBuffer = Buffer.from(data.weightData, 'base64');
    const arrayBuffer = weightBuffer.buffer.slice(weightBuffer.byteOffset, weightBuffer.byteOffset + weightBuffer.byteLength);
    const artifacts = {
        modelTopology: data.modelTopology,
        weightSpecs: data.weightSpecs,
        weightData: arrayBuffer
    };
    return await tf.loadLayersModel(tf.io.fromMemory(artifacts));
};

// --- APUFUNKTIOT ---

function mapTrackCondition(condition) {
    const c = (condition || "").toLowerCase().trim();
    switch (c) {
        case 'heavy track': return 0.0;
        case 'quite heavy track': return 0.25;
        case 'light track': return 1.0;
        case 'winter track': return 0.75;
        case 'unknown':
        case '':
        default: return 0.5;
    }
}

function getSurname(fullName) {
    if (!fullName) return 'unknown';
    const parts = fullName.trim().toLowerCase().split(' ');
    return parts[parts.length - 1];
}

function parseDateString(pvmStr) {
    if (!pvmStr || pvmStr.trim() === '' || pvmStr === '0' || pvmStr === 'NaT') return null;
    if (pvmStr.includes('-')) return new Date(pvmStr);
    if (pvmStr.includes('.')) {
        const osat = pvmStr.split('.');
        if (osat.length === 3) {
            const paiva = parseInt(osat[0]);
            const kuukausi = parseInt(osat[1]) - 1;
            let vuosi = parseInt(osat[2]);
            if (vuosi < 100) vuosi += 2000;
            return new Date(vuosi, kuukausi, paiva);
        }
    }
    return null;
}

function adjustKmTime(km, matka) {
    if (!km || isNaN(km) || km <= 0) return km;
    const eroMetreina = 2100 - matka;
    const kompensaatio = (eroMetreina / 2000) * 1.0;  // kerroin optimoitu: korrelaatio hist_sij 0.031→0.090
    return km + kompensaatio;
}

// --- DATAN KÄSITTELY ---

async function loadAndCleanData(tiedostopolku, isTraining = true) {
    let maps = {
        hevoset: {}, valmentajat: {}, ohjastajat: {}, radat: {},
        counts: { h: 1, v: 1, o: 1, r: 1 }
    };

    if (!isTraining && fs.existsSync(MAPPINGS_FILE)) {
        maps = JSON.parse(fs.readFileSync(MAPPINGS_FILE, 'utf8'));
        console.log(`  Ladattu mappings: ${Object.keys(maps.valmentajat).length} valmentajaa, ${Object.keys(maps.ohjastajat).length} ohjastajaa, ${Object.keys(maps.radat).length} rataa`);
    }

    const getMapID = (map, name, type) => {
        if (!name || name === 'Unknown' || name === '' || name === '0') return 0;
        const cleanName = (type === 'ohjastaja') ? getSurname(name) : name.trim().toLowerCase();
        if (isTraining) {
            if (!map[cleanName]) {
                map[cleanName] = maps.counts[type[0]];
                maps.counts[type[0]]++;
            }
            return map[cleanName];
        }
        return map[cleanName] || 0;
    };

    const rawRows = [];
    return new Promise((resolve, reject) => {
        fs.createReadStream(tiedostopolku)
            .pipe(csv({ separator: ';' }))
            .on('data', (row) => {
                if (row.Nimi && row.Scratched !== '1') {
                    rawRows.push(row);
                }
            })
            .on('end', async () => {
                const stats = { SH: { en: [], km: [] }, LV: { en: [], km: [] } };

                rawRows.forEach(r => {
                    const breed = r.Is_Suomenhevonen === '1' ? 'SH' : 'LV';
                    const km = adjustKmTime(parseFloat(r.Km_aika), parseFloat(r.Matka));
                    if (km > 0) stats[breed].km.push(km);
                    if (parseFloat(r.Ennatys_nro) > 0) stats[breed].en.push(parseFloat(r.Ennatys_nro));
                });

                const means = { SH: { en: 28.0, km: 29.0 }, LV: { en: 15.0, km: 16.0 } };
                ['SH', 'LV'].forEach(breedCode => {
                    if (stats[breedCode].en.length) means[breedCode].en = stats[breedCode].en.reduce((a, b)=>a+b)/stats[breedCode].en.length;
                    if (stats[breedCode].km.length) means[breedCode].km = stats[breedCode].km.reduce((a, b)=>a+b)/stats[breedCode].km.length;
                });

                // 1. Ryhmittele lähdöittäin
                const raceGroups = {};
                rawRows.forEach(r => {
                    const raceKey = `${r.RaceID}_${r.Current_Start_Date}`;
                    if (!raceGroups[raceKey]) raceGroups[raceKey] = [];
                    const alreadyAdded = raceGroups[raceKey].some(h => h.Nimi === r.Nimi);
                    if (!alreadyAdded) raceGroups[raceKey].push(r);
                });

                // 2. Ranking-mapit
                const betRankingMap = {};
                const winRankingMap = {};
                const rankingDataMap = {};

                Object.values(raceGroups).forEach(horsesInRace => {
                    const hasBetData   = horsesInRace.some(h => parseFloat(h.Peli_pros) > 0);
                    const hasWinData = horsesInRace.some(h => parseFloat(h.Voitto_pros) > 0);

                    const betActuals   = horsesInRace.filter(h => parseFloat(h.Peli_pros) > 0);
                    const winActuals = horsesInRace.filter(h => parseFloat(h.Voitto_pros) > 0);

                    const betOrder   = [...betActuals].sort((a, b) => (parseFloat(b.Peli_pros)||0) - (parseFloat(a.Peli_pros)||0));
                    const winOrder = [...winActuals].sort((a, b) => (parseFloat(b.Voitto_pros)||0) - (parseFloat(a.Voitto_pros)||0));

                    horsesInRace.forEach(h => {
                        const key = `${h.RaceID}_${h.Current_Start_Date}_${h.Nimi}`;
                        const neutral = (horsesInRace.length + 1) / 2;

                        const betVal = parseFloat(h.Peli_pros) || 0;
                        const betKnown = (hasBetData && betVal > 0) ? 1 : 0;
                        betRankingMap[key] = betKnown
                            ? betOrder.findIndex(j => j.Nimi === h.Nimi) + 1
                            : neutral;

                        const winVal = parseFloat(h.Voitto_pros) || 0;
                        const winKnown = (hasWinData && winVal > 0) ? 1 : 0;
                        winRankingMap[key] = winKnown
                            ? winOrder.findIndex(j => j.Nimi === h.Nimi) + 1
                            : neutral;

                        rankingDataMap[key] = { peliKnown: betKnown, voittoKnown: winKnown };
                    });
                });

                // 3. Ryhmittele hevosittain ja rakenna featuret
                const competitors = {};
                rawRows.forEach(r => {
                    const id = r.RaceID + '_' + r.Nimi;
                    if (!competitors[id]) competitors[id] = [];
                    competitors[id].push(r);
                });

                const X_hist = [];
                const X_static = [];
                const Y = [];
                const metadata = [];

                // ─── TÄSSÄ KOHTAA KUTSUTAAN getMapID → maps täyttyy ───
                Object.values(competitors).forEach(hist => {

                    const histRows = hist.filter(h => {
                        const pvm   = (h.Hist_PVM || '').toString().trim();
                        const kuski = (h.Ohjastaja || '').toString().trim();
                        return pvm !== '' && pvm !== '0' && pvm !== 'NaT'
                            && kuski !== '' && kuski !== '0' && kuski !== '<undefined>';
                    });
                    const histKnown = histRows.length > 0 ? 1 : 0;

                    let prevIndexNorm = 0;
                    if (histRows.length > 0) {
                        let indexScore = 0;
                        let validCount = 0;
                        histRows.forEach(h => {
                            const sij = parseInt(h.Hist_Sij);
                            if (isNaN(sij)) return; // hyl/kesk/tyhjä → ohita
                            validCount++;
                            if      (sij === 1) indexScore += 1.00;
                            else if (sij === 2) indexScore += 0.50;
                            else if (sij === 3) indexScore += 0.33;
                        });
                        prevIndexNorm = validCount > 0 ? indexScore / validCount : 0;
                    }

                    const current = hist[0];

                    const rankKey = `${current.RaceID}_${current.Current_Start_Date}_${current.Nimi}`;
                    const horseCount = (raceGroups[`${current.RaceID}_${current.Current_Start_Date}`] || []).length || 10;
                    const neutral = (horseCount + 1) / 2;

                    const peliRankingFinal   = betRankingMap[rankKey]   ?? neutral;
                    const voittoRankingFinal = winRankingMap[rankKey] ?? neutral;
                    const rankMeta           = rankingDataMap[rankKey]   ?? { peliKnown: 0, voittoKnown: 0 };

                    const rotu = current.Is_Suomenhevonen === '1' ? 'SH' : 'LV';
                    const cartSpec = (current.Current_Special_Cart || "UNKNOWN").toUpperCase();
                    const cSpecActive = cartSpec === 'YES' ? 1 : 0;
                    const cSpecKnown = (cartSpec === 'YES' || cartSpec === 'NO') ? 1 : 0;

                    const etu = (current.Kengat_Etu || "UNKNOWN").toUpperCase();
                    const taka = (current.Kengat_Taka || current.Kengat_taka || "UNKNOWN").toUpperCase();

                    const etuActive = etu === 'HAS_SHOES' ? 1 : 0;
                    const etuKnown  = (etu === 'HAS_SHOES' || etu === 'NO_SHOES') ? 1 : 0;
                    const takaActive = taka === 'HAS_SHOES' ? 1 : 0;
                    const takaKnown  = (taka === 'HAS_SHOES' || taka === 'NO_SHOES') ? 1 : 0;

                    const staticFeats = [
                        (parseFloat(current.Nro) || 1) / 20,
                        getMapID(maps.valmentajat, current.Valmentaja, 'valmentaja') / 2000,  // ← getMapID kutsu
                        (parseFloat(current.Ennatys_nro) || means[rotu].en) / 50,
                        getMapID(maps.ohjastajat, current.Current_Ohjastaja, 'ohjastaja') / 3000,  // ← getMapID kutsu
                        (parseFloat(current.Ika) || 5) / 15,
                        (parseInt(current.Sukupuoli) || 2) / 3,
                        parseInt(current.Is_Suomenhevonen) || 0,
                        etuActive, etuKnown, takaActive, takaKnown,
                        parseInt(current.Kengat_etu_changed) || 0,
                        parseInt(current.Kengat_taka_changed) || 0,
                        (parseFloat(current.Current_Distance) || 2100) / 3100,
                        parseInt(current.Current_Is_Auto) || 0,
                        (parseFloat(current.Peli_pros) || 0) / 100,
                        (parseFloat(current.Voitto_pros) || 0) / 100,
                        parseFloat(current.Voitto_pros) > 0 ? 1 : 0,
                        parseInt(current.Is_Auto_Record) || 0,
                        cSpecActive, cSpecKnown,
                        peliRankingFinal / 20, rankMeta.peliKnown,
                        voittoRankingFinal / 20, rankMeta.voittoKnown,
                        histKnown, prevIndexNorm
                    ];

                    const historySeq = hist.slice(0, MAX_HISTORY).map((h) => {
                        const pvm = (h.Hist_PVM || "").toString().trim();
                        const kuski = (h.Ohjastaja || "").toString().trim();

                        if ((pvm === '' || pvm === '0' || pvm === 'NaT') && (kuski === '' || kuski === '0' || kuski === '<undefined>')) {
                            return new Array(25).fill(-1);
                        }

                        const dateNow = parseDateString(h.Current_Start_Date);
                        const datePrev = parseDateString(h.Hist_PVM);
                        let daysSince = 30;
                        if (dateNow && datePrev) daysSince = Math.min(365, (dateNow - datePrev) / (1000*60*60*24));

                        const hSpec = (h.Hist_Special_Cart || "UNKNOWN").toUpperCase();
                        const hSpecActive = hSpec === 'YES' ? 1 : 0;
                        const hSpecKnown = (hSpec === 'YES' || hSpec === 'NO') ? 1 : 0;

                        const hEtuStr  = (h.Hist_kengat_etu  || "UNKNOWN").toUpperCase();
                        const hTakaStr = (h.Hist_kengat_taka || "UNKNOWN").toUpperCase();
                        const hEtuActive  = hEtuStr  === 'HAS_SHOES' ? 1 : 0;
                        const hEtuKnown   = (hEtuStr  === 'HAS_SHOES' || hEtuStr  === 'NO_SHOES') ? 1 : 0;
                        const hTakaActive = hTakaStr === 'HAS_SHOES' ? 1 : 0;
                        const hTakaKnown  = (hTakaStr === 'HAS_SHOES' || hTakaStr === 'NO_SHOES') ? 1 : 0;

                        const rawMatka = parseFloat(h.Matka) || 0;
                        const matkaKnown = rawMatka > 0 ? 1 : 0;
                        const matkaFinal = matkaKnown ? (rawMatka / 3100) : 0.67;

                        let rawKm = parseFloat(h.Km_aika) || 0;
                        const kmKnown = rawKm > 0 ? 1 : 0;
                        const kmFinal = kmKnown ? (rawKm / 100) : (means[rotu].km / 100);

                        let rawPalkinto = parseFloat(h.Palkinto) || 0;
                        const palkintoKnown = rawPalkinto > 0 ? 1 : 0;
                        const palkintoFinal = palkintoKnown ? (Math.log1p(rawPalkinto) / 10) : 0.55;

                        let rawKerroin = parseFloat(h.Kerroin);
                        const kerroinKnown = (!isNaN(rawKerroin) && rawKerroin > 0) ? 1 : 0;
                        const kerroinFinal = kerroinKnown ? (Math.log1p(rawKerroin) / 5) : 0.5;

                        const sijRaw = parseInt(h.Hist_Sij);
                        const sijKnown = (!isNaN(sijRaw) && sijRaw > 0) ? 1 : 0;
                        const sijFinal = sijKnown ? (sijRaw / 20) : 0.5;

                        return [
                            kmFinal, kmKnown,
                            matkaFinal, matkaKnown,
                            daysSince / 365,
                            sijFinal, sijKnown,
                            palkintoFinal, palkintoKnown,
                            kerroinFinal, kerroinKnown,
                            parseInt(h.Hist_Is_Auto) || 0,
                            parseInt(h.Laukka) || 0,
                            (parseInt(h.RataNro) || 1) / 30,
                            getMapID(maps.ohjastajat, h.Ohjastaja, 'ohjastaja') / 3000,  // ← getMapID kutsu
                            getMapID(maps.radat, h.Rata, 'rata') / 500,                  // ← getMapID kutsu
                            parseInt(h.Hylatty) || 0,
                            parseInt(h.Keskeytys) || 0,
                            hEtuActive, hEtuKnown,
                            hTakaActive, hTakaKnown,
                            hSpecActive, hSpecKnown,
                            mapTrackCondition(h.Track_Condition)
                        ];
                    });

                    while (historySeq.length < MAX_HISTORY) historySeq.push(new Array(25).fill(-1));

                    X_hist.push(historySeq);
                    X_static.push(staticFeats);

                    if (isTraining) {
                        const sijoitus = parseInt(current.SIJOITUS);
                        Y.push((sijoitus >= 1 && sijoitus <= 3) ? 1 : 0);
                    } else {
                        metadata.push({ Nro: current.Nro, Nimi: current.Nimi, Ohjastaja: current.Current_Ohjastaja });
                    }
                });

                // ─── TALLENNETAAN VASTA TÄSSÄ kun getMapID on täyttänyt maps:in ───
                if (isTraining) {
                    fs.writeFileSync(MAPPINGS_FILE, JSON.stringify(maps, null, 2));
                    console.log(`  ✓ Mappings tallennettu: ${Object.keys(maps.valmentajat).length} valmentajaa, ${Object.keys(maps.ohjastajat).length} ohjastajaa, ${Object.keys(maps.radat).length} rataa`);
                }

                // ─── Datametatiedot front-endiä varten ───
                // totalStarts = yksittäiset (hevonen × current-lähtö) -parit.
                // Jokaiseen hevoseen sisältyy 1 current + max 8 historia = max 9 lähtöä,
                // mutta totalStarts laskee vain uniikkeja current-lähtöjä.
                const totalRows  = rawRows.length;              // CSV-rivien kokonaismäärä
                const totalStarts = Object.keys(competitors).length;  // uniikkeja hevonen+lähtö -pareja

                const allDates = rawRows
                    .map(r => r.Current_Start_Date || '')
                    .filter(d => d && d !== '0' && d !== 'NaT')
                    .map(d => parseDateString(d))
                    .filter(Boolean)
                    .sort((a, b) => a - b);

                const dataStartDate = allDates.length > 0 ? allDates[0].toISOString().slice(0, 10) : null;
                const dataEndDate   = allDates.length > 0 ? allDates[allDates.length - 1].toISOString().slice(0, 10) : null;

                if (isTraining) {
                    console.log(`  ✓ Dataa: ${totalRows} riviä (${totalStarts} starttia), ${dataStartDate} → ${dataEndDate}`);
                }

                resolve({
                    hist: tf.tensor3d(X_hist),
                    static: tf.tensor2d(X_static),
                    y: isTraining ? tf.tensor2d(Y, [Y.length, 1]) : null,
                    metadata: metadata,
                    histFeatureCount: 25,
                    staticFeatureCount: 27,
                    dataMeta: { totalRows, totalStarts, dataStartDate, dataEndDate }
                });
            })
            .on('error', reject);
    });
}

// --- MALLIN RAKENNUS ---
const buildMixedModel = (timeSteps, histFeatures, staticFeatures) => {
    const historyInput = tf.input({ shape: [timeSteps, histFeatures], name: 'history_input' });
    let maskedInput = tf.layers.masking({ maskValue: -1 }).apply(historyInput);
    let lstm1 = tf.layers.lstm({
        units: 64, returnSequences: true, recurrentDropout: 0.1,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }).apply(maskedInput);
    let lstm2 = tf.layers.lstm({
        units: 32, returnSequences: false, recurrentDropout: 0.1,
        kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }).apply(lstm1);
    let histBn   = tf.layers.batchNormalization().apply(lstm2);
    let histDrop = tf.layers.dropout({ rate: 0.3 }).apply(histBn);

    const staticInput = tf.input({ shape: [staticFeatures], name: 'static_input' });
    let static1    = tf.layers.dense({ units: 48, activation: 'relu', kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }).apply(staticInput);
    let staticBn   = tf.layers.batchNormalization().apply(static1);
    let static2    = tf.layers.dense({ units: 32, activation: 'relu' }).apply(staticBn);
    let staticDrop = tf.layers.dropout({ rate: 0.3 }).apply(static2);

    const concatenated = tf.layers.concatenate().apply([histDrop, staticDrop]);
    let dense1 = tf.layers.dense({ units: 48, activation: 'relu' }).apply(concatenated);
    dense1 = tf.layers.batchNormalization().apply(dense1);
    dense1 = tf.layers.dropout({ rate: 0.25 }).apply(dense1);
    let dense2 = tf.layers.dense({ units: 24, activation: 'relu' }).apply(dense1);
    const output = tf.layers.dense({ units: 1, activation: 'sigmoid' }).apply(dense2);

    const model = tf.model({ inputs: [historyInput, staticInput], outputs: output });
    model.compile({
        optimizer: tf.train.adam(0.0003),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    return model;
};

// --- PÄÄTOIMINNOT ---

async function runTraining() {
    console.log('--- ALOITETAAN OPETUS ---');
    const data = await loadAndCleanData(TRAINING_DATA, true);
    const model = buildMixedModel(MAX_HISTORY, data.histFeatureCount, data.staticFeatureCount);

    let bestLoss = Infinity;
    let patienceCounter = 0;
    let lrPatienceCounter = 0;
    const EARLY_STOP_PATIENCE = 16;
    const LR_REDUCE_PATIENCE = 5;
    let epochStartTime;

    await model.fit([data.hist, data.static], data.y, {
        epochs: 50,
        batchSize: 64,
        validationSplit: 0.1,
        shuffle: true,
        classWeight: { 0: 1.0, 1: 1.3 },
        callbacks: {
            onEpochBegin: async () => { epochStartTime = Date.now(); },
            onEpochEnd: async (epoch, logs) => {
                const durationMs = Date.now() - epochStartTime;
                const minutes = Math.floor(durationMs / 60000);
                const seconds = ((durationMs % 60000) / 1000).toFixed(0);
                const vLoss = logs.val_loss;
                const vAcc  = (logs.val_acc || logs.val_accuracy || 0).toFixed(4);
                const currentLR = model.optimizer.learningRate;

                console.log(`Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)} | val_loss=${vLoss.toFixed(4)} | val_acc=${vAcc} | LR=${currentLR.toFixed(6)} | Aika: ${minutes}m ${seconds}s`);

                if (vLoss < bestLoss) {
                    bestLoss = vLoss;
                    patienceCounter = 0;
                    lrPatienceCounter = 0;
                    console.log(`   ⭐ Uusi paras val_loss, tallennetaan...`);
                    await saveModelBase64(model, {
                        epoch:         epoch + 1,
                        loss:          Math.round(logs.loss * 10000) / 10000,
                        val_loss:      Math.round(vLoss * 10000) / 10000,
                        val_acc:       Math.round((logs.val_acc || logs.val_accuracy || 0) * 10000) / 10000,
                        lr:            currentLR,
                        dataStartDate: data.dataMeta.dataStartDate,
                        dataEndDate:   data.dataMeta.dataEndDate,
                        totalRows:     data.dataMeta.totalRows,
                        totalStarts:   data.dataMeta.totalStarts,
                    });
                } else {
                    patienceCounter++;
                    lrPatienceCounter++;
                    if (lrPatienceCounter >= LR_REDUCE_PATIENCE) {
                        model.optimizer.learningRate = currentLR * 0.5;
                        console.log(`   📉 Lasketaan LR: ${(currentLR * 0.5).toFixed(6)}`);
                        lrPatienceCounter = 0;
                    }
                    if (patienceCounter >= EARLY_STOP_PATIENCE) {
                        console.log(`\n--- Early Stopping ---`);
                        model.stopTraining = true;
                    }
                }
            }
        }
    });
}

async function runPrediction() {
    console.log('\n--- ALOITETAAN ENNUSTUS ---');
    const model = await loadModelBase64();
    if (!model) { console.log("Malli-tiedostoa ei löytynyt. Aja opetus ensin."); return; }

    const data = await loadAndCleanData(PREDICTION_DATA, false);
    if (data.hist.shape[0] === 0) { console.log("Ei ennustettavaa dataa."); return; }

    const predictions = model.predict([data.hist, data.static]);
    const scores = await predictions.data();

    const tulokset = data.metadata.map((m, i) => ({
        Nro: m.Nro, Hevonen: m.Nimi, Ohjastaja: m.Ohjastaja,
        Todennakoisyys: (scores[i] * 100).toFixed(1) + '%',
        Malli_Kerroin: (1 / scores[i]).toFixed(2),
        Arvio: scores[i] > 0.5 ? 'PELATTAVA' : 'HUTI'
    })).sort((a, b) => parseFloat(b.Todennakoisyys) - parseFloat(a.Todennakoisyys));

    console.table(tulokset);
    data.hist.dispose();
    data.static.dispose();
    predictions.dispose();
}
(async () => {
    await runTraining();
    await runPrediction();
})();
//runTraining();
// runPrediction();
