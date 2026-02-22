import React, { useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';

// ─── CONFIG ────────────────────────────────────────────────────────────────────
const MAX_HISTORY = 8;

// ─── APUFUNKTIOT — täsmälleen sama kuin ravimalli.js / MATLAB-scraper ─────────

function puhdista(str) {
    if (!str) return '';
    return str
        .replace(/ä/g, 'a').replace(/ö/g, 'o').replace(/å/g, 'a')
        .replace(/Ä/g, 'A').replace(/Ö/g, 'O').replace(/Å/g, 'A')
        .replace(/[^a-zA-Z0-9\s\-\.\ :]/g, '')
        .trim();
}

// Km-aika: EI kompensointia — sama kuin scraper tallentaa suoraan
// MATLAB: kmNum = str2double(strrep(kmNumStr, ',', '.'))
function parseKmAika(kmRaw) {
    if (!kmRaw) return { kmNum: 0, histIsAuto: 0, isLaukka: 0 };
    const s        = String(kmRaw);
    const histIsAuto = s.includes('a') ? 1 : 0;
    const isLaukka   = s.includes('x') ? 1 : 0;
    const kmNumStr   = s.match(/[\d,]+/)?.[0] || '';
    const kmNum      = parseFloat(kmNumStr.replace(',', '.')) || 0;
    return { kmNum, histIsAuto, isLaukka };
}

// Km-ajan matkakompensointi — normalisoi eri matkat vertailukelpoisiksi
// Korrelaatio Hist_sij:iin: ilman=0.031, kompensaatiolla (ero/2000)=0.090
function kompensoiKmAika(km, matka) {
    if (!km || isNaN(km) || km <= 0) return km;
    const eroMetreina = 2100 - matka;
    const kompensaatio = (eroMetreina / 500) * 1.0;
    return km + kompensaatio;
}

// Sijoitus — sama kuin scraper sijRaw-logiikka
function parseSijoitus(sijRaw) {
    if (!sijRaw) return { sijNum: 10, isHyl: 0, isKesk: 0 };
    const s      = String(sijRaw).toLowerCase();
    const isHyl  = /[hdp]/.test(s) ? 1 : 0;
    const isKesk = s.includes('k') ? 1 : 0;
    const m      = s.match(/^\d+/);
    let sijNum   = 10;
    if (m)          sijNum = parseInt(m[0]);
    else if (isHyl) sijNum = 20;
    else if (isKesk)sijNum = 21;
    return { sijNum, isHyl, isKesk };
}

function parseSukupuoli(g) {
    if (!g) return 2;
    if (g === 'TAMMA') return 1;
    if (g === 'ORI')   return 3;
    return 2;
}

// Ennätys — sama kuin scraper eMatch-logiikka
function parseEnnatys(runner) {
    for (const key of ['handicapRaceRecord', 'mobileStartRecord', 'vaultStartRecord']) {
        const val = runner[key];
        if (val) {
            const num = parseFloat(String(val).replace(',', '.'));
            if (!isNaN(num) && num > 0)
                return { ennatys: num, isAutoRecord: key === 'mobileStartRecord' ? 1 : 0 };
        }
    }
    return { ennatys: 0, isAutoRecord: 0 };
}

// Kengät — arvo tulee suoraan API:sta
function parseKengat(val) {
    if (!val) return 'UNKNOWN';
    const s = String(val).toUpperCase();
    if (s === 'HAS_SHOES' || s === 'TRUE' || s === 'YES' || s === '1') return 'HAS_SHOES';
    if (s === 'NO_SHOES'  || s === 'FALSE'|| s === 'NO'  || s === '0') return 'NO_SHOES';
    return 'UNKNOWN';
}

// Peli% — sama kuin scraper: percentage / 100
function parsePeliProsentti(runner) {
    const pct = runner.percentage ?? runner.winPercentage ?? 0;
    return parseFloat(pct) / 100;
}

// Voitto% — sama kuin scraper statBlock: position1/starts*100
function parseVoittoProsentti(runner) {
    try {
        const stats  = runner.statistics?.total || runner.stats?.total;
        if (!stats) return 0;
        const starts = parseInt(stats.starts || stats.startCount || 0);
        const wins   = parseInt(stats.position1 || stats.wins || 0);
        if (starts > 0) return Math.round((wins / starts) * 100 * 100) / 100;
    } catch {}
    return 0;
}

// Radan kunto — sama kuin ravimalli.js mapTrackCondition
function mapTrackCondition(condition) {
    const c = (condition || '').toLowerCase().trim();
    const m = { 'heavy track': 0.0, 'quite heavy track': 0.25, 'light track': 1.0, 'winter track': 0.75 };
    return m[c] ?? 0.5;
}

// Sukunimi — sama kuin ravimalli.js getSurname
function getSurname(fullName) {
    if (!fullName) return 'unknown';
    return fullName.trim().toLowerCase().split(' ').at(-1);
}

// Päivämäärä-parseri — sama kuin ravimalli.js pvmToDate
function pvmToDate(pvmStr) {
    if (!pvmStr || pvmStr.trim() === '' || pvmStr === '0' || pvmStr === 'NaT') return null;
    if (pvmStr.includes('-')) return new Date(pvmStr);
    if (pvmStr.includes('.')) {
        const [d, mo, y] = pvmStr.split('.');
        let year = parseInt(y);
        if (year < 100) year += 2000;
        return new Date(year, parseInt(mo) - 1, parseInt(d));
    }
    return null;
}

// ─── FEATURE BUILDER — identtinen ravimalli.js Object.values(kisaajat).forEach ─

function buildFeatures(runners, maps, currDate, currDist, isSH, currIsAuto) {
    const rotu  = isSH ? 'SH' : 'LV';
    const means = { SH: { en: 28.0, km: 29.0 }, LV: { en: 15.0, km: 16.0 } };

    // Ennustuksessa: getMapID palauttaa mappauksen tai 0
    const getMapID = (map, name, type) => {
        if (!name || name === 'Unknown' || name === '' || name === '0') return 0;
        const key = (type === 'ohjastaja') ? getSurname(name) : name.trim().toLowerCase();
        return map[key] || 0;
    };

    // Ranking — sama kuin ravimalli.js peliRankingMap / voittoRankingMap
    const nHevosia  = runners.length;
    const neutraali = (nHevosia + 1) / 2;

    const onkoPeli   = runners.some(r => (parseFloat(r.peliP)   || 0) > 0);
    const onkoVoitto = runners.some(r => (parseFloat(r.voittoP) || 0) > 0);

    const jarjestysPeli   = [...runners.filter(r => (parseFloat(r.peliP)   || 0) > 0)]
        .sort((a, b) => (parseFloat(b.peliP)   || 0) - (parseFloat(a.peliP)   || 0));
    const jarjestysVoitto = [...runners.filter(r => (parseFloat(r.voittoP) || 0) > 0)]
        .sort((a, b) => (parseFloat(b.voittoP) || 0) - (parseFloat(a.voittoP) || 0));

    const X_hist   = [];
    const X_static = [];
    const metadata = [];

    for (const runner of runners) {
        const hPeliKnown   = (onkoPeli   && (parseFloat(runner.peliP)   || 0) > 0) ? 1 : 0;
        const hVoittoKnown = (onkoVoitto && (parseFloat(runner.voittoP) || 0) > 0) ? 1 : 0;
        const peliRank     = hPeliKnown   ? jarjestysPeli.findIndex(r => r.nimi   === runner.nimi) + 1 : neutraali;
        const voittoRank   = hVoittoKnown ? jarjestysVoitto.findIndex(r => r.nimi === runner.nimi) + 1 : neutraali;

        const etu  = (runner.kEtu  || 'UNKNOWN').toUpperCase();
        const taka = (runner.kTaka || 'UNKNOWN').toUpperCase();
        const etuActive  = etu  === 'HAS_SHOES' ? 1 : 0;
        const etuKnown   = (etu  === 'HAS_SHOES' || etu  === 'NO_SHOES') ? 1 : 0;
        const takaActive = taka === 'HAS_SHOES' ? 1 : 0;
        const takaKnown  = (taka === 'HAS_SHOES' || taka === 'NO_SHOES') ? 1 : 0;
        const cSpec      = (runner.currSpecCart || 'UNKNOWN').toUpperCase();
        const cSpecActive = cSpec === 'YES' ? 1 : 0;
        const cSpecKnown  = (cSpec === 'YES' || cSpec === 'NO') ? 1 : 0;

        // 25 staattista featurea — täsmälleen sama järjestys kuin ravimalli.js
        const staticFeats = [
            (parseFloat(runner.nro) || 1) / 20,                                         // 0
            getMapID(maps.valmentajat, runner.valmentaja, 'valmentaja') / 2000,          // 1
            (parseFloat(runner.ennatys) || means[rotu].en) / 50,                         // 2
            getMapID(maps.ohjastajat, runner.currOhj, 'ohjastaja') / 3000,               // 3
            (parseFloat(runner.ika) || 5) / 15,                                          // 4
            (parseInt(runner.sukupuoli) || 2) / 3,                                       // 5
            isSH ? 1 : 0,                                                                // 6
            etuActive, etuKnown, takaActive, takaKnown,                                  // 7-10
            runner.kEtuCh  || 0,                                                         // 11
            runner.kTakaCh || 0,                                                         // 12
            (parseFloat(currDist) || 2100) / 3100,                                       // 13
            currIsAuto ? 1 : 0,                                                          // 14
            (parseFloat(runner.peliP)   || 0) / 100,                                     // 15
            (parseFloat(runner.voittoP) || 0) / 100,                                     // 16
            parseFloat(runner.voittoP) > 0 ? 1 : 0,                                     // 17 voitto_known
            runner.isAutoRecord || 0,                                                    // 18
            cSpecActive, cSpecKnown,                                                     // 19-20
            peliRank   / 20, hPeliKnown,                                                 // 21-22
            voittoRank / 20, hVoittoKnown,                                               // 23-24
        ];

        // Historia-sekvenssit — täsmälleen sama kuin ravimalli.js historySeq
        const hist = runner.priorStarts || [];
        const historySeq = [];

        for (let i = 0; i < MAX_HISTORY; i++) {
            const h = hist[i];

            if (!h) { historySeq.push(new Array(25).fill(-1)); continue; }

            const pvm   = (h.shortMeetDate || h.meetDate || '').toString().trim();
            const kuski = (h.driverFullName || h.driverName || '').toString().trim();
            if ((pvm === '' || pvm === '0' || pvm === 'NaT') &&
                (kuski === '' || kuski === '0' || kuski === '<undefined>')) {
                historySeq.push(new Array(25).fill(-1)); continue;
            }

            const dateNow  = pvmToDate(currDate);
            const datePrev = pvmToDate(pvm);
            let daysSince  = 30;
            if (dateNow && datePrev)
                daysSince = Math.min(365, (dateNow - datePrev) / (1000 * 60 * 60 * 24));

            // KM-AIKA: kompensoidaan matkan mukaan (korrelaatio hist_sij:iin paranee 0.031 → 0.090)
            const { kmNum, histIsAuto, isLaukka } = parseKmAika(h.kmTime);
            const kmKomp  = kompensoiKmAika(kmNum, parseFloat(h.distance) || 2100);
            const kmKnown = kmKomp > 0 ? 1 : 0;
            const kmFinal = kmKnown ? kmKomp / 100 : means[rotu].km / 100;

            const rawMatka   = parseFloat(h.distance) || 0;
            const matkaKnown = rawMatka > 0 ? 1 : 0;
            const matkaFinal = matkaKnown ? rawMatka / 3100 : 0.67;

            // Palkinto ja kerroin — sama jako kuin scraper: firstPrize/100, winOdd/10
            const rawPalkinto   = (parseFloat(h.firstPrize) || 0) / 100;
            const palkintoKnown = rawPalkinto > 0 ? 1 : 0;
            const palkintoFinal = palkintoKnown ? Math.log1p(rawPalkinto) / 10 : 0.55;

            const rawKerroin   = (parseFloat(h.winOdd) || 0) / 10;
            const kerroinKnown = rawKerroin > 0 ? 1 : 0;
            const kerroinFinal = kerroinKnown ? Math.log1p(rawKerroin) / 5 : 0.5;

            const { sijNum, isHyl, isKesk } = parseSijoitus(h.result);
            const sijKnown = (!isNaN(sijNum) && sijNum > 0) ? 1 : 0;
            const sijFinal = sijKnown ? sijNum / 20 : 0.5;

            const hEtuStr  = (h.frontShoes  || 'UNKNOWN').toUpperCase();
            const hTakaStr = (h.rearShoes   || 'UNKNOWN').toUpperCase();
            const hEtuActive  = hEtuStr  === 'HAS_SHOES' ? 1 : 0;
            const hEtuKnown   = (hEtuStr  === 'HAS_SHOES' || hEtuStr  === 'NO_SHOES') ? 1 : 0;
            const hTakaActive = hTakaStr === 'HAS_SHOES' ? 1 : 0;
            const hTakaKnown  = (hTakaStr === 'HAS_SHOES' || hTakaStr === 'NO_SHOES') ? 1 : 0;
            const hSpec      = (h.specialCart || 'UNKNOWN').toUpperCase();
            const hSpecActive = hSpec === 'YES' ? 1 : 0;
            const hSpecKnown  = (hSpec === 'YES' || hSpec === 'NO') ? 1 : 0;

            // 25 historia-featurea — täsmälleen sama järjestys kuin ravimalli.js
            historySeq.push([
                kmFinal,       kmKnown,                                               // 0-1
                matkaFinal,    matkaKnown,                                            // 2-3
                daysSince / 365,                                                      // 4
                sijFinal,      sijKnown,                                              // 5-6
                palkintoFinal, palkintoKnown,                                         // 7-8
                kerroinFinal,  kerroinKnown,                                          // 9-10
                histIsAuto,                                                           // 11
                isLaukka,                                                             // 12
                (parseInt(h.startTrack) || 1) / 30,                                   // 13 RataNro
                getMapID(maps.ohjastajat, h.driverFullName || h.driverName, 'ohjastaja') / 3000, // 14
                getMapID(maps.radat, h.trackCode || h.track, 'rata') / 500,           // 15
                isHyl,         isKesk,                                                // 16-17
                hEtuActive,    hEtuKnown,                                             // 18-19
                hTakaActive,   hTakaKnown,                                            // 20-21
                hSpecActive,   hSpecKnown,                                            // 22-23
                mapTrackCondition(h.trackCondition),                                  // 24
            ]);
        }

        while (historySeq.length < MAX_HISTORY) historySeq.push(new Array(25).fill(-1));

        X_hist.push(historySeq);
        X_static.push(staticFeats);
        metadata.push({ Nro: runner.nro, Nimi: runner.nimi, Ohjastaja: runner.currOhj });
    }

    return { X_hist, X_static, metadata };
}

// ─── JSON-HAKU selkeällä virheilmoituksella ───────────────────────────────────
// Vite-dev-server palauttaa 200 + index.html kun tiedostoa ei löydy public/:sta.
// Tarkistetaan content-type ja sisällön alku ennen JSON.parse:a.

async function lataaJSON(polku) {
    const res = await fetch(polku);
    if (!res.ok)
        throw new Error(`HTTP ${res.status} — tiedostoa ei löydy: ${polku}\nKopioi tiedosto public/-kansioon.`);
    const text = await res.text();
    if (text.trimStart().startsWith('<'))
        throw new Error(`Palvelin palautti HTML:n eikä JSON:ia polusta: ${polku}\nTiedosto puuttuu public/-kansiosta.`);
    return JSON.parse(text);
}

// ─── MAIN APP ──────────────────────────────────────────────────────────────────

export default function App() {
    const [ravit,           setRavit]           = useState([]);
    const [lahdot,          setLahdot]          = useState([]);
    const [valittuCardId,   setValittuCardId]   = useState('');
    const [valittuLahto,    setValittuLahto]    = useState('');
    const [loadingRavit,    setLoadingRavit]    = useState(false);
    const [loadingLahdot,   setLoadingLahdot]   = useState(false);
    const [loadingEnnustus, setLoadingEnnustus] = useState(false);
    const [ennustukset,     setEnnustukset]     = useState([]);
    const [virhe,           setVirhe]           = useState('');
    const [model,           setModel]           = useState(null);
    const [maps,            setMaps]            = useState(null);
    const [modelInfo,       setModelInfo]       = useState(null);
    const [modelStatus,     setModelStatus]     = useState('idle');

    // ── Ladataan malli + mappings kerran ────────────────────────────────────
    useEffect(() => {
        const lataa = async () => {
            setModelStatus('loading');
            try {
                const mData = await lataaJSON('/mappings.json');
                setMaps(mData);

                const mfData = await lataaJSON('/ravimalli-mixed/model_full.json');
                if (mfData.trainingInfo) setModelInfo(mfData.trainingInfo);

                // base64 → ArrayBuffer
                const binary = atob(mfData.weightData);
                const bytes  = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

                const loadedModel = await tf.loadLayersModel(tf.io.fromMemory({
                    modelTopology: mfData.modelTopology,
                    weightSpecs:   mfData.weightSpecs,
                    weightData:    bytes.buffer,
                }));
                setModel(loadedModel);
                setModelStatus('ready');
            } catch (e) {
                console.error('Mallin lataus epäonnistui:', e);
                setVirhe(e.message);
                setModelStatus('error');
            }
        };
        lataa();
    }, []);

    // ── Ravit ────────────────────────────────────────────────────────────────
    useEffect(() => {
        setLoadingRavit(true);
        fetch('/api-veikkaus/api/toto-info/v1/cards/today')
            .then(r => r.json())
            .then(d => setRavit(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingRavit(false));
    }, []);

    // ── Lähdöt ───────────────────────────────────────────────────────────────
    useEffect(() => {
        if (!valittuCardId) { setLahdot([]); return; }
        setLoadingLahdot(true);
        setValittuLahto('');
        setEnnustukset([]);
        fetch(`/api-veikkaus/api/toto-info/v1/card/${valittuCardId}/races`)
            .then(r => r.json())
            .then(d => setLahdot(d.collection || []))
            .catch(console.error)
            .finally(() => setLoadingLahdot(false));
    }, [valittuCardId]);

    // ── Ennustus ─────────────────────────────────────────────────────────────
    const ajaEnnustus = useCallback(async () => {
        if (!valittuCardId || !valittuLahto || !model || !maps) return;
        setLoadingEnnustus(true);
        setVirhe('');
        setEnnustukset([]);

        try {
            const cardData  = await fetch(`/api-veikkaus/api/toto-info/v1/card/${valittuCardId}`).then(r => r.json());
            const currDate  = puhdista(cardData.meetDate || cardData.date || '');

            const race = lahdot.find(l => String(l.number) === String(valittuLahto));
            if (!race) throw new Error('Lähtöä ei löydy');

            const raceId     = String(race.raceId || race.id);
            const currDist   = parseInt(race.distance || 2100);
            const isSH       = (race.breed === 'K' || race.breed === 'FINNHORSE');
            const currIsAuto = (race.startType === 'CAR_START' || race.startType === 'AUTO');

            const runnersRaw = await fetch(`/api-veikkaus/api/toto-info/v1/race/${raceId}/runners`).then(r => r.json());

            // Veikkaus API voi palauttaa hevoset usealla eri tavalla — kokeillaan kaikki vaihtoehdot
            const runnersArr = Array.isArray(runnersRaw)
                ? runnersRaw
                : (runnersRaw.collection || runnersRaw.runners || runnersRaw.data || Object.values(runnersRaw));

            // Debug: näytä konsolissa mitä API palautti
            console.log('[runners] API vastaus rakenne:', Object.keys(runnersRaw));
            console.log('[runners] Löydettiin', runnersArr.length, 'hevosta ennen filtteröintiä');
            if (runnersArr.length > 0) console.log('[runners] Esimerkkirivi:', JSON.stringify(runnersArr[0]).slice(0, 300));

            // MATLAB-scraper ei filtteroi scratched-hevosia pois — tehdään samoin.
            // scraper: is_scratched = double(~isempty(strfind(nykyhetkiBlock, '"scratched":true')))
            // eli scraped-tieto tallennetaan featureksi mutta hevonen EI putoa pois.
            // Filtteröidään vain jos scratched on eksplisiittisesti true (boolean).
            const runners = runnersArr
                .filter(r => r.scratched !== true)
                .map(r => ({
                    nro:          parseInt(r.startNumber || r.number || 0),
                    nimi:         puhdista(r.horseName || r.name || ''),
                    valmentaja:   puhdista(r.coachName || r.trainerName || ''),
                    currOhj:      puhdista(r.driverName || ''),
                    ika:          parseInt(r.horseAge || r.age || 0),
                    sukupuoli:    parseSukupuoli(r.gender),
                    kEtu:         parseKengat(r.frontShoes),
                    kTaka:        parseKengat(r.rearShoes),
                    kEtuCh:       r.frontShoesChanged === true ? 1 : 0,
                    kTakaCh:      r.rearShoesChanged  === true ? 1 : 0,
                    currSpecCart: r.specialCart || 'UNKNOWN',
                    ...parseEnnatys(r),
                    peliP:        parsePeliProsentti(r),
                    voittoP:      parseVoittoProsentti(r),
                    priorStarts:  r.priorStarts || r.previousStarts || r.starts || [],
                }));

            console.log('[runners] Hevosia filtteröinnin jälkeen:', runners.length);

            if (runners.length === 0) {
                const keys = Object.keys(runnersRaw);
                throw new Error(
                    `Ei hevosia lähdössä.\nAPI-vastauksen avaimet: [${keys.join(', ')}]\n` +
                    `Array-pituus ennen filtteröintiä: ${runnersArr.length}\n` +
                    `Tarkista Developer Tools > Console lisätiedot varten.`
                );
            }

            const { X_hist, X_static, metadata } = buildFeatures(
                runners, maps, currDate, currDist, isSH, currIsAuto
            );

            const histTensor   = tf.tensor3d(X_hist);
            const staticTensor = tf.tensor2d(X_static);
            const pred         = model.predict([histTensor, staticTensor]);
            const scores       = await pred.data();
            histTensor.dispose();
            staticTensor.dispose();
            pred.dispose();

            setEnnustukset(
                metadata
                    .map((m, i) => ({ nro: m.Nro, nimi: m.Nimi, ohjastaja: m.Ohjastaja, prob: scores[i] }))
                    .sort((a, b) => b.prob - a.prob)
            );
        } catch (e) {
            console.error(e);
            setVirhe(e.message || 'Tuntematon virhe');
        } finally {
            setLoadingEnnustus(false);
        }
    }, [valittuCardId, valittuLahto, model, maps, lahdot]);

    // ─── UI ───────────────────────────────────────────────────────────────────

    // Pakota body/html täyttämään koko leveys ilman Viten default-margineja
    useEffect(() => {
        document.body.style.margin = '0';
        document.body.style.padding = '0';
        document.body.style.width = '100%';
        document.documentElement.style.width = '100%';
    }, []);

    const SC = { idle: '#888', loading: '#f0a500', ready: '#2ecc71', error: '#e74c3c' };
    const SL = { idle: 'Odottaa', loading: 'Ladataan mallia…', ready: 'Malli valmis', error: 'Latausvirhe' };

    const canRun = valittuLahto && modelStatus === 'ready' && !loadingEnnustus;

    return (
        <div style={{ minHeight: '100vh', background: '#1a1a1a', boxSizing: 'border-box', padding: '40px 24px' }}>
            <div style={{ background: '#0d0f14', width: 'fit-content', minWidth: 700, margin: '0 auto', color: '#e8eaf0',
                fontFamily: "'IBM Plex Mono','Courier New',monospace", padding: '32px 40px', borderRadius: 8 }}>

                {/* Header */}
                <div style={{ marginBottom: 28, borderBottom: '1px solid #1e2330', paddingBottom: 20 }}>
                    <div style={{ fontSize: 11, letterSpacing: 4, color: '#4a90d9', textTransform: 'uppercase', marginBottom: 6 }}>
                        Toto-ennustejärjestelmä
                    </div>
                    <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700, color: '#fff', letterSpacing: -0.5 }}>
                        RaviMalli v5
                    </h1>

                    {/* Status + model info */}
                    <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 16, alignItems: 'center', fontSize: 12 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: '50%', display: 'inline-block',
                background: SC[modelStatus], boxShadow: `0 0 6px ${SC[modelStatus]}` }} />
            <span style={{ color: SC[modelStatus] }}>{SL[modelStatus]}</span>
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
              <span>data rows <b style={{ color: '#aaa' }}>{modelInfo.totalRows?.toLocaleString('fi-FI')}</b></span>
              <span style={{ color: '#333' }}>·</span>
              <span>runners <b style={{ color: '#aaa' }}>{modelInfo.totalStarts?.toLocaleString('fi-FI')}</b></span>
            </span>
                        )}
                    </div>
                </div>

                {/* Mallin latausvirhe */}
                {modelStatus === 'error' && virhe && (
                    <div style={{ padding: '12px 16px', background: '#1a0a0a', border: '1px solid #e74c3c',
                        borderRadius: 4, color: '#e74c3c', fontSize: 12, marginBottom: 24, whiteSpace: 'pre-wrap' }}>
                        ✗ {virhe}
                    </div>
                )}

                {/* Valinnat */}
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 32 }}>
                    <div>
                        <label style={labelStyle}>Ravi</label>
                        <select value={valittuCardId} onChange={e => setValittuCardId(e.target.value)}
                                disabled={loadingRavit} style={selectStyle}>
                            <option value="">{loadingRavit ? 'Ladataan…' : '— Valitse ravit —'}</option>
                            {ravit.map(r => (
                                <option key={r.cardId} value={r.cardId}>{r.country} · {r.trackName}</option>
                            ))}
                        </select>
                    </div>

                    <div>
                        <label style={labelStyle}>Lähtö</label>
                        <select value={valittuLahto} onChange={e => setValittuLahto(e.target.value)}
                                disabled={!valittuCardId || loadingLahdot} style={{ ...selectStyle, minWidth: 160 }}>
                            <option value="">{loadingLahdot ? '…' : '—'}</option>
                            {lahdot.map(l => (
                                <option key={l.raceId} value={l.number}>Lähtö {l.number} · {l.distance}m</option>
                            ))}
                        </select>
                    </div>

                    <button onClick={ajaEnnustus} disabled={!canRun} style={{
                        padding: '10px 24px',
                        background: canRun ? '#4a90d9' : '#1e2330',
                        color:      canRun ? '#fff'    : '#444',
                        border: '1px solid #2a3040', borderRadius: 4,
                        fontFamily: 'inherit', fontSize: 13, letterSpacing: 1,
                        cursor: canRun ? 'pointer' : 'not-allowed', transition: 'all 0.2s',
                    }}>
                        {loadingEnnustus ? 'Lasketaan…' : '▶ Aja ennustus'}
                    </button>
                </div>

                {/* Ennustusvirhe */}
                {virhe && modelStatus !== 'error' && (
                    <div style={{ padding: '12px 16px', background: '#1a0a0a', border: '1px solid #e74c3c',
                        borderRadius: 4, color: '#e74c3c', fontSize: 13, marginBottom: 24 }}>
                        ✗ {virhe}
                    </div>
                )}

                {/* Tulokset */}
                {ennustukset.length > 0 && (
                    <div>
                        <div style={{ fontSize: 11, letterSpacing: 2, color: '#4a90d9',
                            textTransform: 'uppercase', marginBottom: 12 }}>
                            Ennuste · Lähtö {valittuLahto}
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                                <thead>
                                <tr style={{ borderBottom: '1px solid #1e2330' }}>
                                    {['#','Nro','Hevonen','Ohjastaja','Todennäköisyys','Kerroin','Arvio'].map(h => (
                                        <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#4a90d9',
                                            fontWeight: 500, fontSize: 11, letterSpacing: 1, textTransform: 'uppercase' }}>
                                            {h}
                                        </th>
                                    ))}
                                </tr>
                                </thead>
                                <tbody>
                                {ennustukset.map((e, i) => (
                                    <tr key={e.nimi} style={{ borderBottom: '1px solid #12151c',
                                        background: i === 0 ? '#0d1a2a' : i % 2 === 0 ? '#0f1118' : 'transparent' }}>
                                        <td style={{ padding: '10px 12px', color: i < 3 ? '#f0a500' : '#444' }}>
                                            {i < 3 ? ['①','②','③'][i] : `#${i+1}`}
                                        </td>
                                        <td style={{ padding: '10px 12px', color: '#666' }}>{e.nro}</td>
                                        <td style={{ padding: '10px 12px', fontWeight: 600 }}>{e.nimi}</td>
                                        <td style={{ padding: '10px 12px', color: '#aaa' }}>{e.ohjastaja || '—'}</td>
                                        <td style={{ padding: '10px 12px' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                                <div style={{ height: 4, minWidth: 4, width: `${Math.round(e.prob * 120)}px`,
                                                    background: e.prob > 0.5 ? '#2ecc71' : e.prob > 0.3 ? '#f0a500' : '#333',
                                                    borderRadius: 2 }} />
                                                <span>{(e.prob * 100).toFixed(1)}%</span>
                                            </div>
                                        </td>
                                        <td style={{ padding: '10px 12px', color: '#aaa', fontVariantNumeric: 'tabular-nums' }}>
                                            {(1 / e.prob).toFixed(2)}
                                        </td>
                                        <td style={{ padding: '10px 12px' }}>
                      <span style={{ padding: '2px 10px', borderRadius: 3, fontSize: 11, letterSpacing: 1,
                          background: e.prob > 0.5 ? '#0d2a1a' : '#111',
                          color:      e.prob > 0.5 ? '#2ecc71'  : '#444',
                          border: `1px solid ${e.prob > 0.5 ? '#2ecc71' : '#222'}` }}>
                        {e.prob > 0.5 ? 'PELATTAVA' : 'HUTI'}
                      </span>
                                        </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
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
