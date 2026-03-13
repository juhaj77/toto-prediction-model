import { useState, useEffect, useCallback } from "react";

const BASE_URL = "/api-veikkaus/api/toto-info/v1";

const fetchJSON = async (url) => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
};

const today = () => new Date().toISOString().slice(0, 10);

// Syntax-highlighted JSON renderer
function JsonViewer({ data }) {
    if (data === null) return null;

    const highlight = (json) => {
        return json
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(
                /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
                (match) => {
                    let cls = "json-number";
                    if (/^"/.test(match)) {
                        cls = /:$/.test(match) ? "json-key" : "json-string";
                    } else if (/true|false/.test(match)) {
                        cls = "json-bool";
                    } else if (/null/.test(match)) {
                        cls = "json-null";
                    }
                    return `<span class="${cls}">${match}</span>`;
                }
            );
    };

    const formatted = JSON.stringify(data, null, 2);
    return (
        <pre
            className="json-pre"
            dangerouslySetInnerHTML={{ __html: highlight(formatted) }}
        />
    );
}

function StatusBadge({ status, message }) {
    const colors = {
        idle: "#4a5568",
        loading: "#d4a017",
        success: "#38a169",
        error: "#e53e3e",
    };
    return (
        <span style={{ color: colors[status] || "#4a5568", fontSize: "0.78rem", fontFamily: "monospace" }}>
      {status === "loading" && "⟳ "}
            {status === "success" && "✓ "}
            {status === "error" && "✗ "}
            {message}
    </span>
    );
}

function Panel({ title, endpoint, data, status, message, onFetch, children }) {
    const [open, setOpen] = useState(true);
    return (
        <div className="panel">
            <div className="panel-header" onClick={() => setOpen((v) => !v)}>
                <span className="panel-toggle">{open ? "▾" : "▸"}</span>
                <span className="panel-title">{title}</span>
                <span className="panel-endpoint">{endpoint}</span>
                {onFetch && (
                    <button
                        className="fetch-btn"
                        onClick={(e) => { e.stopPropagation(); onFetch(); }}
                        disabled={status === "loading"}
                    >
                        {status === "loading" ? "Haetaan…" : "Hae"}
                    </button>
                )}
                {status && <StatusBadge status={status} message={message} />}
            </div>
            {open && (
                <div className="panel-body">
                    {children}
                    {data && <JsonViewer data={data} />}
                    {status === "idle" && !data && (
                        <div className="hint">Paina "Hae" noutaaksesi data</div>
                    )}
                </div>
            )}
        </div>
    );
}

export default function ViewRaceData() {
    const [dateStr, setDateStr] = useState(today());
    const [cards, setCards] = useState([]);
    const [selectedCard, setSelectedCard] = useState(null);
    const [races, setRaces] = useState([]);
    const [selectedRace, setSelectedRace] = useState(null);

    const [cardsNav, setCardsNav] = useState({ previous: null, next: null });
    const [cardsState, setCardsState] = useState({ status: "idle", data: null, msg: "" });
    const [cardInfoState, setCardInfoState] = useState({ status: "idle", data: null, msg: "" });
    const [racesState, setRacesState] = useState({ status: "idle", data: null, msg: "" });
    const [runnersState, setRunnersState] = useState({ status: "idle", data: null, msg: "" });
    const [raceInfoState, setRaceInfoState] = useState({ status: "idle", data: null, msg: "" });

    const resetCard = () => {
        setSelectedCard(null);
        setCards([]);
        setRaces([]);
        setSelectedRace(null);
        setCardInfoState({ status: "idle", data: null, msg: "" });
        setRacesState({ status: "idle", data: null, msg: "" });
        setRunnersState({ status: "idle", data: null, msg: "" });
        setRaceInfoState({ status: "idle", data: null, msg: "" });
    };

    const fetchCards = useCallback(async () => {
        resetCard();
        setCardsState({ status: "loading", data: null, msg: "Haetaan kortteja…" });
        try {
            const d = await fetchJSON(`${BASE_URL}/cards/date/${dateStr}`);
            const list = d.collection || d.cards || [];
            // Extract dates from nav links like /api/.../cards/date/2026-03-25
            const extractDate = (url) => url ? (url.match(/(\d{4}-\d{2}-\d{2})/) || [])[1] : null;
            setCardsNav({ previous: extractDate(d.previous), next: extractDate(d.next) });
            setCards(list);
            if (list.length === 0) {
                setCardsState({ status: "error", data: d, msg: `Ei kortteja ${dateStr} — kokeile eri päivää` });
            } else {
                setCardsState({ status: "success", data: d, msg: `${list.length} korttia` });
            }
        } catch (e) {
            setCardsState({ status: "error", data: null, msg: e.message });
        }
    }, [dateStr]);

    const fetchCardInfo = useCallback(async () => {
        if (!selectedCard) return;
        setCardInfoState({ status: "loading", data: null, msg: "" });
        try {
            const d = await fetchJSON(`${BASE_URL}/card/${selectedCard}`);
            setCardInfoState({ status: "success", data: d, msg: "OK" });
        } catch (e) {
            setCardInfoState({ status: "error", data: null, msg: e.message });
        }
    }, [selectedCard]);

    const fetchRaces = useCallback(async () => {
        if (!selectedCard) return;
        setRacesState({ status: "loading", data: null, msg: "" });
        setRaces([]);
        setSelectedRace(null);
        try {
            const d = await fetchJSON(`${BASE_URL}/card/${selectedCard}/races`);
            const list = d.collection || d.races || [];
            setRaces(list);
            setRacesState({ status: "success", data: d, msg: `${list.length} lähtöä` });
        } catch (e) {
            setRacesState({ status: "error", data: null, msg: e.message });
        }
    }, [selectedCard]);

    const fetchRunners = useCallback(async () => {
        if (!selectedRace) return;
        setRunnersState({ status: "loading", data: null, msg: "" });
        try {
            const d = await fetchJSON(`${BASE_URL}/race/${selectedRace}/runners`);
            setRunnersState({ status: "success", data: d, msg: "OK" });
        } catch (e) {
            setRunnersState({ status: "error", data: null, msg: e.message });
        }
    }, [selectedRace]);

    const fetchRaceInfo = useCallback(async () => {
        if (!selectedRace) return;
        setRaceInfoState({ status: "loading", data: null, msg: "" });
        try {
            const d = await fetchJSON(`${BASE_URL}/race/${selectedRace}`);
            setRaceInfoState({ status: "success", data: d, msg: "OK" });
        } catch (e) {
            setRaceInfoState({ status: "error", data: null, msg: e.message });
        }
    }, [selectedRace]);

    const cardLabel = (c) => {
        const id = String(c.cardId || c.id || "?");
        const track =
            c.trackName ||
            c.venue?.name ||
            c.track?.name ||
            c.name ||
            c.racecourse ||
            id;
        return `${id} — ${track}`;
    };

    const raceLabel = (r) => {
        const id = String(r.raceId || r.id || "?");
        const num = r.raceNumber || r.number || r.raceNum || id;
        const name = r.raceName || r.name || "";
        return `Lähtö ${num}${name ? " – " + name : ""} (${id})`;
    };

    return (
        <>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body { background: #0d1117; }

        .app {
          min-height: 100vh;
          background: #0d1117;
          color: #c9d1d9;
          font-family: 'IBM Plex Sans', sans-serif;
          padding: 24px 16px 48px;
        }

        .header {
          display: flex;
          align-items: baseline;
          gap: 12px;
          margin-bottom: 28px;
          border-bottom: 1px solid #21262d;
          padding-bottom: 16px;
        }
        .header h1 {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 1.25rem;
          font-weight: 600;
          color: #58a6ff;
          letter-spacing: -0.5px;
        }
        .header .sub {
          font-size: 0.78rem;
          color: #6e7681;
          font-family: 'IBM Plex Mono', monospace;
        }

        .controls {
          display: grid;
          grid-template-columns: auto auto 1fr auto;
          align-items: end;
          gap: 12px;
          margin-bottom: 24px;
          padding: 16px;
          background: #161b22;
          border: 1px solid #21262d;
          border-radius: 8px;
        }

        .field { display: flex; flex-direction: column; gap: 5px; }
        .field label {
          font-size: 0.7rem;
          font-family: 'IBM Plex Mono', monospace;
          color: #6e7681;
          text-transform: uppercase;
          letter-spacing: 0.8px;
        }
        .field input[type="date"],
        .field select {
          background: #0d1117;
          border: 1px solid #30363d;
          color: #c9d1d9;
          padding: 7px 10px;
          border-radius: 6px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.82rem;
          outline: none;
          transition: border-color 0.15s;
          min-width: 160px;
        }
        .field input[type="date"]:focus,
        .field select:focus { border-color: #58a6ff; }
        .field select:disabled { opacity: 0.4; cursor: not-allowed; }

        .go-btn {
          padding: 8px 20px;
          background: #238636;
          color: #fff;
          border: none;
          border-radius: 6px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.82rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
          white-space: nowrap;
        }
        .go-btn:hover { background: #2ea043; }
        .go-btn:disabled { background: #21262d; color: #6e7681; cursor: not-allowed; }

        .panels { display: flex; flex-direction: column; gap: 10px; }

        .panel {
          background: #161b22;
          border: 1px solid #21262d;
          border-radius: 8px;
          overflow: hidden;
        }

        .panel-header {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          cursor: pointer;
          user-select: none;
          background: #1c2128;
          border-bottom: 1px solid #21262d;
          flex-wrap: wrap;
        }
        .panel-header:hover { background: #22272e; }
        .panel-toggle { color: #58a6ff; font-size: 0.9rem; width: 12px; }
        .panel-title {
          font-weight: 600;
          font-size: 0.88rem;
          color: #e6edf3;
          min-width: 140px;
        }
        .panel-endpoint {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          color: #6e7681;
          flex: 1;
        }

        .fetch-btn {
          padding: 4px 14px;
          background: #1f6feb;
          color: #fff;
          border: none;
          border-radius: 5px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
        }
        .fetch-btn:hover { background: #388bfd; }
        .fetch-btn:disabled { background: #21262d; color: #6e7681; cursor: not-allowed; }

        .panel-body { padding: 14px; }

        .hint {
          color: #6e7681;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          padding: 8px 0;
        }

        .json-pre {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.77rem;
          line-height: 1.55;
          background: #0d1117;
          border: 1px solid #21262d;
          border-radius: 6px;
          padding: 14px;
          overflow: auto;
          max-height: 480px;
          white-space: pre;
        }

        .json-key    { color: #79c0ff; }
        .json-string { color: #a5d6ff; }
        .json-number { color: #f2cc60; }
        .json-bool   { color: #ff7b72; }
        .json-null   { color: #8b949e; }

        .section-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.68rem;
          text-transform: uppercase;
          letter-spacing: 1px;
          color: #6e7681;
          margin: 20px 0 8px;
          padding-left: 2px;
        }

        .nav-hint {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 10px 14px;
          background: #1c2128;
          border: 1px solid #30363d;
          border-radius: 8px;
          margin-bottom: 16px;
          flex-wrap: wrap;
        }
        .nav-hint-text {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          color: #d4a017;
        }
        .nav-btn {
          padding: 5px 14px;
          background: #21262d;
          color: #58a6ff;
          border: 1px solid #30363d;
          border-radius: 6px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
        }
        .nav-btn:hover { background: #30363d; }

        @media (max-width: 600px) {
          .controls { grid-template-columns: 1fr 1fr; }
          .go-btn { grid-column: 1 / -1; }
        }
      `}</style>

            <div className="app">
                <div className="header">
                    <h1>🐎 ViewRaceData</h1>
                    <span className="sub">veikkaus.fi / toto-info / v1</span>
                </div>

                <div className="controls">
                    <div className="field">
                        <label>Päivämäärä</label>
                        <input
                            type="date"
                            value={dateStr}
                            onChange={(e) => setDateStr(e.target.value)}
                        />
                    </div>

                    <div className="field">
                        <label>Kortti (Card)</label>
                        <select
                            value={selectedCard || ""}
                            onChange={(e) => {
                                setSelectedCard(e.target.value || null);
                                setRaces([]);
                                setSelectedRace(null);
                                setCardInfoState({ status: "idle", data: null, msg: "" });
                                setRacesState({ status: "idle", data: null, msg: "" });
                                setRunnersState({ status: "idle", data: null, msg: "" });
                                setRaceInfoState({ status: "idle", data: null, msg: "" });
                            }}
                            disabled={cards.length === 0}
                        >
                            <option value="">— Valitse kortti —</option>
                            {cards.map((c) => {
                                const id = String(c.cardId || c.id);
                                return (
                                    <option key={id} value={id}>
                                        {cardLabel(c)}
                                    </option>
                                );
                            })}
                        </select>
                    </div>

                    <div className="field">
                        <label>Lähtö (Race)</label>
                        <select
                            value={selectedRace || ""}
                            onChange={(e) => {
                                setSelectedRace(e.target.value || null);
                                setRunnersState({ status: "idle", data: null, msg: "" });
                                setRaceInfoState({ status: "idle", data: null, msg: "" });
                            }}
                            disabled={races.length === 0}
                        >
                            <option value="">— Valitse lähtö —</option>
                            {races.map((r) => {
                                const id = String(r.raceId || r.id);
                                return (
                                    <option key={id} value={id}>
                                        {raceLabel(r)}
                                    </option>
                                );
                            })}
                        </select>
                    </div>

                    <button className="go-btn" onClick={fetchCards} disabled={!dateStr}>
                        Hae kortit
                    </button>
                </div>

                {cardsState.status === "error" && cardsNav.previous && (
                    <div className="nav-hint">
                        <span className="nav-hint-text">⬡ Ei raveja tällä päivällä. Lähimmät:</span>
                        {cardsNav.previous && (
                            <button className="nav-btn" onClick={() => { setDateStr(cardsNav.previous); }}>
                                ← {cardsNav.previous}
                            </button>
                        )}
                        {cardsNav.next && (
                            <button className="nav-btn" onClick={() => { setDateStr(cardsNav.next); }}>
                                {cardsNav.next} →
                            </button>
                        )}
                    </div>
                )}

                <div className="panels">
                    <div className="section-label">📋 Korttitaso</div>

                    <Panel
                        title="Kortit / päivä"
                        endpoint={`/cards/date/${dateStr}`}
                        data={cardsState.data}
                        status={cardsState.status}
                        message={cardsState.msg}
                    />

                    <Panel
                        title="Kortin info"
                        endpoint={selectedCard ? `/card/${selectedCard}` : "/card/{cardId}"}
                        data={cardInfoState.data}
                        status={cardInfoState.status || (selectedCard ? "idle" : undefined)}
                        message={cardInfoState.msg}
                        onFetch={selectedCard ? fetchCardInfo : undefined}
                    />

                    <Panel
                        title="Lähdöt kortissa"
                        endpoint={selectedCard ? `/card/${selectedCard}/races` : "/card/{cardId}/races"}
                        data={racesState.data}
                        status={racesState.status || (selectedCard ? "idle" : undefined)}
                        message={racesState.msg}
                        onFetch={selectedCard ? fetchRaces : undefined}
                    />

                    <div className="section-label">🏁 Lähtötaso</div>

                    <Panel
                        title="Lähdön info"
                        endpoint={selectedRace ? `/race/${selectedRace}` : "/race/{raceId}"}
                        data={raceInfoState.data}
                        status={raceInfoState.status || (selectedRace ? "idle" : undefined)}
                        message={raceInfoState.msg}
                        onFetch={selectedRace ? fetchRaceInfo : undefined}
                    />

                    <Panel
                        title="Osallistujat"
                        endpoint={selectedRace ? `/race/${selectedRace}/runners` : "/race/{raceId}/runners"}
                        data={runnersState.data}
                        status={runnersState.status || (selectedRace ? "idle" : undefined)}
                        message={runnersState.msg}
                        onFetch={selectedRace ? fetchRunners : undefined}
                    />
                </div>
            </div>
        </>
    );
}
