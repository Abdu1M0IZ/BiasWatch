import { useMemo, useRef, useState } from "react";
import HeroArt from "./components/HeroArt.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const MODES = {
  SINGLE: "single",
  BATCH: "batch",
  CSV: "csv",
};

function EyeIcon({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 120 72"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="120" height="72" fill="none" />

      <rect x="16" y="32" width="8" height="8" fill="#284827" />
      <rect x="24" y="24" width="8" height="8" fill="#284827" />
      <rect x="32" y="16" width="16" height="8" fill="#284827" />
      <rect x="48" y="8" width="24" height="8" fill="#284827" />
      <rect x="72" y="16" width="16" height="8" fill="#284827" />
      <rect x="88" y="24" width="8" height="8" fill="#284827" />
      <rect x="96" y="32" width="8" height="8" fill="#284827" />

      <rect x="24" y="40" width="8" height="8" fill="#284827" />
      <rect x="32" y="48" width="16" height="8" fill="#284827" />
      <rect x="48" y="56" width="24" height="8" fill="#284827" />
      <rect x="72" y="48" width="16" height="8" fill="#284827" />
      <rect x="88" y="40" width="8" height="8" fill="#284827" />

      <rect x="48" y="24" width="24" height="24" fill="#284827" />
      <rect x="56" y="32" width="8" height="8" fill="#f7f3e7" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      width="48"
      height="48"
      viewBox="0 0 48 48"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="48" height="48" fill="none" />

      <rect x="10" y="32" width="28" height="4" fill="#284827" />
      <rect x="10" y="28" width="4" height="8" fill="#284827" />
      <rect x="34" y="28" width="4" height="8" fill="#284827" />

      <rect x="22" y="12" width="4" height="18" fill="#284827" />

      <rect x="18" y="16" width="4" height="4" fill="#284827" />
      <rect x="26" y="16" width="4" height="4" fill="#284827" />
      <rect x="14" y="20" width="4" height="4" fill="#284827" />
      <rect x="30" y="20" width="4" height="4" fill="#284827" />
    </svg>
  );
}

function SendArrowIcon() {
  return (
    <svg
      width="48"
      height="48"
      viewBox="0 0 48 48"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="48" height="48" fill="none" />

      <rect x="14" y="10" width="6" height="6" fill="#fffaf0" />
      <rect x="20" y="16" width="6" height="6" fill="#fffaf0" />
      <rect x="26" y="22" width="6" height="6" fill="#fffaf0" />
      <rect x="20" y="28" width="6" height="6" fill="#fffaf0" />
      <rect x="14" y="34" width="6" height="6" fill="#fffaf0" />
    </svg>
  );
}

function SunIcon({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 96 96"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="96" height="96" fill="none" />

      <rect x="36" y="8" width="24" height="8" fill="#f1c654" />
      <rect x="24" y="16" width="48" height="8" fill="#f1c654" />
      <rect x="16" y="24" width="64" height="8" fill="#f1c654" />
      <rect x="8" y="32" width="80" height="32" fill="#f1c654" />
      <rect x="16" y="64" width="64" height="8" fill="#f1c654" />
      <rect x="24" y="72" width="48" height="8" fill="#f1c654" />
      <rect x="36" y="80" width="24" height="8" fill="#f1c654" />

      <rect x="24" y="24" width="48" height="8" fill="#ffde86" />
      <rect x="16" y="32" width="64" height="16" fill="#ffde86" />
      <rect x="24" y="48" width="48" height="16" fill="#ffde86" />
    </svg>
  );
}

function CloudOneIcon({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 140 70"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="140" height="70" fill="none" />

      <rect x="30" y="34" width="72" height="8" fill="#d8ded0" />
      <rect x="20" y="42" width="92" height="8" fill="#d8ded0" />
      <rect x="38" y="26" width="20" height="8" fill="#d8ded0" />
      <rect x="58" y="18" width="22" height="8" fill="#d8ded0" />
      <rect x="80" y="26" width="18" height="8" fill="#d8ded0" />
    </svg>
  );
}

function CloudTwoIcon({ className = "" }) {
  return (
    <svg
      className={className}
      viewBox="0 0 110 60"
      xmlns="http://www.w3.org/2000/svg"
      shapeRendering="crispEdges"
      aria-hidden="true"
    >
      <rect width="110" height="60" fill="none" />

      <rect x="20" y="28" width="56" height="8" fill="#d8ded0" />
      <rect x="12" y="36" width="72" height="8" fill="#d8ded0" />
      <rect x="34" y="20" width="18" height="8" fill="#d8ded0" />
      <rect x="52" y="12" width="18" height="8" fill="#d8ded0" />
    </svg>
  );
}

function App() {
  const [mode, setMode] = useState(MODES.SINGLE);
  const [singleText, setSingleText] = useState("");
  const [batchText, setBatchText] = useState("");
  const [csvFile, setCsvFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState([]);
  const [resultMeta, setResultMeta] = useState(null);

  const fileInputRef = useRef(null);

  const batchTexts = useMemo(() => {
    return batchText
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
  }, [batchText]);

  const canSubmit = useMemo(() => {
    if (mode === MODES.SINGLE) return singleText.trim().length >= 3;
    if (mode === MODES.BATCH) return batchTexts.length > 0;
    if (mode === MODES.CSV) return Boolean(csvFile);
    return false;
  }, [mode, singleText, batchTexts, csvFile]);

  const clearFeedback = () => {
    setError("");
    setResults([]);
    setResultMeta(null);
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setError("");
  };

  const handleUploadButtonClick = () => {
    setMode(MODES.CSV);
    fileInputRef.current?.click();
  };

  const handleCsvChange = (event) => {
    const file = event.target.files?.[0] || null;
    setCsvFile(file);
    setMode(MODES.CSV);
    setError("");
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    clearFeedback();
    setLoading(true);

    try {
      if (mode === MODES.SINGLE) {
        await handleSinglePrediction();
      } else if (mode === MODES.BATCH) {
        await handleBatchPrediction();
      } else if (mode === MODES.CSV) {
        await handleCsvPrediction();
      }
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const handleSinglePrediction = async () => {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: singleText.trim() }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Single prediction failed.");
    }

    setResults([data]);
    setResultMeta({
      mode: MODES.SINGLE,
      count: 1,
      title: "Prediction result",
    });
  };

  const handleBatchPrediction = async () => {
    const response = await fetch(`${API_BASE_URL}/batch-predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ texts: batchTexts }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Batch prediction failed.");
    }

    setResults(data.predictions || []);
    setResultMeta({
      mode: MODES.BATCH,
      count: data.count || 0,
      title: "Prediction results",
    });
  };

  const handleCsvPrediction = async () => {
    if (!csvFile) {
      throw new Error("Please select a CSV file first.");
    }

    const formData = new FormData();
    formData.append("file", csvFile);

    const response = await fetch(`${API_BASE_URL}/predict-csv`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "CSV prediction failed.");
    }

    setResults(data.predictions || []);
    setResultMeta({
      mode: MODES.CSV,
      count: data.count || 0,
      title: "CSV prediction results",
    });
  };

  return (
    <div className="page-shell">
      <main className="page">
        <section className="hero">
          <SunIcon className="decor sun" />
          <CloudOneIcon className="decor cloud cloud-1" />
          <CloudTwoIcon className="decor cloud cloud-2" />
          <CloudOneIcon className="decor cloud cloud-3" />

          <div className="hero-content">
            <div className="hero-left">
              <div
                className="mode-switch"
                role="tablist"
                aria-label="Prediction mode"
              >
                <button
                  type="button"
                  className={
                    mode === MODES.SINGLE ? "mode-pill active" : "mode-pill"
                  }
                  onClick={() => handleModeChange(MODES.SINGLE)}
                >
                  Single
                </button>
                <button
                  type="button"
                  className={
                    mode === MODES.BATCH ? "mode-pill active" : "mode-pill"
                  }
                  onClick={() => handleModeChange(MODES.BATCH)}
                >
                  Multiple
                </button>
                <button
                  type="button"
                  className={
                    mode === MODES.CSV ? "mode-pill active" : "mode-pill"
                  }
                  onClick={() => handleModeChange(MODES.CSV)}
                >
                  CSV
                </button>
              </div>

              <h1 className="logo" aria-label="BIASWATCH">
                <span className="logo-line logo-line-top">BIAS</span>
                <span className="logo-line logo-line-bottom">
                  <span className="logo-letter">W</span>
                  <EyeIcon className="logo-eye" />
                  <span className="logo-letter">TCH</span>
                </span>
              </h1>

              <p className="tagline">
                Making moderation easier with AI assistance.
              </p>

              <form className="prompt-box" onSubmit={handleSubmit}>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  className="hidden-file-input"
                  onChange={handleCsvChange}
                />

                <button
                  type="button"
                  className="upload-btn"
                  onClick={handleUploadButtonClick}
                  aria-label="Upload CSV"
                  title="Upload CSV"
                >
                  <UploadIcon />
                </button>

                <div className="prompt-input-area">
                  {mode === MODES.SINGLE && (
                    <input
                      type="text"
                      placeholder="Ask BiasWatch anything..."
                      value={singleText}
                      onChange={(e) => setSingleText(e.target.value)}
                    />
                  )}

                  {mode === MODES.BATCH && (
                    <textarea
                      className="prompt-textarea"
                      placeholder={
                        "Enter one text per line...\nExample:\nyou are stupid\nhope you have a nice day"
                      }
                      value={batchText}
                      onChange={(e) => setBatchText(e.target.value)}
                    />
                  )}

                  {mode === MODES.CSV && (
                    <div className="csv-panel">
                      <div className="csv-message">
                        {csvFile
                          ? `Selected file: ${csvFile.name}`
                          : "Upload a CSV file to run bulk predictions."}
                      </div>
                      <div className="csv-hint">
                        Supported columns: tweet, text, cleaned_tweet, content,
                        message.
                      </div>
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  className="send-btn"
                  disabled={!canSubmit || loading}
                  aria-label="Run prediction"
                  title="Run prediction"
                >
                  {loading ? (
                    <span className="loading-text">...</span>
                  ) : (
                    <SendArrowIcon />
                  )}
                </button>
              </form>

              {error && <div className="error-box">{error}</div>}
            </div>

            <HeroArt />
          </div>
        </section>

        <ResultsPanel results={results} meta={resultMeta} />
      </main>
    </div>
  );
}

export default App;
