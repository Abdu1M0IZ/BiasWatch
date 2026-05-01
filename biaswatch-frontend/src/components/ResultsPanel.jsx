function ResultsPanel({ results, meta }) {
  if (!results || results.length === 0) {
    return (
      <section className="results-panel empty">
        <div className="results-header">
          <h2>Results</h2>
          <p>Your predictions will appear here after you submit a request.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="results-panel">
      <div className="results-header">
        <div>
          <h2>{meta?.title || "Prediction results"}</h2>
        </div>
      </div>

      <div className="results-grid">
        {results.map((item, index) => (
          <article className="result-card simple-result-card" key={index}>
            <div className="result-title">
              Classification: <span>{item.label_name}</span>
            </div>

            <div className="result-block">
              <p className="result-text">{item.cleaned_text}</p>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

export default ResultsPanel;