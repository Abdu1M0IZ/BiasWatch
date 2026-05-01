import { useState } from "react";

function HeroArt() {
  const [imageFailed, setImageFailed] = useState(false);

  return (
    <div className="hero-right">
      {!imageFailed ? (
        <img
          src="/robot-island.png"
          alt="BiasWatch robot assistant on floating island"
          className="robot-island"
          onError={() => setImageFailed(true)}
        />
      ) : (
        <div className="robot-placeholder" aria-label="Robot image placeholder">
          <div className="robot-placeholder-inner">
            <div className="placeholder-title">Robot + island image placeholder</div>
            <div className="placeholder-subtitle">
              Put your generated artwork at
            </div>
            <div className="placeholder-path">public/robot-island.png</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default HeroArt;