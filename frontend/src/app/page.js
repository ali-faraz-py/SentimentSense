"use client";

import { useState } from "react";

const ASPECT_COLORS = {
  Food: { bg: "#FFF3D6", text: "#8A5A00", dot: "#E8A400" },
  Service: { bg: "#DCEAFB", text: "#1E4E8C", dot: "#3B82F6" },
  Price: { bg: "#DDF3E4", text: "#1E6B3A", dot: "#22A05A" },
  Atmosphere: { bg: "#F1E4FB", text: "#6B2E9C", dot: "#A855F7" },
  Health: { bg: "#DCF6F2", text: "#106B60", dot: "#14B8A6" },
};

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

      const response = await fetch(`${apiUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Analysis failed. Please try again.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isPositive = result?.sentiment?.label === "POSITIVE";

  const orbGradient = isPositive
    ? "radial-gradient(circle at 35% 30%, #FFE29A, #F5A623 55%, #D97706 100%)"
    : "radial-gradient(circle at 35% 30%, #C7B6FF, #7C5CE0 55%, #4C2A9E 100%)";

  return (
    <main className="min-h-screen bg-cream flex flex-col items-center px-6 py-16">
      <div className="w-full max-w-xl">
        <p className="font-mono text-[11px] tracking-[0.25em] uppercase text-fadedink text-center mb-3">
          Aspect-Based Sentiment Analysis
        </p>
        <h1 className="font-display italic text-4xl sm:text-5xl text-center text-ink leading-tight">
          What are you really saying?
        </h1>
        <p className="font-body text-center text-fadedink mt-4 max-w-md mx-auto">
          Paste a review below. This finds what it's about — food, service,
          price — and how the writer actually feels about each.
        </p>

        <div className="mt-10 bg-white rounded-2xl border border-line shadow-sm overflow-hidden">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="The pizza was great but the service was slow..."
            rows={5}
            className="paper-lines w-full p-6 font-body text-[15px] text-ink placeholder-fadedink/60 resize-none outline-none"
          />
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!text.trim() || loading}
          className="mt-5 w-full py-3.5 bg-ink text-cream font-body font-medium rounded-xl disabled:opacity-40 disabled:cursor-not-allowed hover:opacity-90 transition cursor-pointer"
        >
          {loading ? "Reading between the lines..." : "Analyze"}
        </button>

        {error && (
          <p className="mt-4 text-sm text-red-600 text-center">{error}</p>
        )}

        {result && (
          <div className="mt-12 flex flex-col items-center">
            <div className="relative w-36 h-36 flex items-center justify-center">
              <div
                className="orb-pulse absolute inset-0 rounded-full"
                style={{ background: orbGradient, filter: "blur(2px)" }}
              />
              <div
                className="orb-spin absolute inset-0 rounded-full border-2 border-dashed opacity-30"
                style={{ borderColor: isPositive ? "#D97706" : "#4C2A9E" }}
              />
              <div className="relative text-center">
                <p className="font-display text-3xl font-semibold text-white drop-shadow-sm">
                  {(result.sentiment.confidence * 100).toFixed(0)}%
                </p>
              </div>
            </div>

            <p className="font-display italic text-xl mt-4">
              {isPositive ? "Positive" : "Negative"}
            </p>

            {result.aspects.length > 0 && (
              <div className="mt-8 w-full">
                <p className="font-mono text-[11px] tracking-[0.2em] uppercase text-fadedink text-center mb-4">
                  Aspects Detected
                </p>
                <div className="flex flex-wrap justify-center gap-2.5">
                  {result.aspects.map((a) => {
                    const colors = ASPECT_COLORS[a.label] || {
                      bg: "#EEE",
                      text: "#333",
                      dot: "#999",
                    };
                    return (
                      <div
                        key={a.label}
                        className="flex items-center gap-2 px-4 py-2 rounded-full font-body text-sm font-medium"
                        style={{ backgroundColor: colors.bg, color: colors.text }}
                      >
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: colors.dot }}
                        />
                        {a.label}
                        <span className="opacity-60 text-xs">
                          {(a.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        <p className="mt-16 text-xs text-center text-fadedink">
          Powered by DistilBERT + zero-shot classification · Educational demo
        </p>
      </div>
    </main>
  );
}