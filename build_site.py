"""Generate a static dashboard HTML from analysis results."""

import json
import sys
from datetime import datetime
from pathlib import Path


def build_site(results_path: str = "results/latest.json", output_dir: str = "_site"):
    with open(results_path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    market = data.get("market_overview", {})
    candidates = data.get("strong_candidates", [])
    alpha = data.get("alpha_rankings", [])
    trends = data.get("trends", [])
    momentum = data.get("momentum", [])
    breakouts = data.get("breakouts", [])

    date_str = meta.get("analysis_date", "")[:10]
    sentiment = market.get("sentiment", "N/A")
    bullish_pct = market.get("bullish_pct", 0)
    num_symbols = meta.get("symbols_analyzed", 0)

    # Determine sentiment color
    if sentiment == "BULLISH":
        sentiment_color = "#22c55e"
    elif sentiment == "BEARISH":
        sentiment_color = "#ef4444"
    else:
        sentiment_color = "#eab308"

    # Build top picks rows
    picks_rows = ""
    for i, c in enumerate(candidates[:10], 1):
        trend = c.get("trend_direction", "")
        trend_cls = "up" if "up" in trend.lower() else "down" if "down" in trend.lower() else "neutral"
        mom = c.get("momentum_score", 0)
        mom_cls = "up" if mom > 0 else "down" if mom < 0 else "neutral"
        picks_rows += f"""<tr>
            <td>{i}</td>
            <td class="symbol">{c.get('symbol', '')}</td>
            <td class="trend-{trend_cls}">{trend}</td>
            <td class="{mom_cls}">{mom:+.2f}</td>
            <td><strong>{c.get('composite_score', 0):+.3f}</strong></td>
        </tr>"""

    # Build full alpha rankings rows
    alpha_rows = ""
    for r in alpha[:20]:
        score = r.get("composite_score", 0)
        pct = r.get("percentile", 0)
        score_cls = "up" if score > 0 else "down" if score < 0 else "neutral"
        alpha_rows += f"""<tr>
            <td class="symbol">{r.get('symbol', '')}</td>
            <td class="{score_cls}">{score:+.3f}</td>
            <td>{pct:.0f}</td>
        </tr>"""

    # Count trend directions
    uptrends = sum(1 for t in trends if "up" in t.get("trend_direction", "").lower())
    downtrends = sum(1 for t in trends if "down" in t.get("trend_direction", "").lower())
    neutral_trends = len(trends) - uptrends - downtrends

    # Count breakouts
    num_breakouts = sum(1 for b in breakouts if b.get("has_breakout", False))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Analysis - {date_str}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 2rem 1rem;
        }}
        .container {{ max-width: 960px; margin: 0 auto; }}
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid #1e293b;
        }}
        h1 {{ font-size: 1.5rem; font-weight: 600; color: #f8fafc; }}
        .date {{ color: #94a3b8; margin-top: 0.25rem; font-size: 0.9rem; }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: #1e293b;
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .card-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
        .card-value {{ font-size: 1.5rem; font-weight: 700; margin-top: 0.25rem; }}
        h2 {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #f8fafc;
        }}
        .section {{ margin-bottom: 2rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        th {{
            text-align: left;
            padding: 0.6rem 0.75rem;
            border-bottom: 2px solid #334155;
            color: #94a3b8;
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        td {{
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid #1e293b;
        }}
        tr:hover {{ background: #1e293b; }}
        .symbol {{ font-weight: 600; color: #f8fafc; }}
        .up {{ color: #22c55e; }}
        .down {{ color: #ef4444; }}
        .neutral {{ color: #eab308; }}
        .trend-up {{ color: #22c55e; }}
        .trend-down {{ color: #ef4444; }}
        .trend-neutral {{ color: #94a3b8; }}
        footer {{
            text-align: center;
            color: #475569;
            font-size: 0.75rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Market Analysis</h1>
            <div class="date">{date_str}</div>
        </header>

        <div class="cards">
            <div class="card">
                <div class="card-label">Sentiment</div>
                <div class="card-value" style="color: {sentiment_color}">{sentiment}</div>
            </div>
            <div class="card">
                <div class="card-label">Bullish</div>
                <div class="card-value">{bullish_pct:.1f}%</div>
            </div>
            <div class="card">
                <div class="card-label">Uptrends</div>
                <div class="card-value up">{uptrends}</div>
            </div>
            <div class="card">
                <div class="card-label">Downtrends</div>
                <div class="card-value down">{downtrends}</div>
            </div>
            <div class="card">
                <div class="card-label">Breakouts</div>
                <div class="card-value">{num_breakouts}</div>
            </div>
        </div>

        <div class="section">
            <h2>Top Picks</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Trend</th>
                        <th>Momentum</th>
                        <th>Alpha</th>
                    </tr>
                </thead>
                <tbody>{picks_rows}</tbody>
            </table>
        </div>

        <div class="section">
            <h2>Alpha Rankings</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Score</th>
                        <th>Percentile</th>
                    </tr>
                </thead>
                <tbody>{alpha_rows}</tbody>
            </table>
        </div>

        <footer>
            {num_symbols} stocks analyzed &middot; Updated {date_str}
        </footer>
    </div>
</body>
</html>"""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    print(f"Site built: {out / 'index.html'}")


if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/latest.json"
    build_site(results_path)
