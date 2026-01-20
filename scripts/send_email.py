#!/usr/bin/env python3
"""
Send daily market research report via email.
"""
import os
import sys
import json
import glob
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path


def get_latest_results():
    """Find the most recent analysis results."""
    results_dir = Path(__file__).parent.parent / "results"

    # Find all JSON analysis files
    json_files = list(results_dir.glob("market_analysis_*.json"))

    if not json_files:
        return None, None

    # Get the most recent file
    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)

    # Find corresponding CSV file
    timestamp = latest_json.stem.replace("market_analysis_", "")
    csv_file = results_dir / f"alpha_rankings_{timestamp}.csv"

    return latest_json, csv_file if csv_file.exists() else None


def format_email_body(analysis_data):
    """Format the analysis results into an HTML email body."""
    metadata = analysis_data.get("metadata", {})
    market = analysis_data.get("market_overview", {})
    trends = analysis_data.get("trends", [])
    momentum = analysis_data.get("momentum", [])
    breakouts = analysis_data.get("breakouts", [])
    alpha = analysis_data.get("alpha_rankings", [])
    candidates = analysis_data.get("strong_candidates", [])

    # Create HTML email
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 30px; }}
            .metric {{ background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .metric strong {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .bullish {{ color: #27ae60; font-weight: bold; }}
            .bearish {{ color: #e74c3c; font-weight: bold; }}
            .neutral {{ color: #95a5a6; font-weight: bold; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 2px solid #ecf0f1; color: #7f8c8d; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>📊 Daily Market Research Report</h1>

        <div class="metric">
            <strong>Analysis Date:</strong> {metadata.get('analysis_date', 'N/A')}<br>
            <strong>Symbols Analyzed:</strong> {metadata.get('symbols_analyzed', 'N/A')}<br>
            <strong>Lookback Period:</strong> {metadata.get('lookback_days', 'N/A')} days
        </div>

        <h2>🌐 Market Overview</h2>
        <div class="metric">
            <strong>Market Sentiment:</strong> <span class="{market.get('sentiment', 'neutral').lower()}">{market.get('sentiment', 'N/A')}</span><br>
            <strong>Bullish Percentage:</strong> {market.get('bullish_percentage', 0):.1f}%<br>
            <strong>Bearish Percentage:</strong> {market.get('bearish_percentage', 0):.1f}%<br>
            <strong>Neutral Percentage:</strong> {market.get('neutral_percentage', 0):.1f}%
        </div>

        <h2>🚀 Top Momentum Stocks (Top 10)</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Momentum Score</th>
                <th>RSI</th>
            </tr>
    """

    for stock in momentum[:10]:
        html += f"""
            <tr>
                <td><strong>{stock.get('symbol', 'N/A')}</strong></td>
                <td>{stock.get('momentum_score', 0):.4f}</td>
                <td>{stock.get('rsi', 0):.2f}</td>
            </tr>
        """

    html += """
        </table>

        <h2>💎 Breakout Opportunities (Top 10)</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Active Breakouts</th>
                <th>Potential</th>
            </tr>
    """

    for stock in breakouts[:10]:
        potential = "HIGH" if stock.get('high_potential', False) else "MEDIUM"
        html += f"""
            <tr>
                <td><strong>{stock.get('symbol', 'N/A')}</strong></td>
                <td>{stock.get('breakout_count', 0)}</td>
                <td>{potential}</td>
            </tr>
        """

    html += """
        </table>

        <h2>⭐ Top Alpha Ranked Stocks (Top 10)</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Composite Score</th>
                <th>Percentile</th>
            </tr>
    """

    for stock in alpha[:10]:
        html += f"""
            <tr>
                <td><strong>{stock.get('symbol', 'N/A')}</strong></td>
                <td>{stock.get('composite_score', 0):.4f}</td>
                <td>{stock.get('percentile', 0):.1f}%</td>
            </tr>
        """

    html += """
        </table>

        <h2>🎯 Strong Buy Candidates</h2>
    """

    if candidates:
        html += """
        <table>
            <tr>
                <th>Symbol</th>
                <th>Trend</th>
                <th>Momentum</th>
                <th>Alpha Score</th>
            </tr>
        """
        for stock in candidates[:15]:
            html += f"""
            <tr>
                <td><strong>{stock.get('symbol', 'N/A')}</strong></td>
                <td>{stock.get('trend_direction', 'N/A')}</td>
                <td>{stock.get('momentum_score', 0):.4f}</td>
                <td>{stock.get('alpha_score', 0):.4f}</td>
            </tr>
            """
        html += "</table>"
    else:
        html += "<p>No strong buy candidates identified in current market conditions.</p>"

    html += """
        <div class="footer">
            <p>This is an automated daily market research report generated using quantitative analysis.</p>
            <p>Full analysis results are attached as JSON and CSV files.</p>
            <p><strong>Disclaimer:</strong> This report is for informational purposes only and should not be considered investment advice.</p>
        </div>
    </body>
    </html>
    """

    return html


def send_email(smtp_host, smtp_port, from_addr, to_addrs, password, subject, body_html, attachments=None):
    """Send email with attachments."""
    msg = MIMEMultipart()
    msg['From'] = from_addr
    msg['To'] = to_addrs if isinstance(to_addrs, str) else ', '.join(to_addrs)
    msg['Subject'] = subject

    # Attach HTML body
    msg.attach(MIMEText(body_html, 'html'))

    # Attach files
    if attachments:
        for file_path in attachments:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename={os.path.basename(file_path)}'
                    )
                    msg.attach(part)

    # Send email
    try:
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(from_addr, password)
            server.send_message(msg)
        print(f"✓ Email sent successfully to {msg['To']}")
        return True
    except Exception as e:
        print(f"✗ Failed to send email: {e}", file=sys.stderr)
        return False


def main():
    """Main function to send daily market research email."""
    # Get environment variables
    smtp_host = os.getenv('EMAIL_SMTP_HOST')
    smtp_port = os.getenv('EMAIL_SMTP_PORT', '587')
    from_addr = os.getenv('EMAIL_FROM')
    to_addrs = os.getenv('EMAIL_TO')
    password = os.getenv('EMAIL_PASSWORD')

    # Validate required variables
    if not all([smtp_host, from_addr, to_addrs, password]):
        print("✗ Missing required email configuration. Please set environment variables:", file=sys.stderr)
        print("  - EMAIL_SMTP_HOST", file=sys.stderr)
        print("  - EMAIL_SMTP_PORT (optional, defaults to 587)", file=sys.stderr)
        print("  - EMAIL_FROM", file=sys.stderr)
        print("  - EMAIL_TO", file=sys.stderr)
        print("  - EMAIL_PASSWORD", file=sys.stderr)
        return 1

    # Get latest results
    json_file, csv_file = get_latest_results()

    if not json_file:
        print("✗ No analysis results found in results/ directory", file=sys.stderr)
        return 1

    print(f"Found analysis results: {json_file.name}")

    # Load analysis data
    try:
        with open(json_file, 'r') as f:
            analysis_data = json.load(f)
    except Exception as e:
        print(f"✗ Failed to load analysis data: {e}", file=sys.stderr)
        return 1

    # Format email
    subject = f"Daily Market Research - {datetime.now().strftime('%Y-%m-%d')}"
    body_html = format_email_body(analysis_data)

    # Prepare attachments
    attachments = [str(json_file)]
    if csv_file:
        attachments.append(str(csv_file))

    # Send email
    success = send_email(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        from_addr=from_addr,
        to_addrs=to_addrs,
        password=password,
        subject=subject,
        body_html=body_html,
        attachments=attachments
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
