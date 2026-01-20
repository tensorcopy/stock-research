# GitHub Actions Daily Market Research Setup

This document describes the automated daily market research workflow using GitHub Actions.

## Overview

The workflow runs automatically every weekday after market close (5:00 PM ET / 22:00 UTC) to:
1. Analyze the stock market using trend detection, momentum signals, and alpha factors
2. Generate comprehensive reports in JSON and CSV formats
3. Send an email notification with the analysis results and attached reports

## Workflow Configuration

**File:** `.github/workflows/daily-market-research.yml`

**Schedule:**
- Runs Monday-Friday at 22:00 UTC (5:00 PM ET during EST)
- Can also be triggered manually via GitHub Actions UI

**What it does:**
1. Sets up Python 3.11 environment
2. Installs project dependencies
3. Runs `analyze_market.py` to generate market analysis
4. Sends email with results using `scripts/send_email.py`
5. Uploads analysis results as GitHub Actions artifacts (retained for 30 days)

## Required GitHub Secrets

To enable email notifications, you must configure the following secrets in your GitHub repository:

### Setting Up Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add each of the following:

### Required Secrets

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `EMAIL_SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `EMAIL_SMTP_PORT` | SMTP server port (usually 587 for TLS) | `587` |
| `EMAIL_FROM` | Sender email address | `your-email@gmail.com` |
| `EMAIL_TO` | Recipient email address(es) | `recipient@example.com` |
| `EMAIL_PASSWORD` | Email account password or app password | `your-app-password` |

### Email Provider Setup Examples

#### Gmail

1. Enable 2-Factor Authentication on your Google account
2. Generate an App Password:
   - Go to https://myaccount.google.com/security
   - Click on "2-Step Verification"
   - Scroll to "App passwords"
   - Generate a new app password for "Mail"
3. Use these settings:
   - `EMAIL_SMTP_HOST`: `smtp.gmail.com`
   - `EMAIL_SMTP_PORT`: `587`
   - `EMAIL_FROM`: Your Gmail address
   - `EMAIL_PASSWORD`: The app password (16 characters, no spaces)

#### Outlook/Office 365

- `EMAIL_SMTP_HOST`: `smtp-mail.outlook.com` or `smtp.office365.com`
- `EMAIL_SMTP_PORT`: `587`
- `EMAIL_FROM`: Your Outlook/Office 365 email
- `EMAIL_PASSWORD`: Your account password

#### SendGrid (Recommended for production)

1. Sign up at https://sendgrid.com
2. Create an API key with "Mail Send" permissions
3. Use these settings:
   - `EMAIL_SMTP_HOST`: `smtp.sendgrid.net`
   - `EMAIL_SMTP_PORT`: `587`
   - `EMAIL_FROM`: Your verified sender email
   - `EMAIL_PASSWORD`: Your SendGrid API key

## Email Report Contents

The email notification includes:

### HTML Email Body
- **Market Overview**: Sentiment (Bullish/Neutral/Bearish), bullish/bearish percentages
- **Top Momentum Stocks**: Top 10 stocks by momentum score with RSI values
- **Breakout Opportunities**: Top 10 stocks with active breakouts
- **Top Alpha Ranked Stocks**: Top 10 by composite alpha score
- **Strong Buy Candidates**: Stocks with positive confluence across trend, momentum, and alpha

### Attachments
- `market_analysis_YYYYMMDD_HHMMSS.json` - Complete analysis data
- `alpha_rankings_YYYYMMDD_HHMMSS.csv` - Ranked stock universe (Excel-compatible)

## Manual Trigger

To manually run the workflow:

1. Go to your GitHub repository
2. Click **Actions** tab
3. Select **Daily Market Research** workflow
4. Click **Run workflow** button
5. Select branch and click **Run workflow**

## Testing Locally

You can test the email notification locally:

```bash
# Set environment variables
export EMAIL_SMTP_HOST="smtp.gmail.com"
export EMAIL_SMTP_PORT="587"
export EMAIL_FROM="your-email@gmail.com"
export EMAIL_TO="recipient@example.com"
export EMAIL_PASSWORD="your-app-password"

# Run the analysis (generates results in ./results/)
python analyze_market.py

# Test email sending
python scripts/send_email.py
```

## Viewing Results

### GitHub Actions UI
1. Go to **Actions** tab
2. Click on a workflow run
3. Download artifacts under "Artifacts" section (available for 30 days)

### Email
- Check your inbox for daily reports
- Full analysis data attached as JSON and CSV

## Troubleshooting

### Email Not Sending

1. **Check GitHub Actions logs:**
   - Go to Actions tab → Click on failed run → Check "Send email notification" step

2. **Common issues:**
   - **Authentication failed**: Verify email password/app password is correct
   - **Connection refused**: Check SMTP host and port
   - **Invalid credentials**: Some providers require app-specific passwords (Gmail, Yahoo)
   - **Blocked by provider**: Enable "less secure app access" or use app passwords

### Analysis Failing

1. **Check dependencies:**
   ```bash
   pip install -e .
   ```

2. **Test locally:**
   ```bash
   python analyze_market.py
   ```

3. **Common issues:**
   - API rate limits from Yahoo Finance
   - Network connectivity issues
   - Invalid symbols in universe

### No Results Files

- Check that `results/` directory exists
- Verify `analyze_market.py` completed successfully
- Check workflow logs for errors

## Customization

### Change Schedule

Edit `.github/workflows/daily-market-research.yml`:

```yaml
on:
  schedule:
    - cron: '0 22 * * 1-5'  # Change time (format: minute hour day month weekday)
```

Examples:
- `0 21 * * 1-5` - 9:00 PM UTC (4:00 PM ET during EDT)
- `30 22 * * 1-5` - 10:30 PM UTC (5:30 PM ET during EST)
- `0 14 * * *` - 2:00 PM UTC every day (including weekends)

### Change Email Recipients

Update the `EMAIL_TO` secret to include multiple recipients (comma-separated):
```
recipient1@example.com, recipient2@example.com
```

### Modify Email Format

Edit `scripts/send_email.py` to customize:
- HTML styling (CSS in the `<style>` section)
- Report sections and tables
- Number of stocks shown (change slice indices like `[:10]`)
- Subject line format

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use app-specific passwords** instead of account passwords
3. **Rotate passwords periodically**
4. **Use dedicated email accounts** for automation
5. **Consider SendGrid or similar services** for production use
6. **Review GitHub Actions logs** but note they hide secret values

## Cost Considerations

- **GitHub Actions**: 2,000 free minutes/month for public repos, 500 minutes for private repos
- **This workflow uses**: ~2-3 minutes per run
- **Monthly usage**: ~44-66 minutes (22 weekdays × 2-3 minutes)
- **Well within free tier** for most users

## Support

For issues or questions:
- Check GitHub Actions logs
- Review this documentation
- Test email configuration locally
- Verify all required secrets are set

## Next Steps

1. ✅ Configure GitHub Secrets (see above)
2. ✅ Test manual workflow run
3. ✅ Verify email delivery
4. ✅ Wait for first scheduled run (or trigger manually)
5. ✅ Review results and customize as needed
