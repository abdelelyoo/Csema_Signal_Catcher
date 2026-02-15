# Signal Catcher Web App

A Flask-based web interface for the CSEMA Trading System.

## Features

- **Stage 1: Market Screening** - Interactive screening with table display
- **Stage 2: Signal Generation** - Visual signal cards with entry/stop/target
- **Position Sizing** - Detailed calculations with CSEMA brokerage fees
- **Performance Tracking** - View comprehensive reports
- **Responsive Design** - Works on desktop and mobile

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-web.txt
```

### 2. Run the Web App

```bash
python app.py
```

### 3. Access the App

Open your browser and go to: **http://127.0.0.1:5000**

## Web Interface Structure

```
📁 templates/
  ├── base.html              # Base template with navigation
  ├── index.html             # Home page
  ├── screening.html         # Stage 1: Market Screening
  ├── signals.html           # Stage 2: Signal Generation
  ├── position_sizing.html   # Position Sizing with fees
  ├── performance.html       # Performance Tracking
  └── about.html             # About page

📁 static/
  ├── css/
  │   └── style.css          # Main stylesheet
  └── js/
      └── main.js            # JavaScript utilities

📄 app.py                     # Flask application
📄 requirements-web.txt       # Python dependencies
```

## Usage Workflow

### Step 1: Market Screening
1. Go to **Stage 1: Screening**
2. Optionally check "Show ALL tickers" for unfiltered results
3. Click "Run Screening"
4. View position and swing candidates in tables

### Step 2: Signal Generation
1. Go to **Stage 2: Signals**
2. Enter your available cash (default: 100,000 MAD)
3. Click "Generate Signals"
4. Review signal cards with entry, stop, target, and R/R

### Step 3: Position Sizing
1. Go to **Position Sizing**
2. Enter your available cash
3. Click "Calculate Position Sizes"
4. View detailed table with:
   - Position sizes
   - Break-even prices
   - Buy fees and roundtrip fees
   - Net risk/reward ratios

### Step 4: Performance
1. Go to **Performance**
2. View your trading performance report
3. See gross vs net P&L, win rates, and fees impact

## Configuration

### Environment Variables

```bash
# Optional: TradingView credentials for real-time data
TV_USERNAME=your_username
TV_PASSWORD=your_password

# Optional: Custom data directory
DATA_DIR=./data

# Flask settings
FLASK_ENV=development
FLASK_DEBUG=1
```

### Default Settings

- **Data Directory:** `./data`
- **Default Cash:** 100,000 MAD
- **Minimum Turnover:** 100,000 MAD
- **Brokerage Fees:** CSEMA rates (0.6% + 0.2% + 0.1% + VAT)

## API Endpoints

The web app also provides JSON API endpoints:

### POST /api/screening
```bash
curl -X POST http://localhost:5000/api/screening \
  -H "Content-Type: application/json" \
  -d '{"full_watchlist": false}'
```

### POST /api/signals
```bash
curl -X POST http://localhost:5000/api/signals \
  -H "Content-Type: application/json" \
  -d '{"cash": 100000}'
```

## Screenshots

### Home Page
Dashboard with quick access to all stages

### Screening Results
Table view of position and swing candidates with key metrics

### Signal Cards
Visual cards showing entry, stop, target, and risk/reward

### Position Sizing
Detailed table with brokerage fees and break-even calculations

## Customization

### Styling
Edit `static/css/style.css` to customize:
- Colors (CSS variables at the top)
- Layouts
- Typography
- Responsive breakpoints

### Templates
All HTML templates use Jinja2 and extend `base.html`:
- Modify `base.html` for global changes (navbar, footer)
- Edit individual templates for page-specific content

## Development

### Run in Debug Mode
```bash
python app.py
```

### Production Deployment
For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## Troubleshooting

### Issue: No screening results
- Check if data directory exists: `mkdir -p data`
- Try "Show ALL tickers" option
- Check if tvscreener library is installed

### Issue: No signals generated
- Must run Stage 1: Screening first
- Check if watchlist files exist in data directory
- Verify cash amount is sufficient (> 10,000 MAD)

### Issue: Position sizing shows insufficient funds
- Increase cash amount in the form
- Select fewer signals manually
- Check total cost with fees column

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Mobile Support

The web app is fully responsive and works on:
- Desktop browsers
- Tablets
- Mobile phones

## Security Notes

- Never commit `TV_USERNAME` and `TV_PASSWORD` to version control
- Use environment variables for sensitive data
- In production, disable Flask debug mode
- Consider adding authentication for production use

## Performance

The web app uses:
- Server-side rendering for fast initial load
- Minimal JavaScript for interactivity
- Efficient CSS with CSS variables
- Optimized table rendering for large datasets

## Contributing

To add new features:
1. Add route in `app.py`
2. Create template in `templates/`
3. Add styles in `static/css/style.css`
4. Add JavaScript in `static/js/main.js` (if needed)

## License

Same as the main Signal Catcher project.

## Support

For issues or questions:
- Check the main README.md
- Review the voir.txt file
- Report issues at: https://github.com/anomalyco/opencode/issues
