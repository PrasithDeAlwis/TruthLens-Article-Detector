# TruthLens Web Application

A user-friendly web interface for the TruthLens Fake News Detection System.

## 🌐 Features

- **Interactive Web Interface**: Easy-to-use web form for article analysis
- **Real-time Predictions**: Instant feedback with confidence scores
- **Visual Results**: Color-coded predictions with probability distributions
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **REST API**: JSON API for programmatic access
- **Professional UI**: Modern, clean design with TruthLens branding

## 🚀 Quick Start

### 1. Train a Model First

Before running the web app, you need a trained model:

```powershell
cd src
python main.py
```

Follow the prompts to train your model. The web app will automatically load the most recent model.

### 2. Start the Web Server

```powershell
# From the project root directory
python app.py
```

The server will start at: **http://127.0.0.1:5000**

### 3. Open in Browser

Navigate to `http://127.0.0.1:5000` in your web browser.

## 📱 Using the Web App

### Home Page

1. **Enter Article Text**: Paste the news article you want to analyze
2. **Click "Analyze Article"**: Submit the form
3. **View Results**: See prediction, confidence score, and detailed statistics

### Features

- **Character Counter**: Shows article length in real-time
- **Clear Button**: Reset the form quickly
- **Visual Feedback**: Color-coded results (red for fake, green for real)
- **Probability Distribution**: See confidence for each category
- **Text Statistics**: View article length and word count

## 🔌 API Usage

### Check Status

```bash
GET /api/status
```

Response:
```json
{
  "status": "online",
  "model_loaded": true,
  "model_name": "logistic_regression_20231213_120000.pkl"
}
```

### Make Prediction

```bash
POST /api/predict
Content-Type: application/json

{
  "text": "Your article text here..."
}
```

Response:
```json
{
  "success": true,
  "prediction": "real",
  "confidence": 0.89,
  "probabilities": {
    "fake": 0.11,
    "real": 0.89
  },
  "statistics": {
    "text_length": 245,
    "word_count": 42
  }
}
```

### Example with cURL

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Scientists discover amazing new cure..."}'
```

### Example with Python

```python
import requests

url = "http://127.0.0.1:5000/api/predict"
data = {
    "text": "Your article text here..."
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📁 Project Structure

```
TruthLens-Article-Detector/
├── app.py                    # Flask application
├── templates/                # HTML templates
│   ├── base.html            # Base template
│   ├── index.html           # Home page
│   ├── about.html           # About page
│   ├── 404.html             # 404 error page
│   └── 500.html             # 500 error page
├── static/                   # Static assets
│   ├── css/
│   │   └── style.css        # Stylesheet
│   └── js/
│       ├── main.js          # Main JavaScript
│       └── predict.js       # Prediction functionality
└── models/                   # Trained models (auto-loaded)
```

## ⚙️ Configuration

### Change Port

Edit `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use port 8080
```

### Production Deployment

For production, use a proper WSGI server:

```powershell
# Install gunicorn (Linux/Mac) or waitress (Windows)
pip install waitress

# Run with waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Security

Before deploying to production:

1. Change the `SECRET_KEY` in `app.py`
2. Set `debug=False`
3. Use HTTPS
4. Add rate limiting
5. Implement authentication if needed

## 🎨 Customization

### Modify Colors

Edit `static/css/style.css` to change the color scheme:

```css
:root {
    --primary-color: #2563eb;    /* Blue */
    --secondary-color: #7c3aed;  /* Purple */
    --fake-color: #dc2626;       /* Red */
    --real-color: #059669;       /* Green */
}
```

### Add Features

Common additions:

- **History**: Save prediction history
- **Batch Analysis**: Upload CSV files
- **User Accounts**: Add authentication
- **Statistics Dashboard**: Show usage stats
- **Export Results**: Download as PDF/CSV

## 🐛 Troubleshooting

### Port Already in Use

```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID)
taskkill /PID <process_id> /F
```

### Model Not Loading

1. Check if `models/` directory has `.pkl` files
2. Train a model: `cd src && python main.py`
3. Check console for error messages

### Template Not Found

Ensure folder structure is correct:
```
project-root/
├── app.py
├── templates/
└── static/
```

### CSS/JS Not Loading

1. Check browser console for errors
2. Ensure `static` folder exists
3. Clear browser cache (Ctrl+Shift+R)

## 📊 Performance Tips

1. **Model Size**: Smaller models load faster
2. **Caching**: Enable Flask caching for production
3. **CDN**: Use CDN for Font Awesome icons
4. **Compression**: Enable gzip compression
5. **Async**: Use async for concurrent requests

## 🔒 Security Considerations

- **Input Validation**: Text length limits implemented
- **CSRF Protection**: Add Flask-WTF for forms
- **Rate Limiting**: Use Flask-Limiter
- **HTTPS**: Always use SSL in production
- **Sanitization**: HTML escaping enabled by default

## 📚 Resources

- **Flask Documentation**: https://flask.palletsprojects.com/
- **Bootstrap (optional)**: https://getbootstrap.com/
- **Font Awesome Icons**: https://fontawesome.com/

## 🤝 Contributing

Improvements welcome:
- Enhanced UI/UX
- Additional visualizations
- Mobile app integration
- Docker containerization
- Cloud deployment guides

---

**Ready to detect fake news? Start the server and open your browser!** 🚀
