"""
Flask Web Application for TruthLens Fake News Detection
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import glob

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import FakeNewsPredictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Global predictor instance
predictor = None

def load_model():
    """Load the latest trained model"""
    global predictor
    
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    if not os.path.exists(models_dir):
        return None
    
    # Find all model files
    model_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    
    if not model_files:
        return None
    
    # Get the most recent model
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        predictor = FakeNewsPredictor(latest_model)
        return os.path.basename(latest_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model on startup
model_name = load_model()

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_loaded=predictor is not None, model_name=model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if predictor is None:
        return jsonify({
            'error': 'No model loaded. Please train a model first.',
            'success': False
        })
    
    try:
        # Get article text from form
        article_text = request.form.get('article', '').strip()
        
        if not article_text:
            return jsonify({
                'error': 'Please enter article text',
                'success': False
            })
        
        # Make prediction
        analysis = predictor.analyze_article(article_text)
        
        # Format response
        response = {
            'success': True,
            'prediction': analysis['prediction'],
            'confidence': round(analysis['confidence'] * 100, 2),
            'probabilities': {
                label: round(prob * 100, 2) 
                for label, prob in analysis['probabilities'].items()
            },
            'text_length': analysis['text_length'],
            'word_count': analysis['word_count']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if predictor is None:
        return jsonify({
            'error': 'No model loaded',
            'success': False
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing "text" field in request',
                'success': False
            }), 400
        
        article_text = data['text'].strip()
        
        if not article_text:
            return jsonify({
                'error': 'Text field is empty',
                'success': False
            }), 400
        
        # Make prediction
        analysis = predictor.analyze_article(article_text)
        
        response = {
            'success': True,
            'prediction': analysis['prediction'],
            'confidence': analysis['confidence'],
            'probabilities': analysis['probabilities'],
            'statistics': {
                'text_length': analysis['text_length'],
                'word_count': analysis['word_count']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/api/status')
def status():
    """Check API status"""
    return jsonify({
        'status': 'online',
        'model_loaded': predictor is not None,
        'model_name': model_name if predictor else None
    })

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("=" * 60)
    print("TRUTHLENS FAKE NEWS DETECTION - WEB APPLICATION")
    print("=" * 60)
    
    if predictor:
        print(f"✓ Model loaded: {model_name}")
        print(f"✓ Server starting at http://127.0.0.1:5000")
    else:
        print("⚠ WARNING: No model found!")
        print("Please train a model first by running:")
        print("  cd src")
        print("  python main.py")
    
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
