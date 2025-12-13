// TruthLens - Prediction Functionality

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const textarea = document.getElementById('article-text');
    const charCounter = document.getElementById('char-counter');
    const clearBtn = document.getElementById('clear-btn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');

    // Character counter
    if (textarea && charCounter) {
        textarea.addEventListener('input', function() {
            const count = this.value.length;
            charCounter.textContent = `${count} characters`;
            
            if (count < 20) {
                charCounter.style.color = '#ef4444';
            } else {
                charCounter.style.color = '#10b981';
            }
        });
    }

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            textarea.value = '';
            charCounter.textContent = '0 characters';
            charCounter.style.color = '#6b7280';
            hideResult();
            hideError();
        });
    }

    // Form submission
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const text = textarea.value.trim();
            
            // Validation
            if (!text) {
                showError('Please enter article text');
                return;
            }
            
            if (text.length < 20) {
                showError('Article text must be at least 20 characters');
                return;
            }

            // Hide previous results and errors
            hideResult();
            hideError();
            
            // Show loading
            loading.style.display = 'block';

            try {
                // Send request
                const formData = new FormData();
                formData.append('article', text);

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                // Hide loading
                loading.style.display = 'none';

                if (data.success) {
                    displayResult(data);
                } else {
                    showError(data.error || 'An error occurred during prediction');
                }

            } catch (err) {
                loading.style.display = 'none';
                showError('Network error: ' + err.message);
            }
        });
    }

    function displayResult(data) {
        // Update prediction label
        const predictionText = document.getElementById('prediction-text');
        const predictionLabel = predictionText.parentElement;
        
        predictionText.textContent = data.prediction.toUpperCase();
        predictionLabel.className = 'prediction-label ' + data.prediction.toLowerCase();

        // Update confidence bar
        const confidenceFill = document.getElementById('confidence-fill');
        const confidenceValue = document.getElementById('confidence-value');
        
        confidenceFill.style.width = data.confidence + '%';
        confidenceValue.textContent = data.confidence.toFixed(1) + '%';

        // Update probabilities
        const probBars = document.getElementById('prob-bars');
        probBars.innerHTML = '';

        for (const [label, prob] of Object.entries(data.probabilities)) {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';
            
            probItem.innerHTML = `
                <div class="prob-label">
                    <span>${label.toUpperCase()}</span>
                    <span>${prob.toFixed(1)}%</span>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill ${label.toLowerCase()}" style="width: ${prob}%"></div>
                </div>
            `;
            
            probBars.appendChild(probItem);
        }

        // Update statistics
        document.getElementById('text-length').textContent = data.text_length.toLocaleString();
        document.getElementById('word-count').textContent = data.word_count.toLocaleString();

        // Show result
        result.style.display = 'block';
        
        // Scroll to result
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showError(message) {
        error.textContent = message;
        error.style.display = 'block';
        error.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideResult() {
        result.style.display = 'none';
    }

    function hideError() {
        error.style.display = 'none';
    }

    // Sample text button (for demo)
    const sampleTexts = {
        fake: "BREAKING: Scientists discover that eating chocolate makes you fly! New research shows that consuming just 3 chocolate bars per day can give humans the ability to fly up to 100 feet in the air. Doctors are shocked by this discovery and don't want you to know about it!",
        real: "Researchers at the University have published a new study in the journal Nature examining the effects of climate change on coastal ecosystems. The peer-reviewed research, conducted over five years, indicates significant changes in species distribution patterns and habitat conditions."
    };

    // Add sample buttons if desired
    // You can add this to your HTML and uncomment
    /*
    const sampleFakeBtn = document.getElementById('sample-fake');
    const sampleRealBtn = document.getElementById('sample-real');
    
    if (sampleFakeBtn) {
        sampleFakeBtn.addEventListener('click', function() {
            textarea.value = sampleTexts.fake;
            textarea.dispatchEvent(new Event('input'));
        });
    }
    
    if (sampleRealBtn) {
        sampleRealBtn.addEventListener('click', function() {
            textarea.value = sampleTexts.real;
            textarea.dispatchEvent(new Event('input'));
        });
    }
    */
});
