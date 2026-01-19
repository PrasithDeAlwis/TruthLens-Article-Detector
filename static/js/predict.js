// TruthLens - Prediction Functionality

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const urlForm = document.getElementById('url-form');
    const textarea = document.getElementById('article-text');
    const urlInput = document.getElementById('article-url');
    const charCounter = document.getElementById('char-counter');
    const clearBtn = document.getElementById('clear-btn');
    const clearUrlBtn = document.getElementById('clear-url-btn');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    
    // Tab switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and its content
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Hide results and errors when switching tabs
            hideResult();
            hideError();
        });
    });

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

    // Clear button for text input
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            textarea.value = '';
            charCounter.textContent = '0 characters';
            charCounter.style.color = '#6b7280';
            hideResult();
            hideError();
        });
    }
    
    // Clear button for URL input
    if (clearUrlBtn) {
        clearUrlBtn.addEventListener('click', function() {
            urlInput.value = '';
            hideResult();
            hideError();
        });
    }

    // Text form submission
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
    
    // URL form submission
    if (urlForm) {
        urlForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const url = urlInput.value.trim();
            
            // Validation
            if (!url) {
                showError('Please enter a URL');
                return;
            }
            
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                showError('URL must start with http:// or https://');
                return;
            }

            // Hide previous results and errors
            hideResult();
            hideError();
            
            // Show loading
            loading.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Extracting and analyzing article...';
            loading.style.display = 'block';

            try {
                // Send request
                const formData = new FormData();
                formData.append('url', url);

                const response = await fetch('/predict-url', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                // Hide loading
                loading.style.display = 'none';
                loading.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing article...';

                if (data.success) {
                    displayResult(data, true);
                } else {
                    showError(data.error || 'An error occurred during prediction');
                }

            } catch (err) {
                loading.style.display = 'none';
                loading.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing article...';
                showError('Network error: ' + err.message);
            }
        });
    }

    function displayResult(data, fromUrl = false) {
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
        
        // Handle extracted text display for URL input
        const extractedTextSection = document.getElementById('extracted-text-section');
        const extractedTextContent = document.getElementById('extracted-text-content');
        const toggleBtn = document.getElementById('toggle-full-text');
        
        if (fromUrl && data.extracted_text) {
            // Show extracted text section
            extractedTextSection.style.display = 'block';
            extractedTextContent.textContent = data.extracted_text;
            extractedTextContent.classList.remove('expanded');
            
            // Store full text as data attribute
            extractedTextContent.dataset.fullText = data.extracted_text;
            extractedTextContent.dataset.preview = data.extracted_text;
            
            // Reset toggle button
            toggleBtn.innerHTML = 'Show more <i class="fas fa-chevron-down"></i>';
            toggleBtn.onclick = function() {
                if (extractedTextContent.classList.contains('expanded')) {
                    extractedTextContent.classList.remove('expanded');
                    this.innerHTML = 'Show more <i class="fas fa-chevron-down"></i>';
                } else {
                    extractedTextContent.classList.add('expanded');
                    this.innerHTML = 'Show less <i class="fas fa-chevron-up"></i>';
                }
            };
            
            // Add URL stat if from URL
            const statsGrid = document.querySelector('.stats-grid');
            let urlStatItem = document.getElementById('url-stat-item');
            if (!urlStatItem) {
                urlStatItem = document.createElement('div');
                urlStatItem.id = 'url-stat-item';
                urlStatItem.className = 'stat-item';
                statsGrid.appendChild(urlStatItem);
            }
            
            urlStatItem.innerHTML = `
                <i class="fas fa-link"></i>
                <div class="stat-value">${data.extracted_words.toLocaleString()}</div>
                <div class="stat-label">Words Extracted</div>
            `;
        } else {
            // Hide extracted text section for text input
            extractedTextSection.style.display = 'none';
            
            // Remove URL stat if it exists
            const urlStatItem = document.getElementById('url-stat-item');
            if (urlStatItem) {
                urlStatItem.remove();
            }
        }

        // Show result
        result.style.display = 'block';
        
        // Scroll to result
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showError(message) {
        error.innerHTML = '<i class="fas fa-exclamation-triangle"></i> ' + message;
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
