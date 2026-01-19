# URL Article Extraction Feature

## Overview
TruthLens now supports extracting and analyzing news articles directly from URLs. This feature automatically fetches article content from web pages and analyzes it for fake news detection.

## Features

### 1. **Automatic Text Extraction**
- Extracts article content from news URLs
- Intelligently identifies main article content
- Removes ads, navigation, and other non-article elements
- Cleans and normalizes extracted text

### 2. **Comprehensive Error Handling**
The system handles various error scenarios:

#### Invalid URL Errors
- Empty URL
- Missing http:// or https://
- Malformed URL format

#### Network Errors
- **Connection Error**: Cannot connect to server
- **Timeout**: Server takes too long to respond (>15 seconds)
- **Too Many Redirects**: URL redirects excessively

#### HTTP Errors
- **403 Forbidden**: Website blocks automated access
- **404 Not Found**: Page doesn't exist
- **Other HTTP errors**: 500, 503, etc.

#### Content Errors
- **Empty Content**: No text found on the page
- **Too Short**: Article has fewer than 50 words
- **Extraction Failed**: Unable to identify article content

### 3. **Smart Content Detection**
The extractor uses multiple strategies to find article content:
- Searches for `<article>` tags
- Looks for common class names: "article", "post", "content", "entry", "story"
- Finds `<main>` sections
- Falls back to paragraph extraction
- Filters out navigation, ads, and other non-content elements

## Usage

### Web Interface

1. **Open the Application**
   ```
   http://127.0.0.1:5000
   ```

2. **Select URL Input Tab**
   - Click on the "URL Input" tab
   - Paste your news article URL
   - Click "Extract & Analyze"

3. **View Results**
   - Prediction (FAKE or REAL)
   - Confidence score
   - Probability distribution
   - Text statistics including words extracted

### API Usage

#### Endpoint: `/predict-url`

**Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict-url \
  -F "url=https://example.com/news-article"
```

**Success Response:**
```json
{
  "success": true,
  "prediction": "real",
  "confidence": 89.5,
  "probabilities": {
    "fake": 10.5,
    "real": 89.5
  },
  "text_length": 2543,
  "word_count": 425,
  "url": "https://example.com/news-article",
  "extracted_words": 425
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Access forbidden - The website is blocking automated access"
}
```

## Technical Implementation

### Dependencies
- **requests**: HTTP library for fetching URLs
- **beautifulsoup4**: HTML parsing and content extraction
- **lxml**: Fast XML and HTML parser

### File Structure
```
src/
  url_extractor.py     # URL text extraction module
app.py                 # Flask routes for URL prediction
templates/
  index.html           # Updated UI with URL input tab
static/
  js/predict.js        # Updated JavaScript for URL handling
  css/style.css        # Tab styling
```

### Key Classes

#### `URLTextExtractor`
Located in `src/url_extractor.py`

**Methods:**
- `validate_url(url)`: Validates URL format
- `extract_text(url)`: Main extraction method
- `_extract_article_content(soup)`: Finds article content in HTML
- `_clean_text(text)`: Cleans extracted text

**Configuration:**
```python
url_extractor = URLTextExtractor(
    timeout=15,        # Request timeout in seconds
    max_retries=3      # Maximum retry attempts
)
```

## Error Messages Reference

| Error | Meaning | Solution |
|-------|---------|----------|
| "URL is empty" | No URL provided | Enter a valid URL |
| "URL must start with http:// or https://" | Missing protocol | Add http:// or https:// |
| "Invalid URL format" | Malformed URL | Check URL syntax |
| "Access forbidden (403)" | Site blocks bots | Try a different article |
| "Page not found (404)" | URL doesn't exist | Verify the URL |
| "Request timeout" | Server too slow | Try again or use different article |
| "Connection error" | Network issue | Check internet connection |
| "Could not extract article content" | No text found | URL may not be a news article |
| "Extracted content is too short" | Less than 50 words | Article too brief for analysis |

## Best Practices

### For Users
1. **Use Direct Article URLs**: Link directly to the article, not homepage
2. **Try Alternative Sources**: If blocked, find the same article on another site
3. **Verify Results**: Always cross-reference with multiple sources
4. **Check URL**: Ensure the URL is complete and accessible

### For Developers
1. **Respect robots.txt**: The extractor includes a user agent but doesn't check robots.txt
2. **Rate Limiting**: Implement rate limiting for production use
3. **Caching**: Consider caching extracted content to reduce requests
4. **Error Logging**: Monitor extraction failures to improve content detection

## Limitations

1. **Paywalled Content**: Cannot extract from sites requiring login/subscription
2. **JavaScript-Heavy Sites**: May not work with single-page applications
3. **Bot Detection**: Some sites actively block automated access
4. **Dynamic Content**: Sites that load content via JavaScript may not work
5. **Language**: Optimized for English-language articles

## Security Considerations

1. **URL Validation**: All URLs are validated before fetching
2. **Timeout Protection**: Requests timeout after 15 seconds
3. **Size Limits**: No explicit size limit (consider adding for production)
4. **SSL Verification**: HTTPS certificates are verified
5. **Redirect Limits**: Prevents infinite redirect loops

## Performance

- **Typical extraction time**: 2-5 seconds
- **Timeout**: 15 seconds maximum
- **Retries**: Up to 3 attempts on failure
- **Memory usage**: Depends on article size (typically < 10MB)

## Future Enhancements

Potential improvements for the feature:
- [ ] Support for PDF articles
- [ ] Multi-language support
- [ ] Screenshot capture for verification
- [ ] Metadata extraction (author, date, source)
- [ ] Article summarization
- [ ] Automatic source credibility checking
- [ ] Support for Twitter/social media posts
- [ ] Batch URL processing

## Troubleshooting

### Issue: "Access forbidden" errors
**Solution**: Some news sites block automated access. Try:
- Using a different news source
- Copying and pasting the text manually
- Using the site's RSS feed if available

### Issue: Empty content extracted
**Solution**: The page structure may be unusual. Try:
- Verifying the URL leads to an article
- Checking if the content loads without JavaScript
- Using the text input method instead

### Issue: Timeout errors
**Solution**: The server is slow or unresponsive. Try:
- Waiting and trying again
- Using a cached version (Google Cache, Wayback Machine)
- Finding the article on a faster site

## Support

For issues or questions:
1. Check error message for specific guidance
2. Verify URL is accessible in a browser
3. Try the text input method as alternative
4. Report persistent issues with example URLs

---

**Version**: 1.0  
**Last Updated**: December 2025  
**Compatibility**: Flask 2.3.3+, Python 3.7+
