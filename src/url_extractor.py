"""
URL Text Extraction Module

This module provides functionality to extract article text from URLs
with comprehensive error handling and content cleaning.

Author: TruthLens Team
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re


class URLTextExtractor:
    """Extract and clean text content from news article URLs"""
    
    def __init__(self, timeout=10, max_retries=3):
        """
        Initialize the URL text extractor.
        
        Args:
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def validate_url(self, url):
        """
        Validate if the provided string is a valid URL.
        
        Args:
            url (str): URL string to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not url or not url.strip():
            return False, "URL is empty"
        
        url = url.strip()
        
        # Check if URL has a scheme
        if not url.startswith(('http://', 'https://')):
            return False, "URL must start with http:// or https://"
        
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False, "Invalid URL format"
            return True, None
        except Exception as e:
            return False, f"Invalid URL: {str(e)}"
    
    def extract_text(self, url):
        """
        Extract text content from a URL.
        
        Args:
            url (str): The URL to extract text from
            
        Returns:
            dict: Dictionary containing:
                - success (bool): Whether extraction was successful
                - text (str): Extracted text content (if successful)
                - error (str): Error message (if failed)
                - url (str): Original URL
                - word_count (int): Number of words extracted
        """
        # Validate URL
        is_valid, error_msg = self.validate_url(url)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'url': url
            }
        
        # Try to fetch the URL
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                # Check response status
                if response.status_code == 403:
                    return {
                        'success': False,
                        'error': 'Access forbidden - The website is blocking automated access',
                        'url': url
                    }
                elif response.status_code == 404:
                    return {
                        'success': False,
                        'error': 'Page not found (404)',
                        'url': url
                    }
                elif response.status_code >= 400:
                    return {
                        'success': False,
                        'error': f'HTTP error {response.status_code}',
                        'url': url
                    }
                
                response.raise_for_status()
                
                # Parse HTML content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 
                                'aside', 'iframe', 'noscript', 'form', 'button',
                                'input', 'select', 'textarea']):
                    tag.decompose()
                
                # Remove common unwanted classes/ids
                unwanted_patterns = [
                    'cookie', 'popup', 'modal', 'advertisement', 'ad-', 'sidebar',
                    'menu', 'navigation', 'social', 'share', 'related', 'comment',
                    'promo', 'widget', 'sponsor'
                ]
                
                for pattern in unwanted_patterns:
                    for element in soup.find_all(class_=re.compile(pattern, re.I)):
                        element.decompose()
                    for element in soup.find_all(id=re.compile(pattern, re.I)):
                        element.decompose()
                
                # Try to find article content using common patterns
                article_text = self._extract_article_content(soup)
                
                if not article_text:
                    return {
                        'success': False,
                        'error': 'Could not extract article content - No text found',
                        'url': url
                    }
                
                # Clean the text
                cleaned_text = self._clean_text(article_text)
                
                if not cleaned_text or len(cleaned_text.split()) < 50:
                    return {
                        'success': False,
                        'error': 'Extracted content is too short - Article must have at least 50 words',
                        'url': url
                    }
                
                return {
                    'success': True,
                    'text': cleaned_text,
                    'url': url,
                    'word_count': len(cleaned_text.split())
                }
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    continue
                return {
                    'success': False,
                    'error': f'Request timeout - The server took too long to respond (>{self.timeout}s)',
                    'url': url
                }
            
            except requests.exceptions.ConnectionError:
                return {
                    'success': False,
                    'error': 'Connection error - Could not connect to the server',
                    'url': url
                }
            
            except requests.exceptions.TooManyRedirects:
                return {
                    'success': False,
                    'error': 'Too many redirects - The URL redirected too many times',
                    'url': url
                }
            
            except requests.exceptions.RequestException as e:
                return {
                    'success': False,
                    'error': f'Request error: {str(e)}',
                    'url': url
                }
            
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Unexpected error: {str(e)}',
                    'url': url
                }
    
    def _extract_article_content(self, soup):
        """
        Extract article content from BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            str: Extracted article text
        """
        # Try common article selectors
        article_selectors = [
            {'name': 'article'},
            {'class_': re.compile(r'article|post-content|entry-content|story-body|article-body', re.I)},
            {'id': re.compile(r'article|post-content|entry-content|story-body|article-body', re.I)},
            {'name': 'main'},
            {'role': 'main'},
        ]
        
        for selector in article_selectors:
            elements = soup.find_all(**selector)
            if elements:
                # Get text from all matching elements
                texts = []
                for elem in elements:
                    # Find only substantial paragraphs (more than 20 characters)
                    paragraphs = elem.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        # Only include paragraphs with meaningful content
                        if text and len(text) > 20 and len(text.split()) > 5:
                            texts.append(text)
                
                if texts and len(' '.join(texts).split()) >= 50:
                    return '\n'.join(texts)
        
        # Fallback: Get all substantial paragraphs from body
        paragraphs = soup.find_all('p')
        if paragraphs:
            texts = []
            for p in paragraphs:
                text = p.get_text().strip()
                # Filter out short paragraphs that are likely navigation/ads
                if text and len(text) > 20 and len(text.split()) > 5:
                    texts.append(text)
            
            if texts and len(' '.join(texts).split()) >= 50:
                return '\n'.join(texts)
        
        # Last resort: get all text but this is less reliable
        return soup.get_text()
    
    def _clean_text(self, text):
        """
        Clean extracted text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove very short lines (likely navigation/menu items - less than 3 words)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip().split()) >= 3]
        text = ' '.join(cleaned_lines)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace one more time
        text = re.sub(r'\s+', ' ', text)
        
        return text
