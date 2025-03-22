# utils.py
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import spacy
from transformers import pipeline
import streamlit as st

# Initialize sentiment analysis model (SiEBERT)
def load_sentiment_model():
    """Load the SiEBERT sentiment analysis model."""
    return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", truncation=True)

sentiment_analyzer = load_sentiment_model()

# Load spaCy model
def load_spacy_model():
    """Load the spaCy English model."""
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def analyze_sentiment(text):
    """Analyze sentiment of the given text using SiEBERT."""
    if not text or len(text.strip()) < 10:
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'label': 'neutral'}
    
    result = sentiment_analyzer(text[:512])[0]  # Truncate to 512 tokens
    label = result['label'].lower()
    score = result['score']
    
    if label == "positive":
        sentiment = {'compound': score, 'pos': score, 'neg': 0, 'neu': 1 - score, 'label': 'positive'}
    elif label == "negative":
        sentiment = {'compound': -score, 'pos': 0, 'neg': score, 'neu': 1 - score, 'label': 'negative'}
    else:
        sentiment = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': score, 'label': 'neutral'}
    
    if score < 0.6:
        sentiment = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1, 'label': 'neutral'}
    
    return sentiment

def search_company_news(company_name, num_articles=10):
    """Search for company news articles on Google News."""
    search_query = f"{company_name} company news -inurl:(subscription login signup)"
    url = f"https://www.google.com/search?q={search_query}&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        news_divs = soup.find_all('div', class_='SoaBEf')
        
        if not news_divs:
            st.warning("No news articles found in Google News search results. Google may have changed its HTML structure.")
            return []
        
        skip_titles = ["access denied", "just a moment", "captcha", "403 forbidden", "subscribe", "login"]
        for div in news_divs[:num_articles * 2]:
            headline = div.find('div', class_='mCBkyc').text.strip() if div.find('div', class_='mCBkyc') else "No headline"
            if any(skip_title in headline.lower() for skip_title in skip_titles):
                continue
                
            link = div.find('a')['href'] if div.find('a') else ""
            if link.startswith('/url?'):
                link = re.search(r'url=(.*?)&', link).group(1) if re.search(r'url=(.*?)&', link) else ""
            
            if link and not any(article['url'] == link for article in articles):
                articles.append({'title': headline, 'url': link})
        
        if not articles:
            st.warning("No valid article URLs extracted from Google News.")
        
        return articles[:num_articles]
    except Exception as e:
        st.error(f"Error searching for news: {str(e)}")
        return []

def extract_article_content(url, company_name):
    """Extract article content from a given URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.text.strip() if soup.title else "Unknown Title"
        
        skip_phrases = ["access denied", "just a moment", "captcha", "403 forbidden", "subscribe", "login"]
        if any(phrase in title.lower() for phrase in skip_phrases):
            return {'valid': False}
        
        article_content = next((container for container in [
            soup.find('article'),
            soup.find('div', class_=['article-content', 'article-body']),
            soup.find('main')
        ] if container), None)
        
        paragraphs = article_content.find_all('p') if article_content else [
            p for p in soup.find_all('p') if len(p.text.strip()) > 50
        ]
        text = ' '.join(p.text.strip() for p in paragraphs if not any(phrase in p.text.lower() for phrase in skip_phrases))
        
        # Relaxed validation: only require text length > 100 and company name in title or text
        if len(text.strip()) < 100:
            return {'valid': False}
        
        if company_name.lower() not in text.lower() and company_name.lower() not in title.lower():
            return {'valid': False}
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        company_sentences = [s for s in sentences if company_name.lower() in s.lower()]
        summary = ' '.join(company_sentences[:5] if len(company_sentences) >= 5 else sentences[:7])
        
        if len(summary) < 100 and len(text) > 100:
            summary = text[:400] + "..." if len(text) > 400 else text
        if len(summary) > 512:
            summary = summary[:509] + "..."
        
        return {'valid': True, 'title': title, 'text': text, 'summary': summary}
    except Exception as e:
        return {'valid': False, 'title': "Extraction Failed", 'text': "", 'summary': f"Error: {str(e)}"}

def extract_topics(summary):
    """Extract topics from article summary using spaCy."""
    if not summary:
        return ["General News"]
    
    doc = nlp(summary)
    entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT", "GPE"]}
    summary_lower = summary.lower()
    
    topic_categories = {
        "Technology": ["software", "tech", "update"],
        "Outages": ["outage", "downtime", "disruption"],
        "Business": ["company", "deal", "partnership"],
        "Financial": ["revenue", "profit", "stock"]
    }
    
    topics = {topic for topic, keywords in topic_categories.items() if 
              any(keyword in summary_lower for keyword in keywords) or 
              any(entity in summary_lower for entity in entities if any(keyword in entity for keyword in keywords))}
    
    return list(topics) if topics else ["General News"]

def compare_sentiment(articles):
    """Compare sentiment distribution across articles."""
    if not articles:
        return {'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0}}
    
    sentiment_labels = [article['sentiment']['label'] for article in articles]
    return {
        'sentiment_distribution': {
            'positive': sentiment_labels.count('positive'),
            'neutral': sentiment_labels.count('neutral'),
            'negative': sentiment_labels.count('negative')
        }
    }

def generate_final_sentiment(articles, company_name):
    """Generate a final sentiment summary for the company."""
    if not articles:
        return f"No sufficient data to analyze {company_name}'s news coverage."
    
    sentiment_dist = compare_sentiment(articles)['sentiment_distribution']
    total = sum(sentiment_dist.values())
    if total == 0:
        return f"{company_name}'s news coverage analysis inconclusive."
    
    positive_pct = sentiment_dist['positive'] / total
    negative_pct = sentiment_dist['negative'] / total
    combined_summary = " ".join(article['summary'] for article in articles)
    compound_score = analyze_sentiment(combined_summary)['compound']
    
    analysis = f"{company_name}'s news coverage: {sentiment_dist['positive']} positive, " \
               f"{sentiment_dist['negative']} negative, {sentiment_dist['neutral']} neutral articles. "
    
    if positive_pct > 0.6 and compound_score > 0.2:
        analysis += "Strongly positive outlook."
    elif negative_pct > 0.6 and compound_score < -0.2:
        analysis += "Strongly negative sentiment."
    else:
        analysis += "Mixed outlook."
    
    return analysis

def format_output(company_name, articles):
    """Format the analysis output as a dictionary."""
    formatted_articles = [
        {
            "TITLE": article['title'],
            "SUMMARY": article['summary'],
            "SENTIMENT": article['sentiment']['label'].capitalize(),
            "TOPICS": article['topics']
        } for article in articles
    ]
    
    sentiment_distribution = compare_sentiment(articles)['sentiment_distribution']
    
    return {
        "COMPANY": company_name,
        "ARTICLES": formatted_articles,
        "COMPARATIVE_SENTIMENT_SCORE": {"SENTIMENT_DISTRIBUTION": sentiment_distribution},
        "Final Sentiment Analysis": generate_final_sentiment(articles, company_name)
    }

def analyze_company_news(company_name, num_articles=10):
    """Main function to analyze company news."""
    articles_to_fetch = num_articles * 2
    fetched_articles = search_company_news(company_name, articles_to_fetch)
    
    if not fetched_articles:
        return None
    
    valid_articles = []
    for article in fetched_articles:
        if len(valid_articles) >= num_articles:
            break
        
        time.sleep(random.uniform(0.5, 1.5))  # Avoid rate limiting
        content = extract_article_content(article['url'], company_name)
        
        if content.get('valid', False):
            article.update({
                'title': content['title'],
                'text': content['text'],
                'summary': content['summary'],
                'topics': extract_topics(content['summary']),
                'sentiment': analyze_sentiment(content['summary'])
            })
            valid_articles.append(article)
    
    return {'company_name': company_name, 'articles': valid_articles} if valid_articles else None