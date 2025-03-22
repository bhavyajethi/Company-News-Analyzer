# THIS IS THE MAIN CODE FILE OF THE PROJECT. USE THIS TO RUN THE STREAMLIT APP

import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import time
import random
import spacy
from transformers import pipeline

from transformers import pipeline

# Initialize SiEBERT, a RoBERTa-large model fine-tuned for sentiment analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", truncation=True)

sentiment_analyzer = load_sentiment_model()

def analyze_sentiment(text):
    if not text or len(text.strip()) < 10:  # Minimum length check
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'label': 'neutral'}
    
    # Analyze sentiment with SiEBERT
    result = sentiment_analyzer(text)[0]  # Truncation handled by pipeline
    
    # SiEBERT outputs: "positive" or "negative" (no explicit neutral, but low confidence can imply it)
    label = result['label'].lower()
    score = result['score']  # Confidence score
    
    # Map to your required format
    if label == "positive":
        sentiment = {'compound': score, 'pos': score, 'neg': 0, 'neu': 1 - score, 'label': 'positive'}
    elif label == "negative":
        sentiment = {'compound': -score, 'pos': 0, 'neg': score, 'neu': 1 - score, 'label': 'negative'}
    else:  # Handle unexpected labels, though rare with SiEBERT
        sentiment = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': score, 'label': 'neutral'}
    
    # Threshold for neutral: if confidence is low (<0.6), classify as neutral
    if score < 0.6:
        sentiment = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1, 'label': 'neutral'}
    
    return sentiment

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

def search_company_news(company_name, num_articles=10):
    search_query = f"{company_name} company news -inurl:(subscription login signup)"  # Exclude subscription/login pages
    url = f"https://www.google.com/search?q={search_query}&tbm=nws"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            st.error(f"Failed to fetch page: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        news_divs = soup.find_all('div', class_='SoaBEf')
        
        skip_titles = ["access denied", "just a moment", "captcha", "403 forbidden", "subscribe", "login"]
        for div in news_divs[:num_articles * 2]:  # Fetch extra to filter later
            headline_element = div.find('div', class_='mCBkyc')
            headline = headline_element.text.strip() if headline_element else "No headline"
            
            if any(skip_title in headline.lower() for skip_title in skip_titles):
                continue
                
            link_element = div.find('a')
            link = link_element['href'] if link_element else ""
            
            if link.startswith('/url?'):
                link = re.search(r'url=(.*?)&', link)
                link = link.group(1) if link else ""
            
            if link and not any(article['url'] == link for article in articles):
                articles.append({
                    'title': headline,
                    'url': link
                })
        
        return articles[:num_articles]
    except Exception as e:
        st.error(f"Error searching for news: {str(e)}")
        return []

def extract_article_content(url, company_name):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.6998.89 Safari/537.36",
            "Referer": "https://www.google.com/"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.text.strip() if soup.title else "Unknown Title"
        
        skip_phrases = ["access denied", "just a moment", "cloudflare", "captcha", "403 forbidden", "subscribe now", "log in", "sign up"]
        if any(phrase in title.lower() for phrase in skip_phrases):
            return {'valid': False}
        
        # Enhanced content extraction
        article_content = None
        for container in [
            soup.find('article'),
            soup.find('div', class_=['article-content', 'article-body', 'story-content', 'post-content', 'entry-content']),
            soup.find('div', {'id': ['article-content', 'article-body', 'story-content', 'post-content', 'entry-content']}),
            soup.find('main'),
        ]:
            if container:
                article_content = container
                break
        
        if not article_content:
            paragraphs = [p for p in soup.find_all('p') if len(p.text.strip()) > 50 and company_name.lower() in p.text.lower()]
        else:
            paragraphs = article_content.find_all('p')
        
        text = ' '.join([p.text.strip() for p in paragraphs if not any(phrase in p.text.lower() for phrase in skip_phrases)])
        
        if len(text.strip()) < 150 or company_name.lower() not in text.lower():  # Ensure company relevance
            return {'valid': False}
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Prioritize sentences mentioning the company
        company_sentences = [s for s in sentences if company_name.lower() in s.lower()]
        summary_sentences = company_sentences[:5] if len(company_sentences) >= 5 else sentences[:7]
        summary = ' '.join(summary_sentences)
        
        if len(summary) < 100 and len(text) > 100:
            summary = text[:400] + "..." if len(text) > 400 else text
        
        if len(summary) > 512:
            summary = summary[:509] + "..."
        
        return {
            'valid': True,
            'title': title,
            'text': text,
            'summary': summary
        }
    except Exception as e:
        return {
            'valid': False,
            'title': "Extraction Failed",
            'text': "",
            'summary': f"Error: {str(e)}"
        }

def analyze_sentiment(text):
    if not text:
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0, 'label': 'neutral'}
    
    # Analyze with BERT
    result = sentiment_analyzer(text[:512])[0]  # Truncate to 512 tokens
    label = result['label'].lower()
    score = result['score']
    
    # Initial sentiment mapping
    if label == 'positive':
        sentiment = {'compound': score, 'pos': score, 'neg': 0, 'neu': 1 - score, 'label': 'positive'}
    elif label == 'negative':
        sentiment = {'compound': -score, 'pos': 0, 'neg': score, 'neu': 1 - score, 'label': 'negative'}
    else:
        sentiment = {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1, 'label': 'neutral'}
    
    # Context-based override for common misclassifications
    negative_keywords = ["outage", "problem", "issue", "blocked", "failed", "disruption", "revert"]
    positive_keywords = ["success", "restored", "improved", "launched", "growth"]
    
    text_lower = text.lower()
    if any(kw in text_lower for kw in negative_keywords) and sentiment['label'] == 'positive':
        sentiment = {'compound': -score, 'pos': 0, 'neg': score, 'neu': 1 - score, 'label': 'negative'}
    elif any(kw in text_lower for kw in positive_keywords) and sentiment['label'] == 'negative':
        sentiment = {'compound': score, 'pos': score, 'neg': 0, 'neu': 1 - score, 'label': 'positive'}
    
    return sentiment

def compare_sentiment(articles):
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

def extract_topics(summary):
    if not summary:
        return ["General News"]
    
    doc = nlp(summary)
    entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "EVENT", "LAW", "GPE"]}
    summary_lower = summary.lower()
    
    # Refined topic categories with more specific keywords
    topic_categories = {
        "Technology": ["software", "hardware", "tech", "microsoft", "outlook", "infrastructure", "update", "code"],
        "Outages": ["outage", "downtime", "service disruption", "blocked", "access", "restore"],
        "Business": ["company", "business", "corporate", "deal", "partnership"],
        "Financial": ["revenue", "profit", "sales", "earnings", "stock"],
        "Legal": ["lawsuit", "legal", "court", "dispute", "regulation"],
        "Innovation": ["innovation", "research", "development", "new product"],
        "Electric Vehicles": ["electric vehicle", "ev", "battery", "tesla model", "charging"],
        "Autonomous Vehicles": ["autonomous", "self-driving", "driverless", "autopilot"]
    }
    
    topics = set()
    
    # Match topics based on keywords and entities
    for topic, keywords in topic_categories.items():
        if any(keyword in summary_lower for keyword in keywords) or \
           any(entity in summary_lower for entity in entities if any(keyword in entity for keyword in keywords)):
            topics.add(topic)
    
    # Fallback to "General News" if no specific topics match
    return list(topics) if topics else ["General News"]

def analyze_coverage_differences(articles):
    if len(articles) < 2:
        return []
    
    differences = []
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            art1, art2 = articles[i], articles[j]
            sentiment1, sentiment2 = art1['sentiment']['label'], art2['sentiment']['label']
            topics1, topics2 = art1['topics'], art2['topics']
            
            comparison = f"Article {i+1} ({art1['title'][:30]}...) has {sentiment1} sentiment on {', '.join(topics1[:2])}, " \
                        f"while Article {j+1} ({art2['title'][:30]}...) has {sentiment2} sentiment on {', '.join(topics2[:2])}."
            impact = f"Article {i+1} may {'boost confidence' if sentiment1 == 'positive' else 'raise concerns'}, " \
                    f"while Article {j+1} may {'boost confidence' if sentiment2 == 'positive' else 'raise concerns'}."
            
            differences.append({"Comparison": comparison, "Impact": impact})
    
    return differences

def analyze_topic_overlap(articles):
    if not articles:
        return {"Common Topics": [], "Unique Topics": {}}
    
    all_topics = [set(article['topics']) for article in articles]
    common_topics = set.intersection(*all_topics) if all_topics else set()
    
    unique_topics = {}
    topic_counts = {}
    
    for i, topics in enumerate(all_topics, 1):
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    for i, topics in enumerate(all_topics, 1):
        unique = [topic for topic in topics if topic_counts[topic] == 1]
        unique_topics[f"Article {i}"] = unique if unique else list(topics)
    
    return {
        "Common Topics": list(common_topics),
        "Unique Topics": unique_topics
    }

def generate_final_sentiment(articles, company_name):
    if not articles:
        return f"No sufficient data to analyze {company_name}'s news coverage."
    
    sentiment_dist = compare_sentiment(articles)['sentiment_distribution']
    total = sum(sentiment_dist.values())
    if total == 0:
        return f"{company_name}'s news coverage analysis inconclusive."
    
    positive_pct = sentiment_dist['positive'] / total
    negative_pct = sentiment_dist['negative'] / total
    neutral_pct = sentiment_dist['neutral'] / total
    
    summaries = [article['summary'] for article in articles]
    combined_summary = " ".join(summaries)
    overall_sentiment = analyze_sentiment(combined_summary)
    compound_score = overall_sentiment['compound']
    
    analysis = f"{company_name}'s news coverage: {sentiment_dist['positive']} positive, " \
              f"{sentiment_dist['negative']} negative, {sentiment_dist['neutral']} neutral articles. "
    
    if positive_pct > 0.6 and compound_score > 0.2:
        analysis += "Strongly positive outlook suggests growth potential."
    elif negative_pct > 0.6 and compound_score < -0.2:
        analysis += "Strongly negative sentiment indicates challenges."
    elif positive_pct > negative_pct and compound_score > 0:
        analysis += "Generally favorable outlook with stability."
    elif negative_pct > positive_pct and compound_score < 0:
        analysis += "Caution advised due to challenges."
    else:
        analysis += "Mixed outlook with no clear trend."
    
    return analysis

def format_output(company_name, articles):
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            "TITLE": article['title'],
            "SUMMARY": article['summary'],
            "SENTIMENT": article['sentiment']['label'].capitalize(),
            "TOPICS": article['topics']
        })
    
    sentiment_distribution = compare_sentiment(articles)['sentiment_distribution']
    
    return {
        "COMPANY": company_name,
        "ARTICLES": formatted_articles,
        "COMPARATIVE_SENTIMENT_SCORE": {
            "SENTIMENT_DISTRIBUTION": {
                "POSITIVE": sentiment_distribution['positive'],
                "NEGATIVE": sentiment_distribution['negative'],
                "NEUTRAL": sentiment_distribution['neutral']
            }
        },
        "Coverage Differences": analyze_coverage_differences(articles),
        "Topic Overlap": analyze_topic_overlap(articles),
        "Final Sentiment Analysis": generate_final_sentiment(articles, company_name),
        "Audio": "[Play Hindi Speech]"
    }

def analyze_company_news(company_name, num_articles=10):
    articles_to_fetch = num_articles * 2
    fetched_articles = search_company_news(company_name, articles_to_fetch)
    
    if not fetched_articles:
        st.warning(f"No news articles found for {company_name}")
        return None
    
    valid_articles = []
    for article in fetched_articles:
        if len(valid_articles) >= num_articles:
            break
        
        time.sleep(random.uniform(0.5, 1.5))
        content = extract_article_content(article['url'], company_name)
        
        if content.get('valid', False):
            article.update({
                'title': content['title'],
                'text': content['text'],
                'summary': content['summary'],
                'topics': extract_topics(content['summary'])
            })
            article['sentiment'] = analyze_sentiment(article['summary'])
            valid_articles.append(article)
    
    if not valid_articles:
        st.warning(f"No valid news articles found for {company_name}")
        return None
    
    return {
        'company_name': company_name,
        'articles': valid_articles,
        'comparison': compare_sentiment(valid_articles)
    }

# Streamlit app
def main():
    st.title("🔍 Company News Sentiment Analyzer")
    
    company_name = st.text_input("Select a company:")
    num_articles = st.slider("Number of articles to analyze:", min_value=10, step=1)
    
    if st.button("Analyze News"):
        if company_name:
            with st.spinner(f"Searching for news about {company_name}..."):
                results = analyze_company_news(company_name, num_articles)
                
                if results:
                    st.success(f"Analysis completed for {company_name}! Found {len(results['articles'])} articles.")
                    st.json(format_output(company_name, results['articles']))
                else:
                    st.error(f"No valid news articles found for {company_name}.")
        else:
            st.warning("Please select a company.")

if __name__ == "__main__":
    main()