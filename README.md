# Company News Sentiment Analyzer

A Streamlit app and FastAPI service to analyze news sentiment for a given company using SiEBERT and spaCy.

## Project Setup

Follow these steps to install and run the application locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/company-news-analyzer.git
   cd company-news-analyzer
2. **Install dependencies: Ensure you have Python 3.8 or higher installed. Then, install the required packages:**
   pip install -r requirements.txt
   **Install the spaCy English model:**
   python -m spacy download en_core_web_sm
3. **Run the streamlit app:**
   streamlit run app.py
4. **streamlit run app.py**
   uvicorn api:app --reload


## Model Details 
The project uses the following models and techniques for its core functionalities:
  **Sentiment Analysis:**
    1. **Model: SiEBERT (siebert/sentiment-roberta-large-english)**
    A RoBERTa-large model fine-tuned for sentiment analysis in English.
    Provides binary sentiment classification ("positive" or "negative") with a confidence score.
    Used to analyze the sentiment of news article summaries.
    Source: Hugging Face Transformers library.
    Truncation is enabled to handle long inputs (max 512 tokens).
  **Topic Extraction:**
    Model: spaCy (en_core_web_sm)
    A small English language model for natural language processing.
    Used to extract entities (e.g., organizations, products, events) from article summaries.
    Entities are matched against predefined topic categories (e.g., "Technology", "Outages") to identify relevant topics.
    Source: spaCy library.
  **Summarization:**
    No pre-trained model is used for summarization. Instead, a rule-based approach is implemented:
    Extracts sentences from the article that mention the company name.
    Selects the first 5 sentences mentioning the company (or the first 7 sentences if fewer are available).
    Limits the summary to 512 characters to avoid exceeding model input limits.
  **Text-to-Speech (TTS):**
    TTS is not implemented in the current version of the project. The output includes a placeholder for audio ("Audio": "[Play Hindi Speech]"), but no TTS model is integrated.
    If TTS were to be added, a model like gTTS (Google Text-to-Speech) or pyttsx3 could be used to generate Hindi speech from the final sentiment analysis.
## Usage
    The project includes a FastAPI-based API to programmatically analyze news sentiment for a company. The API is defined in api.py and provides the following endpoint:
    Endpoint: GET /analyze/{company_name}
  **Parameters:**
    company_name (path parameter): The name of the company to analyze (e.g., "Tesla").
    num_articles (query parameter, optional): Number of articles to fetch (default: 10, max: 20).
    Response: A JSON object containing the company name, articles, sentiment distribution, and final sentiment analysis.
  **How to Access the API:**
    Run the API locally:
    uvicorn api:app --reload
  **Use Postman, curl, or the FastAPI interactive docs to test the endpoint:**
    EXAMPLE : curl "http://localhost:8000/analyze/Tesla?num_articles=5"
  **Using Postman:**
    Open Postman.
    Create a new GET request.
    Set the URL to http://localhost:8000/analyze/Tesla?num_articles=5.
    Send the request and view the JSON response.
  **Using FastAPI Docs:**
    Visit http://localhost:8000/docs in your browser.
    Use the interactive interface to test the /analyze/{company_name} endpoint.
  **API Usage**
    This project does not use any third-party APIs. Instead, it:
    
    Scrapes news articles directly from Google News using the requests and BeautifulSoup libraries.
    Performs sentiment analysis and topic extraction locally using pre-trained models (SiEBERT and spaCy).

  ## Assumptions and Limitations##
  **Assumptions**
    Internet Access: The application assumes a stable internet connection to fetch news articles from Google News and download pre-trained models (SiEBERT, spaCy) on first run.
    Article Availability: It assumes that Google News has recent, relevant articles for the specified company. If no articles are found, the app will display an error message.
    Sentiment Model Accuracy: The SiEBERT model is assumed to provide accurate sentiment analysis for English news articles. It may misclassify sentiment in cases of sarcasm, complex language, or domain-specific jargon.
    Topic Extraction: The topic extraction logic assumes that predefined categories (e.g., "Technology", "Outages") cover most news topics. Uncommon topics may be classified as "General News."
    Language: The app assumes all news articles are in English, as the models (SiEBERT, spaCy) are trained on English data.

**Limitations**
    Resource Intensive: The SiEBERT model is large (~1.4 GB) and may cause memory issues on low-resource environments like Hugging Face Spaces’ free tier. Reducing the number of articles (e.g., to 5) can mitigate this.
    Rate Limiting: Google News may block requests if too many are made in a short time. The app includes delays to reduce this risk, but it may still fail under heavy use.
    API Deployment: The FastAPI-based API cannot be deployed on Hugging Face Spaces, as Spaces only supports Streamlit or Gradio apps. The API must be run locally.
    Article Quality: The app may fail to extract meaningful content from articles behind paywalls, with heavy JavaScript, or lacking proper HTML structure.
    Sentiment Granularity: SiEBERT provides binary sentiment (positive/negative). Neutral sentiment is inferred based on a confidence threshold (<0.6), which may not always be accurate.

    
