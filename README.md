# Company News Sentiment Analyzer
A Streamlit app and FastAPI service to analyze news sentiment for a given company using SiEBERT and spaCy.

## Project Setup
Follow these steps to install and run the application locally:

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/company-news-analyzer.git
cd company-news-analyzer
```

2. **Install Dependencies:**  
Ensure you have Python 3.8 or higher installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

3. **Install spaCy Model:**  
Download the English language model for spaCy:
```bash
python -m spacy download en_core_web_sm
```

4. **Run the Streamlit App:**
```bash
streamlit run app.py
```

5. **Run the FastAPI Service:**
```bash
uvicorn api:app --reload
```

## Model Details
The project uses the following models and techniques:

### **Sentiment Analysis**
- **Model:** SiEBERT (`siebert/sentiment-roberta-large-english`).
- **Description:** A RoBERTa-large model fine-tuned for sentiment analysis.
- **Output:** Binary classification (`positive` or `negative`) with a confidence score.
- **Truncation:** Enabled for inputs over 512 tokens.
- **Source:** Hugging Face Transformers Library.

### **Topic Extraction**
- **Model:** spaCy (`en_core_web_sm`).
- **Description:** A small English language model for natural language processing.
- **Function:** Extracts entities (e.g., organizations, products, events) from article summaries.
- **Topic Matching:** Entities are matched against predefined categories (e.g., `Technology`, `Outages`).
- **Source:** spaCy Library.

### **Summarization**
- No pre-trained model is used for summarization.
- **Method:**
  - Extract sentences mentioning the company name.
  - Select the first 5 sentences (or up to 7 if fewer are available).
  - Limit summary to 512 characters.

### **Text-to-Speech (TTS)**
- **Note:** TTS is not implemented in the current version.
- **Placeholder:** `[Play Hindi Speech]`
- **Future Options:** Models like `gTTS` (Google Text-to-Speech) or `pyttsx3` can be integrated for Hindi speech output.

## API Usage
The FastAPI-based API allows for programmatic news sentiment analysis.

### **Endpoint**
- **GET /analyze/{company_name}**

### **Parameters**
- `company_name` (path parameter): The company name to analyze (e.g., `Tesla`).
- `num_articles` (optional query parameter): Number of articles to fetch (default: 10, max: 20).

### **Response**
A JSON object containing:
- Company name
- Articles analyzed
- Sentiment distribution
- Final sentiment analysis

### **Examples**

**Using curl:**
```bash
curl "http://localhost:8000/analyze/Tesla?num_articles=5"
```

**Using Postman:**
1. Open Postman.
2. Create a new GET request.
3. Enter URL: `http://localhost:8000/analyze/Tesla?num_articles=5`
4. Click Send.

**Using FastAPI Docs:**
1. Run the API (`uvicorn api:app --reload`).
2. Visit [http://localhost:8000/docs](http://localhost:8000/docs).
3. Test the `/analyze/{company_name}` endpoint using the interactive API UI.

## Assumptions and Limitations

### **Assumptions**
- **Internet Access:** Required to fetch news from Google News and download models.
- **Article Availability:** Assumes Google News has recent articles for the specified company.
- **Model Accuracy:** SiEBERT may misclassify sentiment in cases involving sarcasm or domain-specific jargon.
- **Language Support:** The app assumes all articles are in English.

### **Limitations**
- **Resource Intensive:** SiEBERT is large (~1.4 GB) and may cause memory issues on low-resource machines.
- **Rate Limiting:** Google News may block excessive requests. Delays are added to minimize this.
- **API Deployment:** FastAPI cannot be deployed on Hugging Face Spaces. It must be run locally.
- **Article Quality:** Articles behind paywalls or with poor HTML structure may not be parsed correctly.
- **Sentiment Granularity:** Only positive or negative sentiment is classified. Neutral sentiment is inferred if the confidence score is below 0.6.

## Future Improvements
- Implement Hindi text-to-speech (TTS) using gTTS or pyttsx3.
- Enhance sentiment analysis by using a more granular sentiment model.
- Add support for multiple languages.
- Improve rate limiting mechanisms for more stable web scraping.

## Conclusion
This project provides a simple and effective way to analyze news sentiment for companies using state-of-the-art NLP models. For any contributions or issues, please raise a GitHub issue or pull request.

