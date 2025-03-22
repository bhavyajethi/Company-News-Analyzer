# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import analyze_company_news, format_output
import uvicorn

app = FastAPI(
    title="Company News Sentiment Analyzer API",
    description="API to analyze news sentiment for a given company.",
    version="1.0.0"
)

# Response model for structured output
class AnalysisResponse(BaseModel):
    company: str
    articles: list
    comparative_sentiment_score: dict
    final_sentiment_analysis: str

@app.get("/analyze/{company_name}", response_model=AnalysisResponse)
async def analyze_company(company_name: str, num_articles: int = 10):
    """
    Analyze news sentiment for a given company.

    Args:
        company_name (str): Name of the company to analyze.
        num_articles (int, optional): Number of articles to fetch (default: 10, max: 20).

    Returns:
        dict: Structured JSON response with analysis results.

    Raises:
        HTTPException: If no articles are found or an error occurs.
    """
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name cannot be empty.")
    
    if num_articles < 1 or num_articles > 20:
        raise HTTPException(status_code=400, detail="Number of articles must be between 1 and 20.")
    
    try:
        # Perform analysis using utility functions
        results = analyze_company_news(company_name, num_articles)
        if not results or not results.get("articles"):
            raise HTTPException(status_code=404, detail=f"No valid news articles found for {company_name}.")
        
        # Format the output
        formatted_result = format_output(company_name, results["articles"])
        return formatted_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Optional: Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the API locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)