# app.py
import streamlit as st
from utils import analyze_company_news, format_output

def main():
    """Main function to run the Streamlit app."""
    st.title("🔍 Company News Sentiment Analyzer")
    
    company_name = st.text_input("Enter a company name:")
    num_articles = st.slider("Number of articles to analyze:", min_value=10, step=1)
    
    if st.button("Analyze News"):
        if not company_name:
            st.warning("Please enter a company name.")
            return
        
        with st.spinner(f"Searching for news about {company_name}..."):
            results = analyze_company_news(company_name, num_articles)
            if results:
                st.success(f"Analysis completed for {company_name}! Found {len(results['articles'])} articles.")
                st.json(format_output(company_name, results['articles']))
            else:
                st.error(f"No valid news articles found for {company_name}.")

if __name__ == "__main__":
    main()