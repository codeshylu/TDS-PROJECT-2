# # main.py

from fastapi import FastAPI, HTTPException
import uvicorn
import base64
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Get API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize clients only if keys exist
client = None
gemini_model = None

if openai_api_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
    except ImportError:
        print("OpenAI package not installed. Skipping OpenAI client.")

if google_api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
    except ImportError:
        print("Google generativeai package not installed. Skipping Google Gemini.")

# Root endpoint
@app.get("/")
async def home():
    return {"message": "TDS Project 2 API is running"}

# Request body model
class AnalyzeDataRequest(BaseModel):
    query: str
    data_url: Optional[str] = None

# Main POST endpoint
@app.post("/analyze_data")
async def analyze_data(request_data: AnalyzeDataRequest):
    query = request_data.query
    data_url = request_data.data_url

    try:
        if "Scrape the list of highest grossing films from Wikipedia" in query:
            if data_url and "wikipedia.org/wiki/List_of_highest-grossing_films" in data_url:
                return await handle_wikipedia_films_analysis(query, data_url)
            else:
                raise HTTPException(status_code=400, detail="Please provide the correct Wikipedia URL in data_url.")
        elif "The Indian high court judgement dataset" in query:
            return await handle_indian_high_court_judgments(query)
        else:
            raise HTTPException(status_code=400, detail="Unknown task description. Please provide a supported query.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during analysis: {e}")

# Wikipedia films handler
async def handle_wikipedia_films_analysis(query: str, data_url: str):
    try:
        dfs = pd.read_html(data_url, match='Worldwide gross', header=0)
        if not dfs:
            raise ValueError("No table found with 'Worldwide gross' header.")
        df = dfs[0]
        df.columns = df.columns.str.replace(r'\[.*?\]', '', regex=True).str.strip()
        df.rename(columns={'Gross': 'Worldwide gross'}, inplace=True)
        df['Worldwide gross'] = (
            df['Worldwide gross'].astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(r'\[.*?\]', '', regex=True)
        )
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'].astype(str).str[:4], errors='coerce')
        df.dropna(subset=['Worldwide gross', 'Year', 'Rank'], inplace=True)

        # Query answers
        q1 = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2020)].shape[0]
        earliest_film = df[df['Worldwide gross'] >= 1_500_000_000].sort_values(
            by=['Year', 'Worldwide gross'], ascending=[True, False]
        ).iloc[0]['Title']
        q3 = df['Rank'].corr(df['Worldwide gross'])

        # Scatterplot
        plt.figure(figsize=(8, 6))
        sns.regplot(
            x='Rank', y='Worldwide gross', data=df,
            line_kws={'color':'red', 'linestyle':'--'}, scatter_kws={'alpha':0.6}
        )
        plt.title('Rank vs Worldwide Gross')
