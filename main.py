# from fastapi import FastAPI, HTTPException
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

# Load environment variables from .env (local dev only)
load_dotenv()

# Get API keys (Render env vars take priority)
openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize LLM clients if keys exist
if openai_api_key:
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
else:
    print("OpenAI API key not found. Skipping OpenAI client initialization.")

if google_api_key:
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    print("Google API key not found. Skipping Google Gemini initialization.")

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "TDS Project 2 API is running"}

# Request body model
class AnalyzeDataRequest(BaseModel):
    query: str
    data_url: Optional[str] = None

@app.post("/analyze_data")
async def analyze_data(request_data: AnalyzeDataRequest):
    try:
        query = request_data.query
        data_url = request_data.data_url

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

async def handle_wikipedia_films_analysis(query: str, data_url: str):
    try:
        dfs = pd.read_html(data_url, match='Worldwide gross', header=0)
        if not dfs:
            raise ValueError("No table found with 'Worldwide gross' header.")

        df = dfs[0]
        df.columns = df.columns.str.replace(r'\[.*?\]', '', regex=True).str.strip()
        df.rename(columns={'Gross': 'Worldwide gross'}, inplace=True)

        df['Worldwide gross'] = (
            df['Worldwide gross']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.replace(r'\[.*?\]', '', regex=True)
        )
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')

        df['Year'] = (
            df['Year']
            .astype(str)
            .str.replace(r'\[.*?\]', '', regex=True)
            .str.strip()
        )
        df['Year'] = pd.to_numeric(df['Year'].str[:4], errors='coerce')

        df.dropna(subset=['Worldwide gross', 'Year', 'Rank'], inplace=True)

        # 1. $2B before 2020
        q1 = df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2020)].shape[0]
        # 2. Earliest > $1.5B
        earliest_film = df[df['Worldwide gross'] >= 1_500_000_000].sort_values(
            by=['Year', 'Worldwide gross'], ascending=[True, False]
        ).iloc[0]['Title']
        # 3. Correlation
        q3 = df['Rank'].corr(df['Worldwide gross'])

        # Scatterplot
        plt.figure(figsize=(8, 6))
        sns.regplot(
            x='Rank', y='Worldwide gross', data=df,
            line_kws={'color':'red', 'linestyle':'--'}, scatter_kws={'alpha':0.6}
        )
        plt.title('Rank vs Worldwide Gross')
        plt.xlabel('Rank')
        plt.ylabel('Worldwide Gross ($)')
        plt.grid(True, linestyle='--', alpha=0.7)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return {
            "query_1_answer": q1,
            "query_2_answer": earliest_film,
            "query_3_answer": q3,
            "plot_image_data_uri": f"data:image/png;base64,{img_base64}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Wikipedia task: {e}")

# Placeholder for Indian high court judgments handler
async def handle_indian_high_court_judgments(query: str):
    return {"message": "Indian high court judgments handler not implemented yet."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
