# main.py
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import base64
import io
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from pydantic import BaseModel # <--- ADD THIS IMPORT
from typing import Optional # <--- ADD THIS IMPORT if using Optional
from dotenv import load_dotenv # <--- ADD THIS IMPORT for loading .env

# Load environment variables from .env file
load_dotenv() # <--- ADD THIS CALL

# Initialize FastAPI app
app = FastAPI()
@app.get("/")
def home():
    return {"message": "TDS Project 2 API is running"}

# --- NEW: Define the request body structure using Pydantic BaseModel ---
class AnalyzeDataRequest(BaseModel):
    query: str
    data_url: Optional[str] = None
    # If you later decide to handle direct file uploads (e.g., base64 encoded),
    # you'd add a field like: data_file: Optional[str] = None

# --- Placeholder for LLM Client (Choose one based on your preference) ---
# Example for OpenAI (replace with your actual API key)
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key: # Only try to import if key is present
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
else:
    print("WARNING: OPENAI_API_KEY not found in .env. OpenAI client not initialized.")


# Example for Google Gemini (replace with your actual API key)
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key: # Only try to import if key is present
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    # You might choose a specific model here, e.g., 'gemini-pro', 'gemini-1.5-flash'
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    print("WARNING: GOOGLE_API_KEY not found in .env. Google Gemini client not initialized.")


# --- MODIFIED: Change endpoint path and function signature ---
@app.post("/analyze_data") # Changed path for clarity
async def analyze_data(request_data: AnalyzeDataRequest): # Modified function signature
    try:
        # Instead of reading raw body, access attributes from request_data
        received_query = request_data.query
        received_data_url = request_data.data_url

        print(f"Received query: {received_query}")
        print(f"Received data_url: {received_data_url}")

        # --- LLM Interaction (Core Logic) ---
        # Adjust your conditional logic to use received_query and received_data_url
        if "Scrape the list of highest grossing films from Wikipedia" in received_query:
            if received_data_url and "wikipedia.org/wiki/List_of_highest-grossing_films" in received_data_url:
                return await handle_wikipedia_films_analysis(received_query, received_data_url)
            else:
                raise HTTPException(status_code=400, detail="For Wikipedia film analysis, please provide the correct Wikipedia URL in data_url.")
        elif "The Indian high court judgement dataset" in received_query:
            # For this task, data_url might not be directly used if the S3 path is hardcoded inside the handler
            return await handle_indian_high_court_judgments(received_query)
        else:
            raise HTTPException(status_code=400, detail="Unknown task description. Please provide a supported data analysis task or refine your query.")

    except HTTPException as http_exc:
        # Re-raise HTTPException directly
        raise http_exc
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error during analysis: {e}")
async def handle_wikipedia_films_analysis(query: str, data_url: str):
    """
    Handles the Wikipedia highest-grossing films analysis using pandas.read_html.
    """
    url = data_url

    try:
        # --- NEW: Use pandas.read_html to directly parse tables ---
        # It will return a list of DataFrames. 'match' helps find the right table.
        dfs = pd.read_html(url, match='Worldwide gross', header=0)

        if not dfs:
            raise ValueError("No table found with 'Worldwide gross' header using pandas.read_html. The Wikipedia page structure might have changed significantly.")

        # We assume the first DataFrame that matches 'Worldwide gross' is the correct one
        df = dfs[0]

        # --- Remaining data cleaning and analysis (adjust column names if needed based on read_html output) ---
        # It's crucial to ensure column names from pd.read_html match what's expected below.
        # Print df.columns after read_html if you face issues to verify names.

        # Rename columns to match expected names if pd.read_html returns slightly different ones
        # For example, 'Worldwide gross (billions)' might become 'Worldwide gross' after cleaning
        df.columns = df.columns.str.replace(r'\[.*?\]', '', regex=True).str.strip() # Remove [n 1] type notes from headers
        df.rename(columns={
            'Worldwide gross': 'Worldwide gross', # Ensures consistent name, might be 'Worldwide gross (USD)' etc.
            'Gross': 'Worldwide gross' # Sometimes it's just 'Gross'
            # Add other potential column name mappings if pd.read_html gives different names
        }, inplace=True)
        # ... (code above this point remains unchanged) ...
        # Clean 'Worldwide gross' column
        df['Worldwide gross'] = df['Worldwide gross'].astype(str)
        # Remove currency symbols and commas (using regex=False for literal match)
        df['Worldwide gross'] = df['Worldwide gross'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        # Remove any bracketed notes like '[n 1]', '[C]', etc. (using regex=True)
        df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'\[.*?\]', '', regex=True)
        # Convert to float, coercing errors to NaN. This will turn 'T2257844554' into NaN.
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')

        # Clean 'Year' column (this regex is also updated for robustness)
        df['Year'] = df['Year'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.strip() # Remove notes like [n 1] from year
        df['Year'] = pd.to_numeric(df['Year'].str[:4], errors='coerce') # Take first 4 chars for year

        # Drop rows with NaN values in critical columns
        df.dropna(subset=['Worldwide gross', 'Year', 'Rank'], inplace=True)

        # ... (rest of your analysis code below this point remains unchanged) ...

        # 1. How many $2 bn movies were released before 2020?
        billion_2_threshold = 2_000_000_000 # 2 billion
        q1_answer = df[(df['Worldwide gross'] >= billion_2_threshold) & (df['Year'] < 2020)].shape[0]

        # 2. Which is the earliest film that grossed over $1.5 bn?
        billion_1_5_threshold = 1_500_000_000 # 1.5 billion
        # Find movies >= 1.5bn, sort by year, then by worldwide gross (desc)
        # If multiple have the same earliest year, pick the highest grossing for that year.
        earliest_film_df = df[df['Worldwide gross'] >= billion_1_5_threshold].sort_values(
            by=['Year', 'Worldwide gross'], ascending=[True, False]
        ).iloc[0]
        q2_answer = earliest_film_df['Title']

        # 3. What's the correlation between the Rank and Peak?
        # Assuming 'Peak' refers to 'Worldwide gross' as there's no 'Peak' column.
        # This is an interpretation. The LLM needs to make such interpretations.
        # If 'Peak' was meant to be another column, the LLM should identify it.
        q3_answer = df['Rank'].corr(df['Worldwide gross'])

        # 4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
        # Assume Peak is Worldwide gross
        plt.figure(figsize=(8, 6))
        sns.regplot(x='Rank', y='Worldwide gross', data=df,
                                 line_kws={'color':'red', 'linestyle':'--'}, scatter_kws={'alpha':0.6})
        plt.title('Scatterplot of Rank vs Worldwide Gross')
        plt.xlabel('Rank')
        plt.ylabel('Worldwide Gross ($)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100) # dpi to control size
        buf.seek(0)
        img_bytes = buf.getvalue()
        plt.close() # Close plot to free memory

        # Base64 encode the image
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_data_uri = f"data:image/png;base64,{img_base64}"

        # Check image size
        if len(img_bytes) > 100_000:
            print(f"WARNING: Image size is {len(img_bytes)} bytes, which is over 100KB.")
            # You might need to adjust DPI or figure size to reduce size

        return {
            "query_1_answer": q1_answer,
            "query_2_answer": q2_answer,
            "query_3_answer": q3_answer,
            "plot_image_data_uri": img_data_uri
        }

    except Exception as e:
        print(f"Error handling Wikipedia analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing Wikipedia task: {e}")