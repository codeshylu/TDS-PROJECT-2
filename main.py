from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import requests
import re

app = FastAPI()

class AnalyzeRequest(BaseModel):
    query: str
    data_url: str

@app.get("/")
def root():
    return {"message": "API is running. Use POST /analyze_data with your request."}

def parse_query(query: str):
    """
    Parse queries like:
    'How many $2B movies were released before 2000?'
    Supports: $X B, before YEAR, after YEAR
    Returns: threshold (float in billions), year_min (int), year_max (int)
    """
    money_match = re.search(r"\$(\d+\.?\d*)\s*B", query, re.IGNORECASE)
    before_match = re.search(r"before\s+(\d{4})", query, re.IGNORECASE)
    after_match = re.search(r"after\s+(\d{4})", query, re.IGNORECASE)
    
    threshold = float(money_match.group(1)) if money_match else 0
    year_max = int(before_match.group(1)) if before_match else 9999
    year_min = int(after_match.group(1)) if after_match else 0
    
    return threshold, year_min, year_max

@app.post("/analyze_data")
def analyze_data(request: AnalyzeRequest):
    try:
        # Fetch data
        response = requests.get(request.data_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {response.status_code}")

        # Parse tables
        tables = pd.read_html(response.text)
        if len(tables) == 0:
            raise HTTPException(status_code=500, detail="No tables found at the URL")
        
        df = tables[0]  # first table assumed relevant

        # Clean numeric columns safely
        if "Worldwide gross" in df.columns:
            df["Worldwide gross"] = (
                df["Worldwide gross"]
                .replace("[\$,]", "", regex=True)
                .apply(pd.to_numeric, errors="coerce") / 1e9
            )
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

        # Parse query
        threshold, year_min, year_max = parse_query(request.query)

        # Filter safely
        filtered_df = df[
            (df.get("Worldwide gross", pd.Series([0]*len(df))) >= threshold) &
            (df.get("Year", pd.Series([0]*len(df))) >= year_min) &
            (df.get("Year", pd.Series([0]*len(df))) < year_max)
        ]

        # Scatter plot if data available
        img_base64 = ""
        if "Rank" in df.columns and "Peak" in df.columns:
            scatter_df = df.dropna(subset=["Rank", "Peak"])
            if not scatter_df.empty:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x="Rank", y="Peak", data=scatter_df)
                try:
                    m, b = np.polyfit(scatter_df["Rank"], scatter_df["Peak"], 1)
                    plt.plot(scatter_df["Rank"], m*scatter_df["Rank"] + b, linestyle="dotted", color="red")
                except Exception:
                    pass
                plt.title("Rank vs Peak")
                buf = BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Prepare safe response
        first_title = str(filtered_df.iloc[0]["Title"]) if not filtered_df.empty and "Title" in filtered_df.columns else "N/A"
        first_gross = str(filtered_df["Worldwide gross"].iloc[0]) if not filtered_df.empty and "Worldwide gross" in filtered_df.columns else 0.0

        result = [
            len(filtered_df),
            first_title,
            first_gross,
            f"data:image/png;base64,{img_base64}" if img_base64 else None
        ]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")























