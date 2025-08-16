from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import base64, io, re
import numpy as np
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# --- Helper to fetch and scrape Wikipedia table ---
def scrape_highest_grossing_films(url: str) -> pd.DataFrame:
    try:
        # Try fetching the page
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {response.status_code}")

        # Use pandas to read HTML tables
        tables = pd.read_html(response.text)
        if not tables:
            raise HTTPException(status_code=500, detail="No tables found on the page")

        df = tables[0]

        # Standardize column names
        df.columns = [c.strip() for c in df.columns]

        # Clean numeric columns
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        if "Rank" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        if "Peak" in df.columns:
            df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")
        if "Worldwide gross" in df.columns:
            df["Gross"] = (
                df["Worldwide gross"]
                .astype(str)
                .replace(r"[\$,]", "", regex=True)
                .replace(r"bn", "000000000", regex=True)
                .replace(r"m", "000000", regex=True)
            )
            df["Gross"] = pd.to_numeric(df["Gross"], errors="coerce")

        return df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

@app.post("/analyze_data")
async def analyze_data(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "")
        data_url = body.get("data_url", "https://en.wikipedia.org/wiki/List_of_highest-grossing_films")

        df = scrape_highest_grossing_films(data_url)

        # --- Question 1: Count $2bn movies before 2000 ---
        count_2bn_before_2000 = len(df[(df["Gross"] >= 2_000_000_000) & (df["Year"] < 2000)])

        # --- Question 2: Earliest movie above 1.5bn ---
        film_1_5bn = df[df["Gross"] >= 1_500_000_000].sort_values("Year").iloc[0]["Title"]

        # --- Question 3: Correlation Rank vs Peak ---
        corr_rank_peak = df["Rank"].corr(df["Peak"])

        # --- Question 4: Scatterplot Rank vs Peak ---
        plt.scatter(df["Rank"], df["Peak"])
        m, b = np.polyfit(df["Rank"].dropna(), df["Peak"].dropna(), 1)
        plt.plot(df["Rank"], m*df["Rank"]+b, linestyle="dotted", color="red")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        plt.close()

        return [count_2bn_before_2000, film_1_5bn, corr_rank_peak, f"data:image/png;base64,{encoded}"]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

















