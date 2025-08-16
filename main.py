from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import base64, io, re
import numpy as np
from bs4 import BeautifulSoup
import requests

app = FastAPI()

# --- Helper to scrape Wikipedia table ---
def scrape_highest_grossing_films(url: str) -> pd.DataFrame:
    try:
        # Use requests + BeautifulSoup to fetch page content
        response = requests.get(url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {response.status_code}")
        tables = pd.read_html(response.text)
        df = tables[0]

        # Standardize column names
        df.columns = [c.strip() for c in df.columns]

        # Clean Year column
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        # Clean Rank and Peak columns
        if "Rank" in df.columns:
            df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
        if "Peak" in df.columns:
            df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")

        # Clean Worldwide gross (remove $ , bn , m)
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

        # --- Count of movies ≥ $2bn before 2000 ---
        count_2bn_before_2000 = int(len(df[(df["Gross"] >= 2_000_000_000) & (df["Year"] < 2000)]))

        # --- Earliest movie ≥ $1.5bn ---
        earliest_movie_1_5bn = df[df["Gross"] >= 1_500_000_000].sort_values("Year").iloc[0]["Title"]

        # --- Correlation between Rank and Peak ---
        corr_rank_peak = float(df["Rank"].corr(df["Peak"]))

        # --- Scatterplot Rank vs Peak ---
        plt.scatter(df["Rank"], df["Peak"])
        m, b = np.polyfit(df["Rank"].dropna(), df["Peak"].dropna(), 1)
        plt.plot(df["Rank"], m*df["Rank"]+b, linestyle="dotted", color="red")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode()
        plt.close()
        scatterplot_base64 = f"data:image/png;base64,{encoded}"

        # --- Return all results in a single list ---
        return [count_2bn_before_2000, earliest_movie_1_5bn, corr_rank_peak, scatterplot_base64]

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")


















