from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import base64, io
import numpy as np
import requests

app = FastAPI()

# --- Helper to scrape Wikipedia table ---
def scrape_highest_grossing_films(url: str) -> pd.DataFrame:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html = response.text

        # Parse tables from HTML content (not URL)
        tables = pd.read_html(html)
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
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@app.post("/analyze_data")
async def analyze_data(request: Request):
    try:
        body = await request.json()
        query = body.get("query", "").lower()
        data_url = body.get(
            "data_url", "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        )

        df = scrape_highest_grossing_films(data_url)

        # --- Question 1 ---
        if "2 bn" in query and "before 2000" in query:
            count = len(df[(df["Gross"] >= 2_000_000_000) & (df["Year"] < 2000)])
            return [count]

        # --- Question 2 ---
        elif "earliest" in query and "1.5 bn" in query:
            film = df[df["Gross"] >= 1_500_000_000].sort_values("Year").iloc[0]
            return [film["Title"]]

        # --- Question 3 ---
        elif "correlation" in query and "rank" in query and "peak" in query:
            corr = df["Rank"].corr(df["Peak"])
            return [corr]

        # --- Question 4 ---
        elif "scatterplot" in query and "rank" in query and "peak" in query:
            plt.figure(figsize=(6,4))
            plt.scatter(df["Rank"], df["Peak"])
            m, b = np.polyfit(df["Rank"].dropna(), df["Peak"].dropna(), 1)
            plt.plot(df["Rank"], m*df["Rank"]+b, linestyle="dotted", color="red")
            plt.xlabel("Rank")
            plt.ylabel("Peak")
            plt.title("Rank vs Peak")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode()
            plt.close()

            return [f"data:image/png;base64,{encoded}"]

        else:
            raise HTTPException(
                status_code=400, detail="Unknown task description. Please provide a supported query."
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}













