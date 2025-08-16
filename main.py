from fastapi import FastAPI, Request, HTTPException
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64, io

app = FastAPI()

# --- Helper to scrape Wikipedia table ---
def scrape_highest_grossing_films(url: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(url)
        df = tables[0]  # First table is usually the highest-grossing films
        df.columns = [c.strip() for c in df.columns]

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

        # --- Question 1 ---
        if "2 bn" in query and "before 2000" in query:
            answer = [str(len(df[(df["Gross"] >= 2_000_000_000) & (df["Year"] < 2000)]))]
            return answer

        # --- Question 2 ---
        elif "earliest" in query and "1.5 bn" in query:
            film = df[df["Gross"] >= 1_500_000_000].sort_values("Year").iloc[0]
            answer = [str(film["Title"])]
            return answer

        # --- Question 3 ---
        elif "correlation" in query and "Rank" in query and "Peak" in query:
            corr = df["Rank"].corr(df["Peak"])
            answer = [str(corr)]
            return answer

        # --- Question 4 ---
        elif "scatterplot" in query and "Rank" in query and "Peak" in query:
            plt.scatter(df["Rank"], df["Peak"])
            m, b = np.polyfit(df["Rank"].dropna(), df["Peak"].dropna(), 1)
            plt.plot(df["Rank"], m*df["Rank"]+b, linestyle="dotted", color="red")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode()
            plt.close()
            answer = [f"data:image/png;base64,{encoded}"]
            return answer

        else:
            raise HTTPException(status_code=400, detail="Unknown task description. Please provide a supported query.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")















