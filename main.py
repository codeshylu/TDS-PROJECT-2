from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from scipy.stats import linregress

app = FastAPI()

# Root endpoint
@app.get("/")
def home():
    return {"message": "TDS Project 2 API is running"}

# Request body
class AnalyzeDataRequest(BaseModel):
    query: str
    data_url: Optional[str] = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"


# Helper: scrape Wikipedia table
def scrape_table(url):
    try:
        tables = pd.read_html(url)
        df = tables[0]  # first table on page (highest-grossing films)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


# Helper: make scatterplot and return base64
ndef make_scatterplot(df):
    try:
        x = pd.to_numeric(df["Rank"], errors="coerce")
        y = pd.to_numeric(df["Peak"], errors="coerce")
        mask = x.notna() & y.notna()
        x = x[mask]
        y = y[mask]

        slope, intercept, r_value, _, _ = linregress(x, y)
        line = slope * x + intercept

        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, alpha=0.6)
        plt.plot(x, line, "r--")
        plt.xlabel("Rank")
        plt.ylabel("Peak")
        plt.title("Rank vs Peak with Regression Line")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_base64}"
        return data_uri[:100000]  # keep under 100KB
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plotting failed: {str(e)}")


@app.post("/analyze_data")
async def analyze_data(request_data: AnalyzeDataRequest):
    query = request_data.query.strip()
    url = request_data.data_url
    df = scrape_table(url)

    results = []

    if "How many $2 bn movies were released before 2000" in query:
        # Gross column contains $ strings, parse billions
        df_copy = df.copy()
        df_copy["Worldwide gross"] = df_copy["Worldwide gross"].replace("[$,]", "", regex=True)
        df_copy["Worldwide gross"] = pd.to_numeric(df_copy["Worldwide gross"], errors="coerce") / 1e9
        df_copy["Year"] = pd.to_numeric(df_copy["Year"], errors="coerce")
        count = df_copy[(df_copy["Worldwide gross"] >= 2) & (df_copy["Year"] < 2000)].shape[0]
        results.append(str(count))

    elif "earliest film that grossed over $1.5 bn" in query:
        df_copy = df.copy()
        df_copy["Worldwide gross"] = df_copy["Worldwide gross"].replace("[$,]", "", regex=True)
        df_copy["Worldwide gross"] = pd.to_numeric(df_copy["Worldwide gross"], errors="coerce") / 1e9
        df_copy["Year"] = pd.to_numeric(df_copy["Year"], errors="coerce")
        filtered = df_copy[df_copy["Worldwide gross"] > 1.5]
        if not filtered.empty:
            earliest = filtered.sort_values("Year").iloc[0]["Title"]
            results.append(str(earliest))
        else:
            results.append("None")

    elif "correlation between the Rank and Peak" in query:
        try:
            corr = df["Rank"].corr(df["Peak"])
            results.append(str(round(corr, 6)))
        except Exception:
            results.append("Correlation unavailable")

    elif "scatterplot of Rank and Peak" in query:
        data_uri = make_scatterplot(df)
        results.append(data_uri)

    else:
        raise HTTPException(status_code=400, detail="Unknown task description. Please provide a supported query.")

    return results



