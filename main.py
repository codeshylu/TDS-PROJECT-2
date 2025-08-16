from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import base64
import io

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    data_url: str = None  # optional, only needed for scrape task

# ---- Helper: scrape Wikipedia table ----
def scrape_wikipedia(url: str):
    tables = pd.read_html(url)
    df = tables[0]  # first table = highest-grossing films
    # Ensure consistent columns
    df = df.rename(columns={df.columns[0]: "Title", df.columns[1]: "Year", df.columns[2]: "Worldwide gross"})
    # Clean gross column
    df["Worldwide gross"] = (
        df["Worldwide gross"].replace(r"[\$,]", "", regex=True).astype(float) / 1e9
    )
    return df

# ---- Helper: create chart and return base64 ----
def make_chart(df, title="Top Movies"):
    plt.figure(figsize=(8, 5))
    plt.bar(df["Title"], df["Worldwide gross"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Gross (in $ billions)")
    plt.title(title)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

@app.post("/")
def handle_query(request: QueryRequest):
    query = request.query.lower()

    # ---- Case 1: Scrape highest grossing films ----
    if "scrape" in query and "highest grossing films" in query:
        if not request.data_url:
            raise HTTPException(status_code=400, detail="Missing data_url for scrape task")

        df = scrape_wikipedia(request.data_url)
        chart = make_chart(df.head(10), "Top 10 Grossing Films")

        # Format response as: [index, title, gross, chart]
        response = []
        for idx, row in df.head(10).iterrows():
            response.append([idx, row["Title"], round(row["Worldwide gross"], 6), f"data:image/png;base64,{chart}"])
        return response

    # ---- Case 2: Count $2bn movies before 2000 ----
    elif "$2 bn" in query and "before 2000" in query:
        if not request.data_url:
            raise HTTPException(status_code=400, detail="Missing data_url for scrape task")

        df = scrape_wikipedia(request.data_url)
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

        filtered = df[(df["Worldwide gross"] >= 2.0) & (df["Year"] < 2000)]
        chart = make_chart(filtered, "$2bn Movies before 2000")

        response = []
        for idx, row in filtered.iterrows():
            response.append([idx, row["Title"], round(row["Worldwide gross"], 6), f"data:image/png;base64,{chart}"])
        return response

    else:
        raise HTTPException(status_code=400, detail="Unknown task description. Please provide a supported query.")





