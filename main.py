# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any
import os
import io
import re
import base64
import requests
import pandas as pd
import numpy as np

# Headless plotting for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- FastAPI app ----------------
app = FastAPI(title="TDS Project 2 API")

WIKI_FILMS_URL = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"


class AnalyzeDataRequest(BaseModel):
    query: str
    data_url: Optional[str] = None


@app.get("/")
def home():
    return {"message": "TDS Project 2 API is running"}


@app.post("/analyze_data")
async def analyze_data(body: AnalyzeDataRequest) -> List[Any]:
    """
    For queries that reference the Wikipedia 'highest-grossing films' dataset,
    this endpoint computes and returns a 4-element JSON array:

    [ <answer1:int>, <answer2:str>, <answer3:float>, <answer4:data_uri:str> ]

    1) How many $2 bn movies were released before 2000?
    2) Which is the earliest film that grossed over $1.5 bn?
    3) What's the correlation between the Rank and Peak? (Pearson)
    4) A scatterplot of Rank (x) vs Peak (y) with a dotted red regression line,
       returned as a base64 data URI (PNG).
    """
    url = (body.data_url or "").strip()
    q = (body.query or "").lower()

    # Accept if either URL or query points to the target dataset
    if (not url and "highest grossing" in q and "film" in q) or (
        WIKI_FILMS_URL.split("https://")[-1] in (url or "")
        or "wikipedia.org/wiki/list_of_highest-grossing_films" in (url or "").lower()
    ):
        if not url:
            url = WIKI_FILMS_URL
        try:
            df = _load_highest_grossing_films(url)
            answers = _compute_answers_and_plot(df)
            return answers  # [count_2bn_before_2000, earliest_1_5bn_title, corr, data_uri]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    # If it’s some other dataset/task, keep the original behavior (explicit error)
    raise HTTPException(
        status_code=400,
        detail="Unknown task description. Please provide a supported query."
    )


# ---------------- Helpers ----------------

def _fetch_html(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text


def _load_highest_grossing_films(url: str) -> pd.DataFrame:
    """Load the table with Rank/Peak/Title/Year/Worldwide gross."""
    html = _fetch_html(url)
    tables = pd.read_html(html, flavor="lxml")

    target = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if ("rank" in cols) and ("peak" in cols) and ("title" in cols):
            target = t.copy()
            break
    if target is None:
        # Fallback: pick table containing 'rank' and 'title'
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if ("rank" in cols) and ("title" in cols):
                target = t.copy()
                break
    if target is None:
        raise ValueError("Could not find expected films table on the page.")

    # Normalize column names (remove footnotes like [a], [1], etc.)
    target.columns = (
        target.columns.astype(str)
        .str.replace(r"\[.*?\]", "", regex=True)
        .str.strip()
    )

    # Identify needed columns
    def _find_col(name_part: str) -> str:
        for c in target.columns:
            if name_part.lower() in c.lower():
                return c
        raise ValueError(f"Expected column containing '{name_part}' not found.")

    col_rank = _find_col("rank")
    col_peak = _find_col("peak")
    col_title = _find_col("title")
    # "Worldwide gross" sometimes appears as "Worldwide gross" or "Gross"
    col_gross = None
    for key in ("worldwide gross", "gross"):
        try:
            col_gross = _find_col(key)
            break
        except Exception:
            pass
    if not col_gross:
        raise ValueError("Gross column not found.")

    # Optional Year column
    try:
        col_year = _find_col("year")
    except Exception:
        # Some variants keep year inside title or another column; fail if absent
        raise ValueError("Year column not found.")

    # Clean data
    df = target[[col_rank, col_peak, col_title, col_year, col_gross]].copy()
    df.columns = ["Rank", "Peak", "Title", "Year", "Worldwide gross"]

    # Strip footnotes from strings
    for c in ["Title", "Year", "Worldwide gross"]:
        df[c] = df[c].astype(str).str.replace(r"\[.*?\]", "", regex=True).str.strip()

    # Parse year as 4-digit int
    df["Year"] = df["Year"].apply(_parse_year)

    # Parse gross to numeric (USD)
    df["Worldwide gross"] = df["Worldwide gross"].apply(_parse_money_to_number)

    # Numeric rank/peak
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")

    # Drop missing rows needed for our calculations
    df = df.dropna(subset=["Rank", "Peak", "Title", "Year", "Worldwide gross"])

    # Ensure correct dtypes
    df["Rank"] = df["Rank"].astype(int)
    df["Peak"] = df["Peak"].astype(int)
    df["Year"] = df["Year"].astype(int)
    df["Worldwide gross"] = pd.to_numeric(df["Worldwide gross"], errors="coerce")

    return df


def _parse_year(s: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return int(m.group(0)) if m else None


def _parse_money_to_number(s: str) -> Optional[float]:
    """
    Convert strings like '$2,923,706,026' into 2923706026.0.
    If the page uses suffixes, strip them; keep only digits and dots/commas appropriately.
    """
    # Remove anything except digits and decimal point
    cleaned = re.sub(r"[^0-9.]", "", s)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def _compute_answers_and_plot(df: pd.DataFrame) -> List[Any]:
    # 1) Count of $2B movies before 2000
    count_2bn_before_2000 = int((df["Worldwide gross"] >= 2_000_000_000) & (df["Year"] < 2000)).sum()

    # 2) Earliest film with >= $1.5B
    elig = df[df["Worldwide gross"] >= 1_500_000_000].copy()
    if elig.empty:
        earliest_title = ""
    else:
        elig = elig.sort_values(by=["Year", "Worldwide gross"], ascending=[True, False])
        earliest_title = str(elig.iloc[0]["Title"])

    # 3) Correlation between Rank and Peak (Pearson)
    corr = float(pd.to_numeric(df["Rank"], errors="coerce").corr(pd.to_numeric(df["Peak"], errors="coerce")))
    corr = float(np.round(corr, 6))

    # 4) Scatter plot (Rank vs Peak) + dotted red regression line
    data_uri = _scatter_rank_peak_as_data_uri(df)

    return [count_2bn_before_2000, earliest_title, corr, data_uri]


def _scatter_rank_peak_as_data_uri(df: pd.DataFrame) -> str:
    x = df["Rank"].astype(float).values
    y = df["Peak"].astype(float).values

    # Fit a simple linear regression line with numpy (no SciPy dependency)
    # y = m*x + b
    m, b = np.polyfit(x, y, 1)

    # Keep the figure small so the base64 stays under ~100 KB
    plt.figure(figsize=(4.0, 3.2), dpi=90)
    plt.scatter(x, y, alpha=0.7)
    xline = np.linspace(x.min(), x.max(), 100)
    yline = m * xline + b
    plt.plot(xline, yline, linestyle="--", color="red")
    plt.title("Rank vs Peak")
    plt.xlabel("Rank")
    plt.ylabel("Peak")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------- Local dev entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))




