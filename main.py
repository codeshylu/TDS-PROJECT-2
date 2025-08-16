from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import requests

app = FastAPI()

class AnalyzeRequest(BaseModel):
    query: str
    data_url: str

@app.get("/")
def root():
    return {"message": "API is running. Use POST /analyze_data with your request."}

@app.post("/analyze_data")
def analyze_data(request: AnalyzeRequest):
    try:
        # Fetch data
        response = requests.get(request.data_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {response.status_code}")

        # Parse tables using pandas
        tables = pd.read_html(response.text)
        if len(tables) == 0:
            raise HTTPException(status_code=500, detail="No tables found at the URL")
        
        df = tables[0]  # Assuming the first table is the relevant one

        # Convert relevant columns to numeric safely
        if "Worldwide gross" in df.columns:
            df["Worldwide gross"] = (
                df["Worldwide gross"]
                .replace("[\$,]", "", regex=True)
                .apply(pd.to_numeric, errors="coerce") / 1e9
            )
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)

        # Example filter: movies over $2B before 2000
        filtered_df = df[(df.get("Worldwide gross", 0) >= 2) & (df.get("Year", 0) < 2000)]

        # Prepare scatter plot safely
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

        # Prepare response safely
        result = [
            len(filtered_df),
            str(filtered_df.iloc[0]["Title"]) if not filtered_df.empty else None,
            str(filtered_df["Worldwide gross"].iloc[0]) if not filtered_df.empty else None,
            f"data:image/png;base64,{img_base64}"
        ]
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")





















