from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import io

app = FastAPI()

# List the origins that are allowed to make requests to this API
origins = [
    "https://caefiss-benchmarker.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)

# --- ENABLE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Define a Pydantic model if you want strict validation
# Or just use "data: dict" in the function signature for flexibility

@app.get("/")
async def health_check():
    return {"status": "API is running"}

@app.get("api/test")
async def test():
    return {"status": "Python API is working!"}

@app.post("/generate_chart")
async def generate_chart(data: dict):
    try:
        # Pass the dictionary directly to your processing function
        image_bytes = process_tickets_from_json(data)
        
        # Encode to Base64 to send back in a JSON response
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        return {
            "status": "success",
            "image": encoded_image
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

def process_tickets_from_json(data: dict) -> bytes:
    extracted_records = []

    for item in data.get("items", []):
        for component in item.get("componentChanges", []):
            results = component.get("changeItems", {}).get("results", [])
            for result in results:
                if result.get("fieldName") == "Log" and "changeTo" in result:
                    try:
                        log_data = json.loads(result["changeTo"])
                        extracted_records.append(log_data)
                    except json.JSONDecodeError:
                        continue

    if not extracted_records:
        raise ValueError("No valid ticket log data found")

    df = pd.DataFrame(extracted_records)

    # --- Data cleaning ---
    df["inProgressTriggerTime"] = pd.to_datetime(df["inProgressTriggerTime"], errors="coerce")
    df["doneTriggerTime"] = pd.to_datetime(df["doneTriggerTime"], errors="coerce")
    df["points"] = pd.to_numeric(df["storyPoints"], errors="coerce").fillna(0)

    df["turnaround_hrs"] = (
        df["doneTriggerTime"] - df["inProgressTriggerTime"]
    ).dt.total_seconds() / 3600

    df = df.dropna(subset=["turnaround_hrs"])

    # --- Aggregation ---
    stats_df = (
        df.groupby("points")["turnaround_hrs"]
        .mean()
        .reset_index()
        .sort_values("points")
    )

    # --- Benchmarks ---
    benchmarks = {
        1: 2,
        2: 4,
        3: 16,
        5: 24,
        8: 40,
        13: 80,
        21: 120,
    }

    stats_df["expected_hrs"] = stats_df["points"].map(benchmarks)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        stats_df["points"],
        stats_df["turnaround_hrs"],
        marker="o",
        linewidth=3,
        label="Actual Avg Turnaround",
    )

    ax.plot(
        stats_df["points"],
        stats_df["expected_hrs"],
        linestyle="--",
        marker="s",
        alpha=0.8,
        label="Expected Benchmark",
    )

    ax.set_xlabel("Story Points")
    ax.set_ylabel("Time (Hours)")
    ax.set_title("Actual vs Expected Turnaround Time per Story Point")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_xticks(stats_df["points"])

    for _, row in stats_df.iterrows():
        ax.annotate(
            f"{row['turnaround_hrs']:.1f}h",
            (row["points"], row["turnaround_hrs"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # --- Export image to memory ---
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)

    buf.seek(0)
    return buf.read()

