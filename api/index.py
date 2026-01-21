from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
import io

app = FastAPI()

# List of origins that are allowed to make requests to this API
origins = [
    "https://caefiss-benchmarker.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # Allows specific origins
    allow_credentials=True,
    allow_methods=["*"],              # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],              # Allows all headers
)

@app.get("/")
async def health_check():
    return {"status": "API is running"}

@app.get("api/test")
async def test():
    return {"status": "Python API is working!"}

@app.post("/generate_chart")
async def generate_chart(data: dict):
    try:
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
    
def calculate_business_hours(start_time, end_time):
    """
    Calculates the number of business hours (8-5, Mon-Fri) between two datetimes.
    """
    if pd.isnull(start_time) or pd.isnull(end_time) or start_time > end_time:
        return 0

    # Define business hours
    day_start = 8  # 8 AM
    day_end = 17   # 5 PM
    hours_per_day = day_end - day_start

    # Generate a range of business days (Mon-Fri)
    # np.busday_count counts full days between two dates
    business_days = np.busday_count(start_time.date(), end_time.date())

    # Initial calculation: full business days * 9 hours
    total_hrs = business_days * hours_per_day

    # Adjust for the partial first day:
    # Subtract time passed before the ticket started if it started after 8am
    start_hour = start_time.hour + start_time.minute / 60
    start_adj = max(0, min(hours_per_day, start_hour - day_start))
    
    # Adjust for the partial last day:
    # Add time passed in the final day up to 5pm
    end_hour = end_time.hour + end_time.minute / 60
    end_adj = max(0, min(hours_per_day, end_hour - day_start))

    return total_hrs - start_adj + end_adj

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

    # --- Business Hours Calculation ---
    df["turnaround_hrs"] = df.apply(
        lambda row: calculate_business_hours(row["inProgressTriggerTime"], row["doneTriggerTime"]), 
        axis=1
    )

    # --- Drop empty values ---
    df = df.dropna(subset=["turnaround_hrs"])

    # --- Aggregation (by story points) ---
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

