import sqlite3
import pandas as pd
import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from enhancements import enhance_weekly
from predictions import train_and_predict

# ---------------------------
# Configuration
# ---------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "inventory_secret"
UPLOAD_DIR = Path("uploads")
DB_PATH    = Path("inventory.db")
ALLOWED_EXT = {"csv"}

UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------
# Database Initialization
# ---------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        # metadata
        con.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                filename     TEXT NOT NULL,
                uploaded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # raw data
        con.execute("""
            CREATE TABLE IF NOT EXISTS weekly_sales_raw (
                Week             INTEGER,
                Product_Name     TEXT,
                Price_Bought     REAL,
                Quantity_Bought  INTEGER,
                Price_Sold       REAL,
                Quantity_Sold    INTEGER,
                Start_Date       TEXT,
                End_Date         TEXT
            )
        """)
        # enhanced data (final table)
        con.execute("""
            CREATE TABLE IF NOT EXISTS weekly_sales (
                Week               INTEGER,
                Product_Name       TEXT,
                Price_Bought       REAL,
                Quantity_Bought    INTEGER,
                Price_Sold         REAL,
                Quantity_Sold      INTEGER,
                Start_Date         TEXT,
                End_Date           TEXT,
                Discount_Rate      REAL,
                Promo_Flag         INTEGER,
                Stockout_Flag      INTEGER,
                Qty_Base           REAL,
                Week_Index         INTEGER,
                CAGR_Units_Sold    REAL,
                Lag_Qty            REAL,
                Lag_Price          REAL,
                Price_Elasticity   REAL
            )
        """)

# ---------------------------
# Helpers
# ---------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def record_upload(filename: str):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO uploads (filename) VALUES (?)", (filename,))

def insert_raw(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql("weekly_sales_raw", con, if_exists="append", index=False)

def rebuild_enhanced():
    """
    Read all raw data, enhance it, and overwrite the enhanced table.
    """
    with sqlite3.connect(DB_PATH) as con:
        raw = pd.read_sql("SELECT * FROM weekly_sales_raw", con)
    if raw.empty:
        return
    enhanced = enhance_weekly(raw)
    # overwrite weekly_sales
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM weekly_sales")
        enhanced.to_sql("weekly_sales", con, if_exists="append", index=False)

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("home"))

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            flash("Only CSV files are allowed.", "danger")
            return redirect(url_for("home"))

        path = UPLOAD_DIR / filename
        file.save(path)

        # Load and validate CSV
        df = pd.read_csv(path)
        required = [
            "Product_Name","Price_Bought","Quantity_Bought",
            "Price_Sold","Quantity_Sold","Start_Date","End_Date"
        ]
        if not all(col in df.columns for col in required):
            flash("CSV missing required columns!", "danger")
            return redirect(url_for("home"))

        # Infer Week
        start_str = df.loc[0, "Start_Date"]
        try:
            start_dt = datetime.datetime.strptime(start_str, "%d-%m-%Y")
        except ValueError:
            flash("Start_Date must be DD-MM-YYYY.", "danger")
            return redirect(url_for("home"))

        prev_end = (start_dt - datetime.timedelta(days=1)).strftime("%d-%m-%Y")
        with sqlite3.connect(DB_PATH) as con:
            row = con.execute(
                "SELECT Week FROM weekly_sales_raw WHERE End_Date = ? "
                "ORDER BY Week DESC LIMIT 1", (prev_end,)
            ).fetchone()
        inferred_week = (row[0] + 1) if row else 1
        df["Week"] = inferred_week

        # Record & insert raw
        try:
            record_upload(filename)
            insert_raw(df[["Week"] + required])
            # Rebuild enhanced table
            rebuild_enhanced()
            flash(f"Week {inferred_week} data stored & enhanced.", "success")
        except Exception as e:
            flash(f"Error: {e}", "danger")

        return redirect(url_for("home"))

    # GET: show recent uploads
    with sqlite3.connect(DB_PATH) as con:
        uploads = con.execute(
            "SELECT filename, uploaded_at FROM uploads ORDER BY id DESC LIMIT 5"
        ).fetchall()
        # fetch the latest week present in weekly_sales
        lw = con.execute("SELECT MAX(Week) FROM weekly_sales").fetchone()[0] or 0

    return render_template("index.html", uploads=uploads, latest_week=lw)

# route for predictions
@app.route("/predict/<int:week>")
def predict(week):
    try:
        df_pred = train_and_predict(week)
    except Exception as e:
        flash(f"Prediction error: {e}", "danger")
        return redirect(url_for("home"))

    # Prepare data for Chart.js
    chart = {
        "labels": df_pred["Product_Name"].tolist(),
        "bought": df_pred["Quantity_Bought"].tolist(),
        "pred": df_pred["Predicted_Qty"].tolist()
    }
    return render_template("predictions.html", week=week, next_week=week+1, chart_data=chart)

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
