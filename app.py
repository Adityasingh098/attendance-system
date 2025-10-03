import os
import io
import threading
import sqlite3
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file
from model import train_model_background, extract_embedding_for_image, MODEL_PATH, load_model_if_exists, predict_with_model

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---- DB init ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()
init_db()

# ---- Train status ----
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)
def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)
write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---- Routes ----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    data = request.form
    name = data.get("name","").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    roll = data.get("roll","").strip()
    cls = data.get("class","").strip()
    sec = data.get("sec","").strip()
    reg_no = data.get("reg_no","").strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    os.makedirs(folder, exist_ok=True)
    for f in files:
        fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
        path = os.path.join(folder, fname)
        f.save(path)
        saved += 1
    return jsonify({"saved": saved})

@app.route("/train_model", methods=["GET"])
def train_model_route():
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p,m: write_train_status({"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    return jsonify({"status":"started"}), 202

@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error":"no face detected"}), 200
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        pred_label, conf = predict_with_model(clf, emb)
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        name = row[0] if row else "Unknown"
        ts = datetime.datetime.utcnow().isoformat()
        c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (int(pred_label), name, ts))
        conn.commit()
        conn.close()
        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf)}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
