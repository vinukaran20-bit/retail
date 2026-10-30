"""
app.py  -  Final phase Flask application
Pages : /login  /  (inspect)  /camera  /history  /analytics (admin)  /export.csv (admin)
API   : /api/camera_frame (used by the webcam page)
"""
import base64
import csv
import io
import os
import uuid
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import (Flask, Response, flash, jsonify, redirect, render_template,
                   request, session, url_for)

import db
from alerts import alerts_enabled, send_defect_alert
from config import UPLOAD_DIR, RESULT_DIR, SECRET_KEY, REVIEW_THRESHOLD
from inference import inspect_bgr, model_ready

app = Flask(__name__)
app.secret_key = SECRET_KEY
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
db.init_db()


# ---------------- auth helpers ----------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        if session.get("role") != "admin":
            flash("That page is for administrators only.")
            return redirect(url_for("index"))
        return fn(*a, **kw)
    return wrapper


@app.context_processor
def inject_globals():
    return {"user": session.get("user"), "role": session.get("role"),
            "alerts_on": alerts_enabled(), "threshold": int(REVIEW_THRESHOLD * 100)}


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = db.verify_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if u:
            session["user"], session["role"] = u["username"], u["role"]
            return redirect(request.args.get("next") or url_for("index"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- logging helper ----------------
def log_result(result, image_name, source):
    rec = {
        "result_id": uuid.uuid4().hex[:10],
        "image_name": image_name,
        "source": source,
        "verdict": result["verdict"],
        "defect_type": result["defect_type"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "gradcam": os.path.basename(result["gradcam_overlay_path"]) if result["gradcam_overlay_path"] else "",
        "inspected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operator": session.get("user", "-"),
    }
    db.add_inspection(rec)
    if rec["verdict"] == "DEFECTIVE":
        send_defect_alert(rec)
    return rec


# ---------------- upload inspection ----------------
@app.route("/")
@login_required
def index():
    return render_template("index.html", model_ok=model_ready(), history=db.get_inspections(limit=8))


@app.route("/inspect", methods=["POST"])
@login_required
def inspect():
    if not model_ready():
        flash("No trained model found. Run train.py first.")
        return redirect(url_for("index"))
    file = request.files.get("product_image")
    if not file or file.filename == "":
        flash("Please choose a product image.")
        return redirect(url_for("index"))

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        flash("That file is not a readable image.")
        return redirect(url_for("index"))

    result = inspect_bgr(img)
    log_result(result, name, "upload")
    gradcam_filename = os.path.basename(result["gradcam_overlay_path"]) if result["gradcam_overlay_path"] else ""
    return render_template("index.html", model_ok=True, result=result, uploaded_image=name,
                           gradcam_filename=gradcam_filename, history=db.get_inspections(limit=8))


# ---------------- live camera ----------------
@app.route("/camera")
@login_required
def camera():
    return render_template("camera.html", model_ok=model_ready())


@app.route("/api/camera_frame", methods=["POST"])
@login_required
def camera_frame():
    if not model_ready():
        return jsonify(error="No trained model found. Run train.py first."), 400
    data = request.get_json(silent=True) or {}
    try:
        raw = base64.b64decode(data["image"].split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert img is not None
    except Exception:
        return jsonify(error="Could not read the camera frame."), 400

    result = inspect_bgr(img, save_overlay=bool(data.get("log")))
    if data.get("log"):
        name = f"cam_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_DIR, name), img)
        log_result(result, name, "camera")
    overlay = os.path.basename(result["gradcam_overlay_path"]) if result["gradcam_overlay_path"] else ""
    return jsonify(
        verdict=result["verdict"], defect_type=result["defect_type"],
        confidence=result["confidence"], reason=result["reason"],
        overlay_url=url_for("static", filename="results/" + overlay) if overlay else "",
        logged=bool(data.get("log")),
    )


# ---------------- history + CSV ----------------
@app.route("/history")
@login_required
def history():
    verdict = request.args.get("verdict") or None
    source = request.args.get("source") or None
    rows = db.get_inspections(verdict=verdict, source=source)
    return render_template("history.html", rows=rows, verdict=verdict or "", source=source or "")


@app.route("/export.csv")
@admin_required
def export_csv():
    rows = db.get_inspections(verdict=request.args.get("verdict") or None,
                              source=request.args.get("source") or None)
    buf = io.StringIO()
    cols = ["inspected_at", "image_name", "source", "verdict", "defect_type", "confidence", "reason", "operator"]
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow([r[c] for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=inspection_log.csv"})


# ---------------- analytics ----------------
@app.route("/analytics")
@admin_required
def analytics():
    return render_template("analytics.html", s=db.stats())


if __name__ == "__main__":
    app.run(debug=True)
