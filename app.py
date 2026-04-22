from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os, json
from db import (
    init_db, add_patient, get_patients, get_patient, delete_patient, search_patients,
    add_diabetes_record, add_diabetes_pred, get_diabetes_history, get_all_diabetes_preds,
    add_mood_entry, get_mood_history, get_all_mood,
    add_image_record, get_image_history,
    add_drug_check, get_drug_history,
    get_counts, get_recent_activity,
)
from ml import (
    load_rf, load_lr, predict_diabetes, DIABETES_FEATURES,
    analyse_sentiment, top_keywords,
    classify_image, check_drugs, DRUG_REFERENCE,
    MEDICAL_LABELS,
)

app = Flask(__name__)
app.secret_key = "medical_ai_secret_2024"

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache models at startup
_rf_model = None
_lr_model = None

def get_rf():
    global _rf_model
    if _rf_model is None:
        _rf_model = load_rf()
    return _rf_model

def get_lr():
    global _lr_model
    if _lr_model is None:
        _lr_model = load_lr()
    return _lr_model

init_db()


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    counts   = get_counts()
    activity = get_recent_activity()
    dp       = get_all_diabetes_preds()
    mood_raw = get_all_mood()

    # Diabetes pie data
    diab_labels, diab_counts = [], []
    diab_map = {}
    for r in dp:
        diab_map[r["Risk_Level"]] = diab_map.get(r["Risk_Level"], 0) + 1
    for k, v in diab_map.items():
        diab_labels.append(k); diab_counts.append(v)

    # Mood trend data
    mood_dates  = [r["Date"] for r in reversed(mood_raw)]
    mood_scores = [r["Mood_Score"] for r in reversed(mood_raw)]

    return render_template("dashboard.html",
        counts=counts, activity=activity,
        diab_labels=json.dumps(diab_labels),
        diab_counts=json.dumps(diab_counts),
        mood_dates=json.dumps(mood_dates),
        mood_scores=json.dumps(mood_scores),
    )


# ── PATIENTS ──────────────────────────────────────────────────────────────────
@app.route("/patients")
def patients():
    q   = request.args.get("q","").strip()
    pts = search_patients(q) if q else get_patients()
    return render_template("patients.html", patients=pts, q=q)

@app.route("/patients/add", methods=["POST"])
def patients_add():
    name    = request.form.get("name","").strip()
    age     = request.form.get("age", 30)
    gender  = request.form.get("gender","Other")
    contact = request.form.get("contact","").strip()
    if not name:
        flash("Name is required.", "error")
    else:
        pid = add_patient(name, int(age), gender, contact)
        flash(f"Patient '{name}' registered — ID: {pid}", "success")
    return redirect(url_for("patients"))

@app.route("/patients/delete", methods=["POST"])
def patients_delete():
    pid = request.form.get("pid")
    if pid:
        delete_patient(int(pid))
        flash(f"Patient ID {pid} deleted.", "success")
    return redirect(url_for("patients"))


# ── DIABETES ──────────────────────────────────────────────────────────────────
@app.route("/diabetes")
def diabetes():
    pts = get_patients()
    return render_template("diabetes.html", patients=pts, features=DIABETES_FEATURES)

@app.route("/diabetes/predict", methods=["POST"])
def diabetes_predict():
    try:
        pid        = int(request.form.get("patient_id"))
        model_type = request.form.get("model_type","Random Forest")
        feats = [
            float(request.form.get("Pregnancies",0)),
            float(request.form.get("Glucose",120)),
            float(request.form.get("BloodPressure",70)),
            float(request.form.get("SkinThickness",20)),
            float(request.form.get("Insulin",80)),
            float(request.form.get("BMI",25)),
            float(request.form.get("DiabetesPedigreeFunction",0.5)),
            float(request.form.get("Age",35)),
        ]
        model = get_rf() if model_type == "Random Forest" else get_lr()
        if model is None:
            return jsonify({"error": "scikit-learn not installed"}), 500

        risk, conf, extra = predict_diabetes(model, feats, model_type)
        rid = add_diabetes_record(pid, feats[1], feats[5], feats[4], feats[2], feats[3], feats[0], feats[6], feats[7])
        add_diabetes_pred(rid, model_type, risk, conf)

        chart_labels, chart_values, chart_color = [], [], []
        if "importances" in extra:
            items = sorted(extra["importances"].items(), key=lambda x: x[1])
            chart_labels = [i[0] for i in items]
            chart_values = [round(i[1]*100,2) for i in items]
            chart_color  = ["#3b82f6"]*len(items)
            chart_title  = "Feature Importance (%)"
        elif "coefficients" in extra:
            items = sorted(extra["coefficients"].items(), key=lambda x: x[1])
            chart_labels = [i[0] for i in items]
            chart_values = [round(i[1],4) for i in items]
            chart_color  = ["#f87171" if v > 0 else "#4ade80" for v in chart_values]
            chart_title  = "Feature Coefficients"
        else:
            chart_title = ""

        return jsonify({
            "risk": risk, "confidence": round(conf*100,1),
            "record_id": rid, "model": model_type,
            "chart_labels": chart_labels,
            "chart_values": chart_values,
            "chart_color":  chart_color,
            "chart_title":  chart_title,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── MOOD ──────────────────────────────────────────────────────────────────────
@app.route("/mood")
def mood():
    pts = get_patients()
    return render_template("mood.html", patients=pts)

@app.route("/mood/add", methods=["POST"])
def mood_add():
    try:
        pid    = int(request.form.get("patient_id"))
        score  = int(request.form.get("mood_score", 5))
        text   = request.form.get("journal_text","").strip() or f"Mood score: {score}"
        label, polarity = analyse_sentiment(text)
        add_mood_entry(pid, score, text, label, polarity)

        hist  = get_mood_history(pid)
        dates  = [r["Date"] for r in reversed(hist)]
        scores = [r["Mood_Score"] for r in reversed(hist)]
        sent_map = {}
        for r in hist:
            sl = r.get("Sentiment_Label","")
            sent_map[sl] = sent_map.get(sl, 0) + 1

        texts = [r["Journal_Text"] for r in hist if r.get("Journal_Text")]
        kw    = top_keywords(texts)

        return jsonify({
            "label": label, "polarity": round(polarity, 3),
            "score": score,
            "dates": dates, "mood_scores": scores,
            "sent_labels": list(sent_map.keys()),
            "sent_counts": list(sent_map.values()),
            "kw_labels": list(kw.keys())[:15],
            "kw_values": list(kw.values())[:15],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── IMAGE ─────────────────────────────────────────────────────────────────────
@app.route("/image")
def image():
    pts = get_patients()
    return render_template("image.html", patients=pts, image_types=list(MEDICAL_LABELS.keys()))

@app.route("/image/classify", methods=["POST"])
def image_classify():
    try:
        pid        = int(request.form.get("patient_id"))
        image_type = request.form.get("image_type","Chest X-Ray")
        file       = request.files.get("image_file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        img_bytes = file.read()
        filename  = f"img_{pid}_{os.urandom(4).hex()}.jpg"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as f:
            f.write(img_bytes)

        results   = classify_image(img_bytes, image_type)
        top_class = results[0][0]
        top_conf  = results[0][1]
        add_image_record(pid, f"static/uploads/{filename}", image_type, top_class, top_conf)

        return jsonify({
            "results": [{"label": n, "confidence": round(c*100,1)} for n,c in results],
            "image_url": f"/static/uploads/{filename}",
            "image_type": image_type,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── DRUGS ─────────────────────────────────────────────────────────────────────
@app.route("/drugs")
def drugs():
    pts = get_patients()
    return render_template("drugs.html", patients=pts, reference=DRUG_REFERENCE)

@app.route("/drugs/check", methods=["POST"])
def drugs_check():
    try:
        pid        = int(request.form.get("patient_id"))
        drug_list  = [request.form.get(f"drug{i}","").strip() for i in range(1,6)]
        drug_list  = [d for d in drug_list if d]
        if len(drug_list) < 2:
            return jsonify({"error": "Enter at least 2 drugs"}), 400

        interactions, max_sev = check_drugs(drug_list)
        drugs_str   = ", ".join(drug_list)
        details_str = "; ".join(f"{x['drug1']}+{x['drug2']}: {x['details']}" for x in interactions)
        add_drug_check(pid, drugs_str, max_sev, details_str)

        return jsonify({
            "interactions": interactions,
            "max_severity": max_sev,
            "drugs": drugs_str,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.route("/history/<int:pid>")
def history(pid):
    patient = get_patient(pid)
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for("patients"))
    diab_hist  = get_diabetes_history(pid)
    mood_hist  = get_mood_history(pid)
    image_hist = get_image_history(pid)
    drug_hist  = get_drug_history(pid)
    return render_template("history.html",
        patient=patient,
        diab_hist=diab_hist,
        mood_hist=mood_hist,
        image_hist=image_hist,
        drug_hist=drug_hist,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
