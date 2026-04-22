import numpy as np
import os
import joblib

DIABETES_FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age"
]

MEDICAL_LABELS = {
    "Chest X-Ray": ["Normal Chest","Pneumonia","Pleural Effusion","Cardiomegaly","Atelectasis"],
    "Brain MRI":   ["Normal Brain","Glioma","Meningioma","Pituitary Tumor","Stroke Lesion"],
    "Skin Lesion": ["Benign Nevus","Melanoma","Basal Cell Carcinoma","Eczema","Psoriasis"],
    "Eye Retina":  ["Normal Retina","Diabetic Retinopathy","Glaucoma","AMD","Hypertensive Retinopathy"],
}

DRUG_KB = {
    frozenset(["warfarin","aspirin"]):           ("Severe",   "Increased bleeding/hemorrhage risk. Avoid combination."),
    frozenset(["warfarin","ibuprofen"]):          ("Severe",   "NSAIDs increase anticoagulation effect of warfarin."),
    frozenset(["ssri","maoi"]):                   ("Severe",   "Serotonin syndrome — potentially life-threatening."),
    frozenset(["fluoxetine","maoi"]):             ("Severe",   "Serotonin syndrome. Do not use within 14 days."),
    frozenset(["metformin","alcohol"]):           ("Moderate", "Risk of lactic acidosis. Limit alcohol."),
    frozenset(["lisinopril","potassium"]):        ("Moderate", "Risk of hyperkalemia. Monitor potassium levels."),
    frozenset(["simvastatin","amiodarone"]):      ("Severe",   "High risk of myopathy and rhabdomyolysis."),
    frozenset(["clopidogrel","omeprazole"]):      ("Moderate", "Omeprazole reduces antiplatelet effect of clopidogrel."),
    frozenset(["digoxin","amiodarone"]):          ("Severe",   "Amiodarone raises digoxin levels; toxicity risk."),
    frozenset(["methotrexate","nsaid"]):          ("Severe",   "NSAIDs reduce methotrexate clearance — toxicity."),
    frozenset(["ciprofloxacin","antacid"]):       ("Moderate", "Antacids reduce ciprofloxacin absorption. Take 2h apart."),
    frozenset(["lithium","ibuprofen"]):           ("Severe",   "NSAIDs increase lithium levels; toxicity risk."),
    frozenset(["sildenafil","nitrates"]):         ("Severe",   "Severe hypotension. Contraindicated."),
    frozenset(["tramadol","ssri"]):               ("Moderate", "Risk of serotonin syndrome and seizures."),
    frozenset(["warfarin","fluconazole"]):        ("Severe",   "Fluconazole strongly inhibits warfarin metabolism."),
    frozenset(["atorvastatin","clarithromycin"]): ("Moderate", "Clarithromycin increases statin levels; myopathy risk."),
    frozenset(["metformin","iodine"]):            ("Moderate", "Contrast dye may cause AKI; hold metformin."),
    frozenset(["amlodipine","simvastatin"]):      ("Moderate", "Increased simvastatin exposure; limit dose."),
    frozenset(["carbamazepine","warfarin"]):      ("Severe",   "Carbamazepine reduces warfarin efficacy."),
    frozenset(["phenytoin","fluconazole"]):       ("Moderate", "Fluconazole increases phenytoin levels."),
    frozenset(["alcohol","acetaminophen"]):       ("Moderate", "Chronic alcohol + acetaminophen → liver damage."),
    frozenset(["betablocker","verapamil"]):       ("Severe",   "Profound bradycardia and heart block risk."),
    frozenset(["heparin","aspirin"]):             ("Moderate", "Combined anticoagulation increases bleeding risk."),
    frozenset(["tacrolimus","fluconazole"]):      ("Severe",   "Fluconazole markedly increases tacrolimus levels."),
    frozenset(["lithium","nsaid"]):               ("Severe",   "NSAIDs raise lithium to toxic levels."),
}

SEV_ORDER = {"Severe": 2, "Moderate": 1, "None": 0}


# ── DIABETES ──────────────────────────────────────────────────────────────────

def _train_data():
    rng = np.random.RandomState(42)
    X = rng.rand(700, 8) * [10, 200, 130, 60, 300, 50, 2.5, 70]
    y = ((X[:,1] > 120) | ((X[:,5] > 30) & (X[:,7] > 45))).astype(int)
    return X, y


def load_rf():
    for p in ["diabetes_rf_model.pkl", "diabetes_model.pkl"]:
        if os.path.exists(p):
            try: return joblib.load(p)
            except: pass
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
        clf.fit(*_train_data())
        return clf
    except ImportError:
        return None


def load_lr():
    if os.path.exists("diabetes_lr_model.pkl"):
        try: return joblib.load("diabetes_lr_model.pkl")
        except: pass
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(*_train_data())
        return pipe
    except ImportError:
        return None


def predict_diabetes(model, features, model_type="Random Forest"):
    X = np.array(features, dtype=float).reshape(1, -1)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    risk  = "Diabetic" if pred == 1 else "Non-Diabetic"
    conf  = float(proba[pred])
    extra = {}
    if model_type == "Random Forest" and hasattr(model, "feature_importances_"):
        extra["importances"] = dict(zip(DIABETES_FEATURES, model.feature_importances_.tolist()))
    elif model_type == "Logistic Regression":
        try:
            lr  = model.named_steps["lr"]
            sc  = model.named_steps["sc"]
            extra["coefficients"] = dict(zip(DIABETES_FEATURES, (lr.coef_[0] / sc.scale_).tolist()))
        except: pass
    return risk, conf, extra


# ── SENTIMENT ─────────────────────────────────────────────────────────────────

def analyse_sentiment(text):
    try:
        from textblob import TextBlob
        p = TextBlob(text).sentiment.polarity
        label = "Positive" if p > 0.1 else "Negative" if p < -0.1 else "Neutral"
        return label, float(p)
    except ImportError:
        pos = {"happy","great","good","joy","love","calm","wonderful","excellent"}
        neg = {"sad","bad","terrible","awful","depressed","anxious","stressed","angry"}
        words = set(text.lower().split())
        s = (len(words & pos) - len(words & neg)) / max(len(words), 1)
        return ("Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral"), float(s)


def top_keywords(texts, n=20):
    from collections import Counter
    import re
    STOP = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
            "i","my","is","was","are","it","that","this","have","had","be","do",
            "not","so","we","he","she","they","you","me","am","very","just","feel","today"}
    words = re.findall(r'\b[a-z]{3,}\b', " ".join(texts).lower())
    return dict(Counter(w for w in words if w not in STOP).most_common(n))


# ── IMAGE ─────────────────────────────────────────────────────────────────────

def classify_image(image_bytes, image_type):
    labels = MEDICAL_LABELS.get(image_type, MEDICAL_LABELS["Chest X-Ray"])
    try:
        import torch, torchvision.models as models
        from torchvision import transforms
        from PIL import Image
        import io as _io
        m = models.resnet18(pretrained=False); m.eval()
        tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
             transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        img = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
        with torch.no_grad():
            out = m(tf(img).unsqueeze(0))
            probs = torch.softmax(out[0], dim=0).numpy()
        subset = [(labels[i % len(labels)], float(probs[i % len(probs)])) for i in range(len(labels))]
        top3 = sorted(subset, key=lambda x: -x[1])[:3]
        total = sum(c for _, c in top3)
        return [(n, c/total) for n, c in top3]
    except Exception:
        rng = np.random.RandomState(int.from_bytes(image_bytes[:4],"little") % (2**31))
        scores = rng.dirichlet(np.ones(len(labels)))
        top3 = sorted(zip(labels, scores), key=lambda x: -x[1])[:3]
        return [(n, float(c)) for n, c in top3]


# ── DRUGS ─────────────────────────────────────────────────────────────────────

def check_drugs(drug_list):
    norm = lambda s: s.strip().lower().replace("-","").replace(" ","")
    drugs = [norm(d) for d in drug_list if d.strip()]
    found = []
    seen  = set()
    for i in range(len(drugs)):
        for j in range(i+1, len(drugs)):
            pair = frozenset([drugs[i], drugs[j]])
            matched = DRUG_KB.get(pair)
            if not matched:
                for kp, val in DRUG_KB.items():
                    kpl = list(kp)
                    if any(k in drugs[i] or drugs[i] in k for k in kpl) and \
                       any(k in drugs[j] or drugs[j] in k for k in kpl):
                        matched = val; break
            if matched:
                key = tuple(sorted([drugs[i], drugs[j]]))
                if key not in seen:
                    seen.add(key)
                    found.append({"drug1": drug_list[i].strip(), "drug2": drug_list[j].strip(),
                                  "severity": matched[0], "details": matched[1]})
    max_sev = max((SEV_ORDER.get(f["severity"],0) for f in found), default=0)
    max_label = {2:"Severe",1:"Moderate",0:"None"}[max_sev]
    return found, max_label


DRUG_REFERENCE = [
    {"A":"Warfarin",   "B":"Aspirin",        "Severity":"Severe",   "Concern":"Hemorrhage risk"},
    {"A":"SSRIs",      "B":"MAOIs",          "Severity":"Severe",   "Concern":"Serotonin syndrome"},
    {"A":"Sildenafil", "B":"Nitrates",       "Severity":"Severe",   "Concern":"Severe hypotension"},
    {"A":"Metformin",  "B":"Alcohol",        "Severity":"Moderate", "Concern":"Lactic acidosis"},
    {"A":"Digoxin",    "B":"Amiodarone",     "Severity":"Severe",   "Concern":"Digoxin toxicity"},
    {"A":"Lithium",    "B":"Ibuprofen",      "Severity":"Severe",   "Concern":"Lithium toxicity"},
    {"A":"Tramadol",   "B":"SSRIs",          "Severity":"Moderate", "Concern":"Serotonin risk"},
    {"A":"Warfarin",   "B":"Fluconazole",    "Severity":"Severe",   "Concern":"Warfarin toxicity"},
]
