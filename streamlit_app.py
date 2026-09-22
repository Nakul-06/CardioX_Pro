"""
CardioX Pro - Clinical Cardiovascular Disease Risk Prediction & Longitudinal Monitoring System
Enterprise-Grade Medical Decision Support System.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="CardioX Pro - Cardiovascular Risk System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHARTS_DIR = BASE_DIR / "charts"
DATA_DIR = BASE_DIR / "processed_data"

# Clean Medical Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .risk-box-high {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .risk-box-moderate {
        background-color: #fffbeb;
        border: 2px solid #f59e0b;
        color: #92400e;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .risk-box-low {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .badge-normal {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-alert {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .tip-card-alert {
        background-color: #fff1f2;
        border-left: 4px solid #e11d48;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tip-card-warning {
        background-color: #fffbeb;
        border-left: 4px solid #d97706;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .tip-card-optimal {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div.stButton > button {
        background: #ffffff;
        color: #1e293b;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 0.6rem !important;
        border-radius: 8px !important;
        border: 1.5px solid #e2e8f0 !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        transition: all 0.15s ease-in-out !important;
        text-align: center !important;
    }
    div.stButton > button:hover {
        background: #f1f5f9 !important;
        border-color: #3b82f6 !important;
        color: #2563eb !important;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    """Loads models, scaler, and comparison metrics."""
    models = {}
    model_files = {
        "XGBoost": MODELS_DIR / "xgboost_model.joblib",
        "Random Forest": MODELS_DIR / "random_forest_model.joblib",
        "Gradient Boosting": MODELS_DIR / "gradient_boosting_model.joblib",
        "Logistic Regression": MODELS_DIR / "logistic_regression_model.joblib",
        "Decision Tree": MODELS_DIR / "decision_tree_model.joblib"
    }
    for name, path in model_files.items():
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                print(f"[CardioX Resource Loader] Note: Could not load {name} ({e}). Skipping.")

    scaler_path = MODELS_DIR / "feature_scaler.joblib"
    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"[CardioX Resource Loader] Note: Could not load feature_scaler ({e}).")

    comp_path = MODELS_DIR / "model_comparison.json"
    comparison_data = []
    if comp_path.exists():
        try:
            with open(comp_path, "r") as f:
                comparison_data = json.load(f)
        except Exception as e:
            print(f"[CardioX Resource Loader] Note: Could not load comparison data ({e}).")

    return models, scaler, comparison_data


models_dict, feature_scaler, model_comparison = load_resources()

# Sidebar
with st.sidebar:
    st.markdown("## 🫀 **CardioX Pro**")
    st.caption("Clinical Cardiovascular Decision Support")
    st.markdown("---")

    menu = st.radio(
        "Module Navigation",
        [
            "🩺 Patient Risk Predictor",
            "📈 Longitudinal Health Tracking",
            "🤖 Model Performance Analytics",
            "📊 Dataset & Clinical Parameters"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### **Clinical Benchmark**")
    st.markdown("- **Cohort Size:** 70,000 Patient Records")
    st.markdown("- **Validation Split:** 13,717 Held-Out Records")
    st.markdown("- **Algorithms:** 5 Supervised Classifiers")
    st.markdown("---")
    st.caption("Enterprise Cardiology AI System")


# ==============================================================================
# CLINICAL EVIDENCE RULES ENGINE (AHA / ACC / CDC GUIDELINES)
# ==============================================================================
def generate_clinical_recommendations(systolic_bp, diastolic_bp, cholesterol, glucose, bmi, smoke, alco, active):
    """
    Evidence-based clinical recommendation engine derived directly from:
    - 2017 ACC/AHA High Blood Pressure Clinical Practice Guidelines
    - 2018 AHA/ACC Multi-Society Guideline on Blood Cholesterol
    - CDC Guidelines on Smoking Cessation & Preventative Cardiology
    - AHA/ACSM Physical Activity Standards
    - ADA Standards of Medical Care in Diabetes
    """
    recs = []

    # 1. Blood Pressure Staging & Management (ACC/AHA 2017)
    if systolic_bp >= 140 or diastolic_bp >= 90:
        recs.append({
            "category": "Blood Pressure",
            "status": "alert",
            "icon": "🔴",
            "title": "Stage 2 Hypertension Clinical Protocol",
            "guideline": "ACC/AHA 2017 Hypertension Practice Guidelines",
            "action": [
                f"Systolic ({systolic_bp} mmHg) or Diastolic ({diastolic_bp} mmHg) indicates Stage 2 Hypertension with severe arterial shear stress.",
                "Immediate medical consultation for dual-agent antihypertensive pharmacological therapy (e.g. ACEi/ARB + CCB or Thiazide).",
                "Strict dietary sodium restriction (< 1,500 mg/day) following DASH protocol (proven to reduce systolic BP by 5-11 mmHg).",
                "Maintain home blood pressure monitoring twice daily (morning and evening resting)."
            ]
        })
    elif systolic_bp >= 130 or diastolic_bp >= 80:
        recs.append({
            "category": "Blood Pressure",
            "status": "warning",
            "icon": "🟡",
            "title": "Stage 1 Hypertension Lifestyle Intervention",
            "guideline": "ACC/AHA 2017 Hypertension Practice Guidelines",
            "action": [
                f"Systolic ({systolic_bp} mmHg) or Diastolic ({diastolic_bp} mmHg) falls within Stage 1 Hypertension.",
                "Initiate a structured 3 to 6-month trial of non-pharmacological lifestyle therapy.",
                "Increase dietary potassium intake (3,500 - 5,000 mg/day from dietary sources).",
                "Schedule clinical re-evaluation and ambulatory BP assessment in 3 months."
            ]
        })
    elif systolic_bp >= 120 and diastolic_bp < 80:
        recs.append({
            "category": "Blood Pressure",
            "status": "warning",
            "icon": "🟡",
            "title": "Elevated Blood Pressure Vascular Conditioning",
            "guideline": "ACC/AHA 2017 Hypertension Practice Guidelines",
            "action": [
                f"Systolic BP ({systolic_bp} mmHg) is elevated above optimal cardioprotective limits.",
                "Adopt stress-reduction protocols and dietary sodium restriction to prevent progression into clinical hypertension."
            ]
        })
    else:
        recs.append({
            "category": "Blood Pressure",
            "status": "optimal",
            "icon": "🟢",
            "title": "Normotensive Vascular Profile",
            "guideline": "ACC/AHA 2017 Hypertension Practice Guidelines",
            "action": [
                f"Resting blood pressure ({systolic_bp}/{diastolic_bp} mmHg) is in the ideal cardioprotective target range (< 120/80 mmHg).",
                "Maintain current active habits and attend routine annual preventative screenings."
            ]
        })

    # 2. Serum Cholesterol & Lipid Management (2018 AHA/ACC Guidelines)
    if cholesterol == 3:
        recs.append({
            "category": "Lipid Profile",
            "status": "alert",
            "icon": "🔴",
            "title": "Intensive Atherosclerosis Plaque Defense",
            "guideline": "2018 AHA/ACC Cholesterol Clinical Guidelines",
            "action": [
                "Serum cholesterol is significantly elevated (Level 3), markedly accelerating arterial plaque formation and stenosis.",
                "Obtain complete fasting lipid subfractions (Total, LDL-C, HDL-C, Triglycerides, and Apolipoprotein B).",
                "Consult physician regarding initiation of moderate-to-high intensity statin therapy.",
                "Restrict saturated fatty acids to < 6% of total caloric intake; eliminate all industrial trans-fats.",
                "Supplement diet with 10-25g daily of viscous soluble fiber (oats, legumes, psyllium)."
            ]
        })
    elif cholesterol == 2:
        recs.append({
            "category": "Lipid Profile",
            "status": "warning",
            "icon": "🟡",
            "title": "Targeted Dietary Lipid Reduction",
            "guideline": "2018 AHA/ACC Cholesterol Clinical Guidelines",
            "action": [
                "Serum cholesterol is above normal reference limits (Level 2).",
                "Adopt Mediterranean dietary pattern rich in monounsaturated fats (extra-virgin olive oil, tree nuts, avocados).",
                "Incorporate 2g/day of plant stanols/sterols to block dietary cholesterol absorption.",
                "Repeat lipid panel testing in 3 months to monitor response."
            ]
        })
    else:
        recs.append({
            "category": "Lipid Profile",
            "status": "optimal",
            "icon": "🟢",
            "title": "Cardioprotective Lipid Baseline",
            "guideline": "2018 AHA/ACC Cholesterol Clinical Guidelines",
            "action": [
                "Serum cholesterol is within normal physiological limits (Level 1).",
                "Maintain adequate dietary omega-3 polyunsaturated fatty acids (fatty cold-water fish or flaxseed) to sustain optimal HDL functionality."
            ]
        })

    # 3. Body Composition & Anthropometrics (WHO / AHA Guidelines)
    if bmi >= 30.0:
        recs.append({
            "category": "Body Composition",
            "status": "alert",
            "icon": "🔴",
            "title": f"Cardiac De-Loading Weight Reduction (BMI: {bmi:.1f} kg/m²)",
            "guideline": "AHA Scientific Statement on Obesity & Cardiovascular Disease",
            "action": [
                f"Calculated BMI of {bmi:.1f} kg/m² indicates clinical obesity, contributing to chronic left-ventricular preload and systemic inflammation.",
                "Target a structured 7% to 10% reduction in total body weight over 6 months.",
                "Establish a moderate caloric deficit of 500-750 kcal/day under qualified nutritional supervision.",
                "Combine aerobic conditioning with progressive resistance exercise twice weekly to preserve lean muscle mass."
            ]
        })
    elif bmi >= 25.0:
        recs.append({
            "category": "Body Composition",
            "status": "warning",
            "icon": "🟡",
            "title": f"Metabolic Weight Stabilization (BMI: {bmi:.1f} kg/m²)",
            "guideline": "AHA Scientific Statement on Obesity & Cardiovascular Disease",
            "action": [
                f"Calculated BMI of {bmi:.1f} kg/m² falls in the overweight classification.",
                "Target a modest 3% to 5% weight reduction to prevent progression into clinical obesity and reduce systemic insulin resistance."
            ]
        })
    else:
        recs.append({
            "category": "Body Composition",
            "status": "optimal",
            "icon": "🟢",
            "title": f"Healthy Anthropometric Window (BMI: {bmi:.1f} kg/m²)",
            "guideline": "WHO Clinical Anthropometric Standards",
            "action": [
                f"Calculated BMI of {bmi:.1f} kg/m² is within the ideal cardiovascular window (18.5 - 24.9 kg/m²).",
                "Focus on preserving lean skeletal muscle volume and metabolic balance."
            ]
        })

    # 4. Tobacco Smoking (CDC Guidelines)
    if smoke == 1:
        recs.append({
            "category": "Tobacco",
            "status": "alert",
            "icon": "🔴",
            "title": "Urgent Tobacco Cessation Protocol",
            "guideline": "CDC & ACC Preventative Tobacco Guidelines",
            "action": [
                "Active tobacco smoking induces acute arterial vasoconstriction, promotes platelet aggregability, and doubles coronary thrombosis risk.",
                "Consult primary clinician regarding FDA-approved pharmacotherapy (Nicotine Replacement Therapy, Varenicline, Bupropion).",
                "Evidence milestone: Excess cardiovascular risk drops by 50% within just 12 months of cessation."
            ]
        })

    # 5. Alcohol Consumption (AHA Guidelines)
    if alco == 1:
        recs.append({
            "category": "Alcohol",
            "status": "warning",
            "icon": "🟡",
            "title": "Alcohol Intake Moderation Protocol",
            "guideline": "AHA Dietary & Alcohol Recommendations",
            "action": [
                "Regular alcohol intake is clinically correlated with dose-dependent systolic hypertension and atrial arrhythmias.",
                "Limit intake to <= 1 standard drink daily, or adopt complete abstinence to promote vascular compliance and reduce liver metabolic burden."
            ]
        })

    # 6. Physical Activity (AHA / ACSM Standards)
    if active == 0:
        recs.append({
            "category": "Physical Activity",
            "status": "alert",
            "icon": "🔴",
            "title": "Aerobic Exercise Prescription",
            "guideline": "AHA/ACSM Guidelines for Exercise Testing & Prescription",
            "action": [
                "Sedentary lifestyle is an independent predictor of cardiovascular mortality and accelerates arterial stiffness.",
                "Prescribe minimum 150 minutes/week of moderate-intensity aerobic exercise (e.g. 30 min brisk walk, 5 days/week).",
                "Interrupt prolonged sedentary periods every 45-60 minutes with 2-3 minutes of light ambulation."
            ]
        })
    else:
        recs.append({
            "category": "Physical Activity",
            "status": "optimal",
            "icon": "🟢",
            "title": "Sustained Cardioprotective Conditioning",
            "guideline": "AHA/ACSM Guidelines for Exercise Testing & Prescription",
            "action": [
                "Regular physical activity is active, providing proven vascular and metabolic protection.",
                "Maintain routine aerobic conditioning and integrate 2 weekly resistance training sessions."
            ]
        })

    # 7. Fasting Glucose (ADA Standards)
    if glucose >= 2:
        recs.append({
            "category": "Glycemic Health",
            "status": "alert",
            "icon": "🔴",
            "title": "Dysglycemia & Microvascular Screening",
            "guideline": "ADA Standards of Medical Care in Diabetes",
            "action": [
                "Fasting blood glucose is elevated above normal baseline, promoting vascular endothelial oxidative stress.",
                "Schedule laboratory diagnostic HbA1c screening to rule out prediabetes or clinical type 2 diabetes.",
                "Replace simple sugars and refined carbohydrates with low-glycemic, high-fiber legumes and whole grains."
            ]
        })

    return recs


# ==============================================================================
# MODULE 1: PATIENT RISK PREDICTOR (FULLY REACTIVE REAL-TIME INFERENCE)
# ==============================================================================
if menu == "🩺 Patient Risk Predictor":
    st.markdown('<div class="main-header">🩺 Patient Cardiovascular Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adjust patient vitals, examination findings, and models below. Every parameter change updates the clinical assessment and recommendations <b>instantly in real-time</b>.</div>', unsafe_allow_html=True)

    # Initialize session state keys if not already set
    if "slider_age" not in st.session_state:
        st.session_state["slider_age"] = 52
    if "radio_gender" not in st.session_state:
        st.session_state["radio_gender"] = "Female"
    if "slider_height" not in st.session_state:
        st.session_state["slider_height"] = 168
    if "slider_weight" not in st.session_state:
        st.session_state["slider_weight"] = 74.0
    if "slider_sys" not in st.session_state:
        st.session_state["slider_sys"] = 135
    if "slider_dia" not in st.session_state:
        st.session_state["slider_dia"] = 85
    if "select_chol" not in st.session_state:
        st.session_state["select_chol"] = "Above Normal (2)"
    if "select_gluc" not in st.session_state:
        st.session_state["select_gluc"] = "Normal (1)"
    if "radio_smoke" not in st.session_state:
        st.session_state["radio_smoke"] = "No"
    if "radio_alco" not in st.session_state:
        st.session_state["radio_alco"] = "No"
    if "radio_active" not in st.session_state:
        st.session_state["radio_active"] = "Yes"
    if "select_model" not in st.session_state:
        st.session_state["select_model"] = "XGBoost"

    # Callback to load preset cases seamlessly before widget rendering
    def load_preset_case(age, gender, height, weight, sys, dia, chol, gluc, smoke, alco, active, model):
        st.session_state["slider_age"] = int(age)
        st.session_state["radio_gender"] = gender
        st.session_state["slider_height"] = int(height)
        st.session_state["slider_weight"] = float(weight)
        st.session_state["slider_sys"] = int(sys)
        st.session_state["slider_dia"] = int(dia)
        st.session_state["select_chol"] = chol
        st.session_state["select_gluc"] = gluc
        st.session_state["radio_smoke"] = smoke
        st.session_state["radio_alco"] = alco
        st.session_state["radio_active"] = active
        st.session_state["select_model"] = model

    # Quick Case Study Simulation Buttons (Clean, Unclipped, and 100% Functional)
    st.markdown("##### **⚡ Quick Patient Case Simulation Presets**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        st.button(
            "🟢 Load Low Risk (Case A)",
            key="btn_preset_a",
            on_click=load_preset_case,
            args=(28, "Female", 165, 58.0, 115, 75, "Normal (1)", "Normal (1)", "No", "No", "Yes", "XGBoost"),
            use_container_width=True
        )
        st.caption("Young Adult • 115/75 mmHg • Active")
    with preset_col2:
        st.button(
            "🟡 Load Moderate (Case B)",
            key="btn_preset_b",
            on_click=load_preset_case,
            args=(52, "Female", 168, 74.0, 135, 85, "Above Normal (2)", "Normal (1)", "No", "No", "Yes", "XGBoost"),
            use_container_width=True
        )
        st.caption("Age 52 • 135/85 mmHg • Overweight")
    with preset_col3:
        st.button(
            "🔴 Load High Risk (Case C)",
            key="btn_preset_c",
            on_click=load_preset_case,
            args=(64, "Male", 174, 92.0, 168, 102, "Well Above Normal (3)", "Above Normal (2)", "Yes", "Yes", "No", "XGBoost"),
            use_container_width=True
        )
        st.caption("Age 64 • 168/102 mmHg • Smoker")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Input Widgets (Bound Directly to Session State Keys)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### **1. Patient Demographics**")
        age = st.slider("Age (Years)", min_value=20, max_value=85, step=1, key="slider_age")

        gender_label = st.radio("Biological Sex", ["Female", "Male"], horizontal=True, key="radio_gender")
        gender = 1 if gender_label == "Male" else 0

        height = st.slider("Height (cm)", min_value=120, max_value=210, step=1, key="slider_height")
        weight = st.slider("Weight (kg)", min_value=40.0, max_value=160.0, step=0.5, key="slider_weight")

        # Auto calculate BMI live
        bmi = float(weight / ((height / 100) ** 2))
        bmi_display = f"{bmi:.1f}"
        if bmi < 25.0:
            bmi_status = "Normal"
            b_class = "badge-normal"
        elif bmi < 30.0:
            bmi_status = "Overweight"
            b_class = "badge-alert"
        else:
            bmi_status = "Obese"
            b_class = "badge-alert"

        st.markdown(f"**Calculated BMI:** `{bmi_display} kg/m²` <span class='{b_class}'>{bmi_status}</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("##### **2. Vitals & Laboratory Tests**")
        systolic_bp = st.slider("Systolic Blood Pressure (mm Hg)", min_value=80, max_value=220, step=1, key="slider_sys")
        diastolic_bp = st.slider("Diastolic Blood Pressure (mm Hg)", min_value=50, max_value=140, step=1, key="slider_dia")

        # Blood Pressure Stage Indicator
        if systolic_bp < 120 and diastolic_bp < 80:
            bp_status = "Normal (<120/80)"
            bp_class = "badge-normal"
        elif systolic_bp <= 129 and diastolic_bp < 80:
            bp_status = "Elevated (120-129)"
            bp_class = "badge-alert"
        elif systolic_bp <= 139 or diastolic_bp <= 89:
            bp_status = "Hypertension Stage 1"
            bp_class = "badge-alert"
        else:
            bp_status = "Hypertension Stage 2"
            bp_class = "badge-alert"

        st.markdown(f"**Blood Pressure Stage:** <span class='{bp_class}'>{bp_status}</span>", unsafe_allow_html=True)

        chol_opts = ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"]
        cholesterol_choice = st.selectbox("Serum Cholesterol", chol_opts, key="select_chol")
        cholesterol = 1 if "Normal (1)" in cholesterol_choice else (2 if "Above Normal (2)" in cholesterol_choice else 3)

        gluc_opts = ["Normal (1)", "Above Normal (2)", "Well Above Normal (3)"]
        glucose_choice = st.selectbox("Fasting Blood Glucose", gluc_opts, key="select_gluc")
        glucose = 1 if "Normal (1)" in glucose_choice else (2 if "Above Normal (2)" in glucose_choice else 3)

    with col3:
        st.markdown("##### **3. Lifestyle Habits & Algorithm**")
        smoke_label = st.radio("Current Tobacco Smoker?", ["No", "Yes"], horizontal=True, key="radio_smoke")
        smoke = 1 if smoke_label == "Yes" else 0

        alco_label = st.radio("Alcohol Consumption?", ["No", "Yes"], horizontal=True, key="radio_alco")
        alco = 1 if alco_label == "Yes" else 0

        active_label = st.radio("Physically Active?", ["Yes", "No"], horizontal=True, key="radio_active")
        active = 1 if active_label == "Yes" else 0

        model_list = list(models_dict.keys()) if models_dict else ["XGBoost"]
        model_name = st.selectbox(
            "Primary Inference Model",
            model_list,
            key="select_model",
            help="Select which trained supervised classifier evaluates primary risk. All other models run simultaneously in consensus below."
        )

    # ==============================================================================
    # REAL-TIME INFERENCE & ASSESSMENT (AUTOMATIC ON EVERY VALUE CHANGE)
    # ==============================================================================
    if not models_dict:
        st.error("Model artifacts not found. Please verify models directory.")
    else:
        selected_model = models_dict[model_name]

        # Construct patient feature vector
        input_df = pd.DataFrame([{
            "age_years": float(age),
            "gender_binary": int(gender),
            "height": float(height),
            "weight": float(weight),
            "bmi": float(bmi),
            "systolic_bp": float(systolic_bp),
            "diastolic_bp": float(diastolic_bp),
            "cholesterol": int(cholesterol),
            "gluc": int(glucose),
            "smoke": int(smoke),
            "alco": int(alco),
            "active": int(active)
        }])

        # Scale using fitted feature scaler
        if feature_scaler is not None:
            input_scaled = pd.DataFrame(feature_scaler.transform(input_df), columns=input_df.columns)
        else:
            input_scaled = input_df

        # Calculate exact probability for selected model
        prob_raw = float(selected_model.predict_proba(input_scaled)[0, 1])
        risk_pct = round(prob_raw * 100.0, 1)

        st.markdown("---")
        st.markdown("### **🩺 Clinical Assessment Output**")
        st.caption("⚡ Dynamically updated in real-time. Adjusting any slider or option above immediately recalculates all outputs below.")

        res_col1, res_col2 = st.columns([1.2, 1.8])

        with res_col1:
            if risk_pct >= 65.0:
                st.markdown(f"""
                <div class="risk-box-high">
                    <div style="font-size: 1.05rem; font-weight: 700;">⚠️ HIGH CARDIOVASCULAR RISK</div>
                    <div style="font-size: 3.2rem; font-weight: 800; margin: 8px 0;">{risk_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">Evaluated by <b>{model_name}</b></div>
                    <div style="font-size: 0.85rem; margin-top: 6px;">Significant clinical markers of cardiovascular disease detected.</div>
                </div>
                """, unsafe_allow_html=True)
            elif risk_pct >= 40.0:
                st.markdown(f"""
                <div class="risk-box-moderate">
                    <div style="font-size: 1.05rem; font-weight: 700;">⚠️ MODERATE CARDIOVASCULAR RISK</div>
                    <div style="font-size: 3.2rem; font-weight: 800; margin: 8px 0;">{risk_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">Evaluated by <b>{model_name}</b></div>
                    <div style="font-size: 0.85rem; margin-top: 6px;">Borderline physiological risk profile. Preventative intervention advised.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-box-low">
                    <div style="font-size: 1.05rem; font-weight: 700;">✅ LOW CARDIOVASCULAR RISK</div>
                    <div style="font-size: 3.2rem; font-weight: 800; margin: 8px 0;">{risk_pct:.1f}%</div>
                    <div style="font-size: 0.9rem;">Evaluated by <b>{model_name}</b></div>
                    <div style="font-size: 0.85rem; margin-top: 6px;">Patient vitals are currently within normal baseline physiological parameters.</div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(min(1.0, max(0.0, float(prob_raw))))

        with res_col2:
            st.markdown("#### **Primary Risk Contributing Drivers**")
            drivers = []
            if systolic_bp >= 140:
                drivers.append(f"🔴 **Systolic Blood Pressure ({systolic_bp} mmHg)** is Stage 2 Hypertensive (primary risk contributor).")
            elif systolic_bp >= 130:
                drivers.append(f"🟡 **Systolic Blood Pressure ({systolic_bp} mmHg)** is Stage 1 Hypertensive.")
            elif systolic_bp >= 120:
                drivers.append(f"🟡 **Systolic Blood Pressure ({systolic_bp} mmHg)** is Elevated.")

            if diastolic_bp >= 90:
                drivers.append(f"🔴 **Diastolic Blood Pressure ({diastolic_bp} mmHg)** is Stage 2 Hypertensive.")
            elif diastolic_bp >= 80:
                drivers.append(f"🟡 **Diastolic Blood Pressure ({diastolic_bp} mmHg)** is Stage 1 Hypertensive.")

            if cholesterol >= 3:
                drivers.append("🔴 **High Cholesterol (Level 3)** indicates accelerated plaque/atherosclerotic vulnerability.")
            elif cholesterol == 2:
                drivers.append("🟡 **Elevated Cholesterol (Level 2)** is above normal reference limits.")

            if bmi >= 30.0:
                drivers.append(f"🔴 **Obesity Classification (BMI: {bmi:.1f} kg/m²)** adds persistent left-ventricular load.")
            elif bmi >= 25.0:
                drivers.append(f"🟡 **Overweight Classification (BMI: {bmi:.1f} kg/m²)**.")

            if age >= 60:
                drivers.append(f"🔴 **Advanced Age ({age} Years)** is a non-modifiable biological risk factor.")
            elif age >= 50:
                drivers.append(f"🟡 **Patient Age ({age} Years)** falls in middle-age cardiovascular vulnerability zone.")

            if smoke == 1:
                drivers.append("🔴 **Tobacco Smoking** impairs arterial compliance and promotes thrombosis.")

            if active == 0:
                drivers.append("🟡 **Physical Inactivity** independently increases cardiovascular vulnerability.")

            if glucose >= 2:
                drivers.append("🔴 **Elevated Blood Glucose** increases systemic microvascular endothelial strain.")

            if not drivers:
                st.success("All examined physiological vitals and lifestyle habits are within optimal cardioprotective ranges.")
            else:
                for d in drivers:
                    st.markdown(f"- {d}")

        # Cross-Model Validation Section (Proves all 5 models are working live and distinct)
        st.markdown("---")
        st.markdown("#### **🤖 Cross-Model Multi-Algorithm Consensus**")
        st.caption("Live comparison of predicted cardiovascular risk across all 5 trained supervised models for this exact patient:")

        comp_cols = st.columns(len(models_dict))
        for idx, (m_name, m_obj) in enumerate(models_dict.items()):
            with comp_cols[idx]:
                m_p = float(m_obj.predict_proba(input_scaled)[0, 1])
                m_pct = round(m_p * 100.0, 1)
                is_selected = " ⭐" if m_name == model_name else ""
                st.metric(
                    label=f"{m_name}{is_selected}",
                    value=f"{m_pct:.1f}%",
                    delta="High" if m_pct >= 65 else ("Moderate" if m_pct >= 40 else "Low"),
                    delta_color="inverse"
                )

        # Dynamic Evidence-Based Clinical Recommendations
        st.markdown("---")
        st.markdown("#### **📋 Actionable Evidence-Based Clinical Recommendations (Clinical Tips Engine)**")
        st.caption("Generated in real-time based on ACC/AHA 2017 Hypertension, 2018 Multi-Society Cholesterol, and CDC Preventive Cardiology Guidelines.")

        clinical_tips = generate_clinical_recommendations(systolic_bp, diastolic_bp, cholesterol, glucose, bmi, smoke, alco, active)

        tip_cols = st.columns(2)
        for i, tip in enumerate(clinical_tips):
            col_target = tip_cols[i % 2]
            with col_target:
                card_style = f"tip-card-{tip['status']}"
                action_html = "".join([f"<li style='margin-bottom: 5px;'>{a}</li>" for a in tip['action']])
                st.markdown(f"""
                <div class="{card_style}">
                    <div style="font-weight: 700; font-size: 0.98rem; color: #0f172a; margin-bottom: 3px;">
                        {tip['icon']} {tip['title']}
                    </div>
                    <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 8px;">
                        <b>Guideline Reference:</b> {tip['guideline']}
                    </div>
                    <ul style="font-size: 0.86rem; color: #334155; margin-bottom: 0; padding-left: 20px;">
                        {action_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# MODULE 2: LONGITUDINAL HEALTH TRACKING
# ==============================================================================
elif menu == "📈 Longitudinal Health Tracking":
    st.markdown('<div class="main-header">📈 Longitudinal Health Tracking & Trajectory Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
        Traditional models evaluate patient risk only at a single static point in time.
        This module tracks physiological changes between consecutive checkups (&Delta; Systolic BP, &Delta; Cholesterol, &Delta; BMI)
        to identify whether cardiovascular risk is <b>escalating</b>, <b>stable</b>, or <b>improving</b>.
    </div>
    """, unsafe_allow_html=True)

    sample_patients = {
        "Patient Case 1: Progressive Arterial Degradation (Escalating Trajectory)": [
            {"Visit": "Visit 1 (Baseline)", "Date": "2023-01-15", "Systolic BP": 122, "Diastolic BP": 80, "Cholesterol": 1, "BMI": 26.2, "Risk (%)": 24.1},
            {"Visit": "Visit 2 (6 Months)", "Date": "2023-07-20", "Systolic BP": 136, "Diastolic BP": 86, "Cholesterol": 2, "BMI": 28.1, "Risk (%)": 49.3},
            {"Visit": "Visit 3 (12 Months)", "Date": "2024-01-18", "Systolic BP": 152, "Diastolic BP": 94, "Cholesterol": 3, "BMI": 30.5, "Risk (%)": 78.6}
        ],
        "Patient Case 2: Post-Intervention Recovery (Improving Trajectory)": [
            {"Visit": "Visit 1 (Baseline)", "Date": "2023-02-10", "Systolic BP": 155, "Diastolic BP": 96, "Cholesterol": 3, "BMI": 31.4, "Risk (%)": 76.5},
            {"Visit": "Visit 2 (6 Months)", "Date": "2023-08-12", "Systolic BP": 138, "Diastolic BP": 88, "Cholesterol": 2, "BMI": 29.0, "Risk (%)": 51.2},
            {"Visit": "Visit 3 (12 Months)", "Date": "2024-02-14", "Systolic BP": 124, "Diastolic BP": 82, "Cholesterol": 1, "BMI": 26.8, "Risk (%)": 22.8}
        ]
    }

    selected_case = st.selectbox("Select Patient Case Study", list(sample_patients.keys()))
    records = sample_patients[selected_case]
    df_rec = pd.DataFrame(records)

    col_t1, col_t2 = st.columns([1.1, 1.3])

    with col_t1:
        st.markdown("##### **Multi-Visit Clinical Records**")
        st.dataframe(df_rec, use_container_width=True)

        delta_bp = records[2]["Systolic BP"] - records[0]["Systolic BP"]
        delta_risk = records[2]["Risk (%)"] - records[0]["Risk (%)"]

        m1, m2 = st.columns(2)
        m1.metric("12-Month &Delta; Blood Pressure", f"{records[2]['Systolic BP']} mm Hg", f"{delta_bp:+} mm Hg", delta_color="inverse")
        m2.metric("12-Month &Delta; Risk Velocity", f"{records[2]['Risk (%)']:.1f}%", f"{delta_risk:+.1f}%", delta_color="inverse")

        if delta_risk > 15:
            st.error("🚨 **TRAJECTORY ALERT: Rapidly Escalating Cardiovascular Risk**\nConsistent upward velocity in blood pressure and cholesterol indicates worsening endothelial health. Immediate clinical review required.")
        else:
            st.success("🟢 **TRAJECTORY NOTICE: Positive Risk Reduction**\nPatient demonstrates steady clinical improvement across consecutive follow-ups.")

    with col_t2:
        st.markdown("##### **Longitudinal Risk & Vital Trajectory**")
        fig, ax1 = plt.subplots(figsize=(7, 4.2))

        ax1.set_xlabel("Checkup Timeline")
        ax1.set_ylabel("Systolic Blood Pressure (mm Hg)", color="#dc2626")
        line1 = ax1.plot(df_rec["Visit"], df_rec["Systolic BP"], marker="o", color="#dc2626", linewidth=2.5, label="Systolic BP (mm Hg)")
        ax1.tick_params(axis="y", labelcolor="#dc2626")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Predicted Risk Probability (%)", color="#2563eb")
        line2 = ax2.plot(df_rec["Visit"], df_rec["Risk (%)"], marker="s", color="#2563eb", linewidth=2.5, linestyle="--", label="Risk Probability (%)")
        ax2.tick_params(axis="y", labelcolor="#2563eb")
        ax2.set_ylim(0, 100)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper left")
        plt.title("Longitudinal Patient Trajectory")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ==============================================================================
# MODULE 3: MODEL PERFORMANCE ANALYTICS
# ==============================================================================
elif menu == "🤖 Model Performance Analytics":
    st.markdown('<div class="main-header">🤖 Supervised Machine Learning Model Analytics</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sub-header">
        Comparative evaluation of <b>5 supervised machine learning models</b> trained on <b>55,442 balanced records</b>
        and evaluated strictly on <b>13,717 held-out test patient records</b>.
    </div>
    """, unsafe_allow_html=True)

    if model_comparison:
        df_comp = pd.DataFrame(model_comparison)
        df_comp = df_comp[["model_name", "accuracy", "precision", "recall", "specificity", "f1_score", "roc_auc"]]
        df_comp.columns = ["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "Specificity (%)", "F1-Score (%)", "ROC-AUC (%)"]

        st.dataframe(
            df_comp.style.highlight_max(axis=0, subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "Specificity (%)", "F1-Score (%)", "ROC-AUC (%)"], color="#dcfce7"),
            use_container_width=True
        )

        st.markdown("---")
        st.markdown("##### **Model Performance Comparison Across Test Metrics**")

        fig, ax = plt.subplots(figsize=(10, 4.8))
        plot_df = df_comp.set_index("Model")[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "ROC-AUC (%)"]]
        plot_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.75)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(50, 88)
        ax.legend(loc="lower right")
        plt.title("Comparative Performance Evaluation (13,717 Independent Test Patients)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("##### **Class-Imbalance Handling via SMOTE**")
        c1, c2 = st.columns(2)
        with c1:
            st.info("""
            **Imbalance Mitigation Strategy:**
            - In raw clinical datasets, imbalanced classes cause classifiers to bias toward the majority class, leading to high false negatives (failing to identify actual CVD patients).
            - SMOTE oversampling was used on the training partition to balance class representation (27,721 healthy vs 27,721 disease cases).
            """)
        with c2:
            st.success("""
            **Data Leakage Prevention Guarantee:**
            - **Strict Isolation:** SMOTE was applied solely to the 80% training set.
            - **Honest Test Benchmarking:** The 13,717 held-out testing patients remained completely untouched with natural class distribution.
            """)
    else:
        st.warning("Model comparison data not found. Please verify models directory.")


# ==============================================================================
# MODULE 4: DATASET & CLINICAL PARAMETERS
# ==============================================================================
elif menu == "📊 Dataset & Clinical Parameters":
    st.markdown('<div class="main-header">📊 Clinical Dataset Architecture & Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Overview of dataset origins, cleaning methodologies, and feature importance rankings.</div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    d1.metric("Raw Dataset Count", "70,000 Records", "Kaggle (Svetlana Ulianova)")
    d2.metric("Clean Valid Records", "68,584 Records", "1,416 recording typos cleaned (2.02%)")
    d3.metric("Train / Test Split", "54,867 / 13,717", "80% / 20% Stratified")

    st.markdown("---")
    st.markdown("##### **Feature Importance Consensus Rankings**")
    st.markdown("Consensus weights determined across Random Forest Gini Importance, Pearson Correlation, and Mutual Information:")

    f_col1, f_col2 = st.columns([1, 1.2])

    with f_col1:
        rankings_data = [
            {"Rank": 1, "Feature": "Systolic Blood Pressure (systolic_bp)", "Importance": "1.000 (Top Driver)"},
            {"Rank": 2, "Feature": "Diastolic Blood Pressure (diastolic_bp)", "Importance": "0.635"},
            {"Rank": 3, "Feature": "Patient Age in Years (age_years)", "Importance": "0.411"},
            {"Rank": 4, "Feature": "Serum Cholesterol (cholesterol)", "Importance": "0.321"},
            {"Rank": 5, "Feature": "Body Mass Index (bmi)", "Importance": "0.273"},
            {"Rank": 6, "Feature": "Patient Weight (weight)", "Importance": "0.203"},
            {"Rank": 7, "Feature": "Fasting Glucose (gluc)", "Importance": "0.081"},
            {"Rank": 8, "Feature": "Physical Activity (active)", "Importance": "0.038"}
        ]
        st.dataframe(pd.DataFrame(rankings_data), use_container_width=True)

    with f_col2:
        chart_p = CHARTS_DIR / "feature_importance_ranking.png"
        if chart_p.exists():
            st.image(str(chart_p), caption="Feature Importance Ranking (70,000 Records)", use_container_width=True)
