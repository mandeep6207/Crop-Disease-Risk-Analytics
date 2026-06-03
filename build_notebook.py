import nbformat as nbf
import json

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"}
}

cells = []

def code(src, tags=None):
    c = nbf.v4.new_code_cell(src)
    return c

def md(src):
    return nbf.v4.new_markdown_cell(src)


cells.append(md("# 🌿 CropGuard AI - Crop Disease Risk Analytics System\n\n**Predict crop disease risk levels using environmental, soil, and crop features.**\n\n---"))

cells.append(code("""
# ============================================================
# CELL 1: Install & Import Libraries
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, json, joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score)
from xgboost import XGBClassifier

# Paths
BASE = Path('..')
DATA_DIR   = BASE / 'data'
MODELS_DIR = BASE / 'models'
VISUALS_DIR= BASE / 'visuals'
REPORTS_DIR= BASE / 'reports'

for d in [DATA_DIR, MODELS_DIR, VISUALS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("✅ Libraries loaded and directories ready.")
print(f"  pandas  : {pd.__version__}")
print(f"  numpy   : {np.__version__}")
"""))


cells.append(md("## 📂 Step 1: Dataset Generation / Loading"))

cells.append(code("""
# ============================================================
# CELL 2: Dataset Generation / Loading
# ============================================================
DATASET_PATH = DATA_DIR / 'crop_disease_dataset.csv'

def generate_dataset(n=5000, seed=42):
    np.random.seed(seed)
    crop_types = ['Wheat','Rice','Maize','Soybean','Cotton',
                  'Sugarcane','Potato','Tomato','Barley','Sunflower']

    # Profile-based generation for balanced classes
    # ~33% each class by design
    class_profiles = {
        'Low Risk':    {'temp_mean':24,'hum_mean':55,'rain_mean':150,'moist_mean':60,
                        'pest_p':0.1,'leaf_p':0.1,'growth_mean':0.82,'hist_p':0.15},
        'Medium Risk': {'temp_mean':30,'hum_mean':68,'rain_mean':110,'moist_mean':50,
                        'pest_p':0.40,'leaf_p':0.40,'growth_mean':0.65,'hist_p':0.45},
        'High Risk':   {'temp_mean':36,'hum_mean':82,'rain_mean':75,'moist_mean':40,
                        'pest_p':0.75,'leaf_p':0.75,'growth_mean':0.42,'hist_p':0.80},
    }

    records = []
    n_per_class = n // 3
    extra = n - n_per_class * 3

    for idx, (label, profile) in enumerate(class_profiles.items()):
        count = n_per_class + (1 if idx < extra else 0)
        for _ in range(count):
            crop = np.random.choice(crop_types)
            temp       = np.random.normal(profile['temp_mean'], 4)
            humidity   = np.random.normal(profile['hum_mean'], 10)
            rainfall   = np.random.normal(profile['rain_mean'], 40)
            soil_moist = np.random.normal(profile['moist_mean'], 10)
            soil_ph    = np.random.normal(6.5, 0.7)
            nitrogen   = np.random.normal(70 if label=='Low Risk' else 50 if label=='Medium Risk' else 30, 15)
            phosphorus = np.random.normal(50 if label=='Low Risk' else 38 if label=='Medium Risk' else 25, 12)
            potassium  = np.random.normal(60 if label=='Low Risk' else 45 if label=='Medium Risk' else 30, 14)
            sunlight   = np.random.normal(8 if label=='Low Risk' else 7 if label=='Medium Risk' else 5.5, 1.5)
            pest       = int(np.random.random() < profile['pest_p'])
            leaf_disc  = int(np.random.random() < profile['leaf_p'])
            growth_rate= np.random.normal(profile['growth_mean'], 0.10)
            prev_hist  = int(np.random.random() < profile['hist_p'])

            records.append({
                'crop_type': crop,
                'temperature': round(float(np.clip(temp, 5, 50)), 2),
                'humidity': round(float(np.clip(humidity, 20, 100)), 2),
                'rainfall': round(float(np.clip(rainfall, 0, 300)), 2),
                'soil_moisture': round(float(np.clip(soil_moist, 10, 90)), 2),
                'soil_ph': round(float(np.clip(soil_ph, 4.5, 9.0)), 2),
                'nitrogen': round(float(np.clip(nitrogen, 0, 120)), 2),
                'phosphorus': round(float(np.clip(phosphorus, 0, 100)), 2),
                'potassium': round(float(np.clip(potassium, 0, 120)), 2),
                'sunlight_hours': round(float(np.clip(sunlight, 2, 14)), 2),
                'pest_activity': pest,
                'leaf_discoloration': leaf_disc,
                'growth_rate': round(float(np.clip(growth_rate, 0.1, 1.0)), 3),
                'previous_disease_history': prev_hist,
                'disease_risk': label
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded existing dataset: {df.shape}")
else:
    df = generate_dataset(5000)
    df.to_csv(DATASET_PATH, index=False)
    print(f"✅ Generated and saved new dataset: {df.shape}")

print(df['disease_risk'].value_counts())
"""))


cells.append(md("## 🔍 Step 2: Dataset Exploration"))

cells.append(code("""
# ============================================================
# CELL 3: Dataset Exploration
# ============================================================
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape       : {df.shape}")
print(f"Rows        : {df.shape[0]}")
print(f"Columns     : {df.shape[1]}")
print()
print("Data Types:")
print(df.dtypes)
print()
print("First 5 Rows:")
df.head()
"""))

cells.append(code("""
# Statistical Summary
print("STATISTICAL SUMMARY")
print("=" * 60)
df.describe().round(3)
"""))

cells.append(md("## 🔬 Step 3: Missing Value Analysis"))

cells.append(code("""
# ============================================================
# CELL 4: Missing Value Analysis
# ============================================================
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_report = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print("MISSING VALUE ANALYSIS")
print("=" * 60)
print(missing_report[missing_report['Missing Count'] > 0] if missing.sum() > 0 else "✅ No missing values found!")
print(f"\\nTotal missing cells: {df.isnull().sum().sum()}")
print(f"Dataset completeness: {((1 - df.isnull().sum().sum() / df.size) * 100):.2f}%")
"""))

cells.append(md("## 🧹 Step 4: Data Cleaning"))

cells.append(code("""
# ============================================================
# CELL 5: Data Cleaning
# ============================================================
df_clean = df.copy()

# Remove duplicates
n_dups = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"Duplicates removed: {n_dups}")

# Clip outliers using IQR
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
outlier_info = {}
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.01)
    Q3 = df_clean[col].quantile(0.99)
    original = df_clean[col].copy()
    df_clean[col] = df_clean[col].clip(Q1, Q3)
    clipped = (original != df_clean[col]).sum()
    if clipped > 0:
        outlier_info[col] = clipped

print(f"Outliers clipped: {outlier_info}")
print(f"✅ Clean dataset shape: {df_clean.shape}")
print(f"Target distribution:")
print(df_clean['disease_risk'].value_counts(normalize=True).round(3))
"""))


cells.append(md("## ⚙️ Step 5: Feature Engineering"))

cells.append(code("""
# ============================================================
# CELL 6: Feature Engineering - crop_health_score
# ============================================================
df_feat = df_clean.copy()

# 1. Crop Health Score: normalized composite of growth, pH, moisture
df_feat['crop_health_score'] = (
    df_feat['growth_rate'].clip(0.1, 1.0) * 0.4 +
    (1 - abs(df_feat['soil_ph'] - 6.5) / 4).clip(0, 1) * 0.35 +
    (df_feat['soil_moisture'] / 100).clip(0, 1) * 0.25
).round(4)

print("✅ crop_health_score created")
print(df_feat['crop_health_score'].describe())
"""))

cells.append(code("""
# ============================================================
# CELL 7: Feature Engineering - nutrient_balance_score
# ============================================================
# Normalize NPK to 0-1 range and compute balance
df_feat['nutrient_balance_score'] = (
    (df_feat['nitrogen'] / 120).clip(0, 1) * 0.35 +
    (df_feat['phosphorus'] / 100).clip(0, 1) * 0.35 +
    (df_feat['potassium'] / 120).clip(0, 1) * 0.30
).round(4)

print("✅ nutrient_balance_score created")
print(df_feat['nutrient_balance_score'].describe())
"""))

cells.append(code("""
# ============================================================
# CELL 8: Feature Engineering - disease_pressure_index
# ============================================================
# High humidity + pest + leaf discoloration + previous history => high pressure
df_feat['disease_pressure_index'] = (
    (df_feat['humidity'] / 100).clip(0, 1) * 0.30 +
    df_feat['pest_activity'].astype(float) * 0.25 +
    df_feat['leaf_discoloration'].astype(float) * 0.25 +
    df_feat['previous_disease_history'].astype(float) * 0.20
).round(4)

print("✅ disease_pressure_index created")
print(df_feat['disease_pressure_index'].describe())
"""))

cells.append(code("""
# ============================================================
# CELL 9: Feature Engineering - environmental_stress_score
# ============================================================
# High temperature deviation + low rainfall + high/low soil moisture
df_feat['environmental_stress_score'] = (
    (np.abs(df_feat['temperature'] - 25) / 25).clip(0, 1) * 0.30 +
    (1 - (df_feat['rainfall'] / 300).clip(0, 1)) * 0.30 +
    (1 - (df_feat['sunlight_hours'] / 12).clip(0, 1)) * 0.20 +
    (np.abs(df_feat['soil_moisture'] - 60) / 60).clip(0, 1) * 0.20
).round(4)

print("✅ environmental_stress_score created")
print(df_feat['environmental_stress_score'].describe())
"""))

cells.append(code("""
# ============================================================
# CELL 10: Feature Engineering - pest_risk_score
# ============================================================
# Combine pest activity, humidity, temperature extremes
df_feat['pest_risk_score'] = (
    df_feat['pest_activity'].astype(float) * 0.40 +
    (df_feat['humidity'] / 100).clip(0, 1) * 0.30 +
    ((df_feat['temperature'] - 15) / 35).clip(0, 1) * 0.30
).round(4)

print("✅ pest_risk_score created")
print(df_feat['pest_risk_score'].describe())
print()
print(f"Total features after engineering: {df_feat.shape[1]}")
print("New features:", ['crop_health_score','nutrient_balance_score',
      'disease_pressure_index','environmental_stress_score','pest_risk_score'])
"""))


cells.append(md("## 📊 Step 6: Exploratory Data Analysis (EDA)"))

cells.append(code("""
# ============================================================
# CELL 11: EDA - Disease Risk Distribution
# ============================================================
plt.figure(figsize=(10, 6))
colors = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
risk_counts = df_feat['disease_risk'].value_counts()

bars = plt.bar(risk_counts.index,
               risk_counts.values,
               color=[colors[r] for r in risk_counts.index],
               edgecolor='white', linewidth=1.5, width=0.6)

for bar, val in zip(bars, risk_counts.values):
    pct = val / len(df_feat) * 100
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
             f'{val}\\n({pct:.1f}%)', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('🌿 CropGuard AI - Disease Risk Distribution', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Disease Risk Level', fontsize=13)
plt.ylabel('Number of Records', fontsize=13)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'disease_risk_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: disease_risk_distribution.png")
"""))

cells.append(code("""
# ============================================================
# CELL 12: EDA - Temperature Impact
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
bp = axes[0].boxplot([df_feat[df_feat['disease_risk']==r]['temperature'].values
                      for r in ['Low Risk','Medium Risk','High Risk']],
                     labels=['Low Risk','Medium Risk','High Risk'],
                     patch_artist=True)
for patch, color in zip(bp['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('Temperature by Risk Level', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Temperature (°C)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# KDE plot
for risk, color in colors.items():
    subset = df_feat[df_feat['disease_risk']==risk]['temperature']
    axes[1].hist(subset, bins=30, alpha=0.5, color=color, label=risk, edgecolor='white')
axes[1].set_title('Temperature Distribution by Risk', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Temperature (°C)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle('🌡️ Temperature Impact on Crop Disease Risk', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'temperature_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: temperature_impact.png")
"""))

cells.append(code("""
# ============================================================
# CELL 13: EDA - Humidity Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bp2 = axes[0].boxplot([df_feat[df_feat['disease_risk']==r]['humidity'].values
                       for r in ['Low Risk','Medium Risk','High Risk']],
                      labels=['Low Risk','Medium Risk','High Risk'],
                      patch_artist=True)
for patch, color in zip(bp2['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[0].set_title('Humidity by Risk Level', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Humidity (%)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# Scatter
scatter_colors = {'Low Risk':'#2ecc71','Medium Risk':'#f39c12','High Risk':'#e74c3c'}
for risk, color in scatter_colors.items():
    pool = df_feat[df_feat['disease_risk']==risk]
    sub = pool.sample(min(200, len(pool)), random_state=42)
    axes[1].scatter(sub['humidity'], sub['temperature'], alpha=0.5,
                    c=color, label=risk, s=25)
axes[1].set_title('Humidity vs Temperature', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Humidity (%)', fontsize=11)
axes[1].set_ylabel('Temperature (°C)', fontsize=11)
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('💧 Humidity Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'humidity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: humidity_analysis.png")
"""))


cells.append(code("""
# ============================================================
# CELL 14: EDA - Soil Health Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for risk, color in scatter_colors.items():
    pool = df_feat[df_feat['disease_risk']==risk]
    sub = pool.sample(min(200, len(pool)), random_state=42)
    axes[0].scatter(sub['soil_ph'], sub['soil_moisture'], alpha=0.5,
                    c=color, label=risk, s=25)
axes[0].set_title('Soil pH vs Soil Moisture', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Soil pH', fontsize=11)
axes[0].set_ylabel('Soil Moisture (%)', fontsize=11)
axes[0].legend(); axes[0].grid(alpha=0.3)

bp3 = axes[1].boxplot([df_feat[df_feat['disease_risk']==r]['crop_health_score'].values
                       for r in ['Low Risk','Medium Risk','High Risk']],
                      labels=['Low Risk','Medium Risk','High Risk'],
                      patch_artist=True)
for patch, color in zip(bp3['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_title('Crop Health Score by Risk Level', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Crop Health Score', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('🌱 Soil Health Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'soil_health_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: soil_health_analysis.png")
"""))

cells.append(code("""
# ============================================================
# CELL 15: EDA - Rainfall Impact
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bp4 = axes[0].boxplot([df_feat[df_feat['disease_risk']==r]['rainfall'].values
                       for r in ['Low Risk','Medium Risk','High Risk']],
                      labels=['Low Risk','Medium Risk','High Risk'],
                      patch_artist=True)
for patch, color in zip(bp4['boxes'], ['#2ecc71','#f39c12','#e74c3c']):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[0].set_title('Rainfall by Risk Level', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Rainfall (mm)', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

for risk, color in scatter_colors.items():
    pool = df_feat[df_feat['disease_risk']==risk]
    sub = pool.sample(min(200, len(pool)), random_state=42)
    axes[1].scatter(sub['rainfall'], sub['disease_pressure_index'], alpha=0.5,
                    c=color, label=risk, s=25)
axes[1].set_title('Rainfall vs Disease Pressure', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Rainfall (mm)', fontsize=11)
axes[1].set_ylabel('Disease Pressure Index', fontsize=11)
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('🌧️ Rainfall Impact Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'rainfall_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: rainfall_impact.png")
"""))

cells.append(code("""
# ============================================================
# CELL 16: EDA - Nutrient Analysis
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, nutrient, color_list in zip(
        axes,
        ['nitrogen','phosphorus','potassium'],
        [['#27ae60','#f0b429','#c0392b'],
         ['#1abc9c','#e67e22','#8e44ad'],
         ['#2980b9','#d35400','#16a085']]):
    bp5 = ax.boxplot([df_feat[df_feat['disease_risk']==r][nutrient].values
                      for r in ['Low Risk','Medium Risk','High Risk']],
                     labels=['Low\\nRisk','Med\\nRisk','High\\nRisk'],
                     patch_artist=True)
    for patch, clr in zip(bp5['boxes'], color_list):
        patch.set_facecolor(clr); patch.set_alpha(0.7)
    ax.set_title(f'{nutrient.capitalize()} (kg/ha)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('🧪 Nutrient Analysis by Disease Risk', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'nutrient_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: nutrient_analysis.png")
"""))


cells.append(code("""
# ============================================================
# CELL 17: Correlation Heatmap
# ============================================================
numeric_df = df_feat.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, square=True, annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
plt.title('🔥 Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: correlation_heatmap.png")
"""))

cells.append(md("## 🔢 Step 7: Encoding & Train-Test Split"))

cells.append(code("""
# ============================================================
# CELL 18: Encoding
# ============================================================
df_model = df_feat.copy()

# Encode crop_type
le_crop = LabelEncoder()
df_model['crop_type_enc'] = le_crop.fit_transform(df_model['crop_type'])

# Encode target
le_target = LabelEncoder()
df_model['target'] = le_target.fit_transform(df_model['disease_risk'])
label_map = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print("Label mapping:", label_map)

# Feature columns
feature_cols = [
    'crop_type_enc','temperature','humidity','rainfall','soil_moisture',
    'soil_ph','nitrogen','phosphorus','potassium','sunlight_hours',
    'pest_activity','leaf_discoloration','growth_rate','previous_disease_history',
    'crop_health_score','nutrient_balance_score','disease_pressure_index',
    'environmental_stress_score','pest_risk_score'
]

X = df_model[feature_cols].values
y = df_model['target'].values
print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape  : {y.shape}")
"""))

cells.append(code("""
# ============================================================
# CELL 19: Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaling for logistic regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train size : {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test size  : {X_test.shape[0]}  ({X_test.shape[0]/len(X)*100:.1f}%)")
print()
from collections import Counter
print("Train label distribution:", Counter(y_train))
print("Test  label distribution:", Counter(y_test))
"""))


cells.append(md("## 🤖 Step 8: Machine Learning Models"))

cells.append(code("""
# ============================================================
# CELL 20: Logistic Regression
# ============================================================
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial',
                        solver='lbfgs', random_state=42)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr  = f1_score(y_test, y_pred_lr, average='weighted')
prec_lr = precision_score(y_test, y_pred_lr, average='weighted')
rec_lr  = recall_score(y_test, y_pred_lr, average='weighted')

print(f"✅ Logistic Regression Results:")
print(f"   Accuracy  : {acc_lr:.4f} ({acc_lr*100:.2f}%)")
print(f"   F1-Score  : {f1_lr:.4f}")
print(f"   Precision : {prec_lr:.4f}")
print(f"   Recall    : {rec_lr:.4f}")
print()
print(classification_report(y_test, y_pred_lr, target_names=le_target.classes_))
"""))

cells.append(code("""
# ============================================================
# CELL 21: Random Forest
# ============================================================
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=150, max_depth=12,
                            min_samples_leaf=5, random_state=42,
                            n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf  = f1_score(y_test, y_pred_rf, average='weighted')
prec_rf = precision_score(y_test, y_pred_rf, average='weighted')
rec_rf  = recall_score(y_test, y_pred_rf, average='weighted')

print(f"✅ Random Forest Results:")
print(f"   Accuracy  : {acc_rf:.4f} ({acc_rf*100:.2f}%)")
print(f"   F1-Score  : {f1_rf:.4f}")
print(f"   Precision : {prec_rf:.4f}")
print(f"   Recall    : {rec_rf:.4f}")
print()
print(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))
"""))

cells.append(code("""
# ============================================================
# CELL 22: XGBoost Model
# ============================================================
print("Training XGBoost...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, verbosity=0
)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False)
y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb  = f1_score(y_test, y_pred_xgb, average='weighted')
prec_xgb = precision_score(y_test, y_pred_xgb, average='weighted')
rec_xgb  = recall_score(y_test, y_pred_xgb, average='weighted')

print(f"✅ XGBoost Results:")
print(f"   Accuracy  : {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"   F1-Score  : {f1_xgb:.4f}")
print(f"   Precision : {prec_xgb:.4f}")
print(f"   Recall    : {rec_xgb:.4f}")
print()
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))
"""))


cells.append(md("## 📈 Step 9: Model Comparison & Evaluation"))

cells.append(code("""
# ============================================================
# CELL 23: Model Comparison
# ============================================================
comparison = {
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [acc_lr, acc_rf, acc_xgb],
    'F1-Score (Weighted)': [f1_lr, f1_rf, f1_xgb],
    'Precision': [prec_lr, prec_rf, prec_xgb],
    'Recall': [rec_lr, rec_rf, rec_xgb]
}
comp_df = pd.DataFrame(comparison)
comp_df[['Accuracy','F1-Score (Weighted)','Precision','Recall']] = \\
    comp_df[['Accuracy','F1-Score (Weighted)','Precision','Recall']].round(4)
print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(comp_df.to_string(index=False))

best_idx = comp_df['F1-Score (Weighted)'].idxmax()
best_model_name = comp_df.loc[best_idx, 'Model']
print(f"\\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score : {comp_df.loc[best_idx, 'F1-Score (Weighted)']:.4f}")
print(f"   Accuracy : {comp_df.loc[best_idx, 'Accuracy']:.4f}")
"""))

cells.append(code("""
# ============================================================
# CELL 24: Confusion Matrices
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
model_results = [
    ('Logistic Regression', y_pred_lr),
    ('Random Forest', y_pred_rf),
    ('XGBoost', y_pred_xgb)
]
cmaps = ['Blues', 'Greens', 'Oranges']
for ax, (name, preds), cmap in zip(axes, model_results, cmaps):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=le_target.classes_,
                yticklabels=le_target.classes_,
                linewidths=0.5)
    acc = accuracy_score(y_test, preds)
    ax.set_title(f'{name}\\nAccuracy: {acc*100:.2f}%', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

plt.suptitle('🎯 Confusion Matrices - All Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: confusion_matrix.png")
"""))

cells.append(code("""
# ============================================================
# CELL 25: Feature Importance
# ============================================================
rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
xgb_imp = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

colors_rf  = ['#e74c3c' if i >= len(rf_imp)-5 else '#3498db' for i in range(len(rf_imp))]
colors_xgb = ['#e74c3c' if i >= len(xgb_imp)-5 else '#2ecc71' for i in range(len(xgb_imp))]

axes[0].barh(rf_imp.index, rf_imp.values, color=colors_rf, edgecolor='white')
axes[0].set_title('Random Forest Feature Importance', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Importance Score', fontsize=11)
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(xgb_imp.index, xgb_imp.values, color=colors_xgb, edgecolor='white')
axes[1].set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('🌟 Feature Importance Analysis', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(VISUALS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: feature_importance.png")
"""))


cells.append(md("## 💾 Step 10: Save Model & Generate Reports"))

cells.append(code("""
# ============================================================
# CELL 26: Save Best Model
# ============================================================
model_map = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'XGBoost': xgb
}
best_model = model_map[best_model_name]
best_preds = {'Logistic Regression': y_pred_lr,
              'Random Forest': y_pred_rf,
              'XGBoost': y_pred_xgb}[best_model_name]

model_bundle = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder_target': le_target,
    'label_encoder_crop': le_crop,
    'feature_cols': feature_cols,
    'best_model_name': best_model_name
}

model_path = MODELS_DIR / 'crop_disease_predictor.pkl'
joblib.dump(model_bundle, model_path)
print(f"✅ Best model saved: {model_path}")
print(f"   Model      : {best_model_name}")
print(f"   Accuracy   : {accuracy_score(y_test, best_preds)*100:.2f}%")
"""))

cells.append(code("""
# ============================================================
# CELL 27: Save Metrics JSON
# ============================================================
metrics = {
    'project': 'CropGuard AI - Crop Disease Risk Analytics',
    'dataset': {
        'total_records': int(len(df_feat)),
        'features': int(len(feature_cols)),
        'engineered_features': 5,
        'target_classes': list(le_target.classes_)
    },
    'models': {
        'Logistic Regression': {
            'accuracy': round(acc_lr, 4),
            'f1_score': round(f1_lr, 4),
            'precision': round(prec_lr, 4),
            'recall': round(rec_lr, 4)
        },
        'Random Forest': {
            'accuracy': round(acc_rf, 4),
            'f1_score': round(f1_rf, 4),
            'precision': round(prec_rf, 4),
            'recall': round(rec_rf, 4)
        },
        'XGBoost': {
            'accuracy': round(acc_xgb, 4),
            'f1_score': round(f1_xgb, 4),
            'precision': round(prec_xgb, 4),
            'recall': round(rec_xgb, 4)
        }
    },
    'best_model': best_model_name,
    'best_model_accuracy': round(accuracy_score(y_test, best_preds), 4),
    'best_model_f1': round(f1_score(y_test, best_preds, average='weighted'), 4)
}

metrics_path = REPORTS_DIR / 'model_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"✅ Metrics saved: {metrics_path}")
print(json.dumps(metrics, indent=2))
"""))

cells.append(code("""
# ============================================================
# CELL 28: Generate Project Report
# ============================================================
report = f\"\"\"# CropGuard AI - Project Report

## Executive Summary
CropGuard AI is a machine learning system that predicts crop disease risk
levels (Low, Medium, High) using environmental and agronomic data.

## Dataset Overview
- Total Records   : {len(df_feat):,}
- Original Features: 14
- Engineered Features: 5 (crop_health_score, nutrient_balance_score,
  disease_pressure_index, environmental_stress_score, pest_risk_score)
- Total Features Used: {len(feature_cols)}
- Target Classes  : {list(le_target.classes_)}

## Class Distribution
{df_feat['disease_risk'].value_counts().to_string()}

## Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression | {acc_lr*100:.2f}% | {f1_lr:.4f} | {prec_lr:.4f} | {rec_lr:.4f} |
| Random Forest       | {acc_rf*100:.2f}% | {f1_rf:.4f} | {prec_rf:.4f} | {rec_rf:.4f} |
| XGBoost             | {acc_xgb*100:.2f}% | {f1_xgb:.4f} | {prec_xgb:.4f} | {rec_xgb:.4f} |

## Best Model
**{best_model_name}**
- Accuracy : {accuracy_score(y_test, best_preds)*100:.2f}%
- F1-Score : {f1_score(y_test, best_preds, average='weighted'):.4f}

## Key Findings
1. Disease pressure index and pest risk score are top predictors.
2. High humidity (>75%) combined with elevated temperatures increases risk.
3. Nutrient imbalance significantly correlates with high disease risk.
4. Previous disease history is a strong predictor of future risk.
5. Crop health score effectively distinguishes Low vs High risk cases.

## Visualizations Generated
- disease_risk_distribution.png
- temperature_impact.png
- humidity_analysis.png
- soil_health_analysis.png
- rainfall_impact.png
- nutrient_analysis.png
- correlation_heatmap.png
- feature_importance.png
- confusion_matrix.png

## Files
- models/crop_disease_predictor.pkl
- reports/model_metrics.json
- reports/project_report.md
\"\"\"

report_path = REPORTS_DIR / 'project_report.md'
with open(report_path, 'w') as f:
    f.write(report)
print(f"✅ Report saved: {report_path}")
print("\\n" + "="*60)
print("🎉 CropGuard AI Analysis Complete!")
print("="*60)
print(f"  Best Model : {best_model_name}")
print(f"  Accuracy   : {accuracy_score(y_test, best_preds)*100:.2f}%")
print(f"  Model saved: models/crop_disease_predictor.pkl")
print(f"  Visuals    : {len(list(VISUALS_DIR.glob('*.png')))} PNG files generated")
"""))

# Assemble notebook
nb.cells = cells

import os
os.makedirs('notebooks', exist_ok=True)
with open('notebooks/cropguard_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("✅ Notebook created: notebooks/cropguard_analysis.ipynb")
