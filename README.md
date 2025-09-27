# 📊 Automated Exploratory Data Analysis (EDA) — Python Library Cheatsheet

This guide summarizes key Python libraries used for automated EDA, with practical examples and usage notes. Perfect for data scientists, analysts, and business intelligence professionals looking to streamline their data exploration process.

---

## 🧰 Libraries & Examples
### Libraries to install 
```python
!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install ydata-profiling
!pip install sweetviz
!pip install autoviz
!pip install dtale
!pip install missingno
!pip install plotly
!pip install scikit-learn
!pip install dataprep
!pip install pandas-visual-analysis

```

---



### 1. `pandas` — Data Manipulation
```python
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
print(df.describe())
```

---

### 2. `numpy` — Numerical Operations
```python
import numpy as np

arr = np.array([1, 2, 3, 4])
print(np.mean(arr))
print(np.std(arr))
```

---

### 3. `matplotlib` — Basic Plotting
```python
import matplotlib.pyplot as plt

df['age'].hist(bins=10)
plt.title("Age Distribution")
plt.show()
```

---

### 4. `seaborn` — Statistical Visualization
```python
import seaborn as sns

sns.boxplot(x='gender', y='income', data=df)
plt.title("Income by Gender")
plt.show()
```

---

### 5. `pandas_profiling` / `ydata_profiling` — Automated Reports
```python
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
```

---

### 6. `sweetviz` — Comparative EDA
```python
import sweetviz as sv

report = sv.analyze(df)
report.show_html("sweetviz_report.html")
```

---

### 7. `autoviz` — Auto Visualization
```python
from autoviz.AutoViz_Class import AutoViz_Class

AV = AutoViz_Class()
df = AV.AutoViz("data.csv")
```

---

### 8. `dtale` — Interactive DataFrame Explorer
```python
import dtale

d = dtale.show(df)
d.open_browser()
```

---

### 9. `missingno` — Missing Data Visualization
```python
import missingno as msno

msno.matrix(df)
plt.show()
```

---

### 10. `plotly` — Interactive Dashboards
```python
import plotly.express as px

fig = px.scatter(df, x="age", y="income", color="gender")
fig.show()
```

---

### 11. `sklearn` — Preprocessing & Feature Engineering
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])
```

---

### 12. `warnings` — Suppress Warnings
```python
import warnings
warnings.filterwarnings("ignore")
```

---

### 13. `os` / `sys` — File & System Utilities
```python
import os

print(os.getcwd())
```

---

## 🆕 Additional Automated EDA Libraries

### 14. `dataprep.eda` — One-Line EDA
```python
from dataprep.eda import create_report

create_report(df).show_browser()
```
- Generates interactive reports with distributions, correlations, and missing value analysis.

---

### 15. `pandas_visual_analysis` — Real-Time Dashboard
```python
from pandas_visual_analysis import VisualAnalysis

VisualAnalysis(df)
```
- Launches a live dashboard for filtering, plotting, and exploring data.

---

### 16. `RATH` — Augmented Analytics Engine
- RATH is a next-gen open-source tool that automates EDA with causal analysis, dashboard generation, and pattern discovery.
- Supports MySQL, Redshift, Hive, Spark SQL, and more.

> Try the [RATH Online Demo](https://docs.kanaries.net/articles/python-auto-eda) for a hands-on experience.

---

## ✅ Suggested Workflow

1. Load data using `pandas`.
2. Run `pandas_profiling`, `sweetviz`, or `dataprep` for initial insights.
3. Visualize missing data with `missingno`.
4. Use `autoviz`, `plotly`, or `pandas_visual_analysis` for deeper visual analysis.
5. Encode and preprocess with `sklearn`.
6. Explore interactively with `dtale` or `RATH`.

---
