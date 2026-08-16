
# Multi-View Clustering for Localized Global Time Series Forecasting

> **P1 Project — Master of Computer Science, University of Vienna**

A time-series forecasting project investigating whether **multi-view clustering can improve localized electricity-consumption forecasting** compared with a single global forecasting model.

The project implements and evaluates multi-view clustering methods and combines the resulting household groups with cluster-specific forecasting models.

---

## 📌 Overview

Accurate electricity-consumption forecasting is important for energy management, demand planning, and efficient resource allocation. However, households can exhibit very different consumption behaviors, making a single forecasting model less effective for all customers.

This project investigates the following idea:

> **Can households with similar consumption behavior be grouped into clusters, and can forecasting models trained separately for those clusters improve prediction accuracy?**

The project therefore combines:

**Household behavior analysis → Feature engineering → Multi-view clustering → Cluster analysis → Time-series forecasting → Model comparison**

The project is divided into two experimental phases:

- **Phase 1:** Evaluation of clustering approaches on publicly available UCI time-series datasets.
- **Phase 2:** Application of the methodology to a large-scale household electricity-consumption dataset.

---

# 🎯 Objectives

The main objectives of the project are to:

- Analyze time-series consumption behavior.
- Represent the data through multiple complementary views.
- Apply multi-view clustering to identify groups of similar time series.
- Compare multi-view clustering with conventional clustering approaches.
- Train global and cluster-specific forecasting models.
- Evaluate whether clustering improves forecasting performance.
- Investigate different forecasting strategies and feature configurations.
- Analyze the effect of clustering on different household behavior groups.

---

# 🔬 Methodology

The overall workflow is:

```text
                    Raw Time-Series Data
                            │
                            ▼
                   Data Preprocessing
                            │
                            ▼
                  Feature Engineering
                            │
                            ▼
              ┌──────────────────────────┐
              │      Multiple Views      │
              │                          │
              │  View 1   View 2   ...   │
              └──────────────────────────┘
                            │
                            ▼
                  Multi-View Clustering
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
           OFMVC           FMVC         Baselines
                                         K-Means
                                         k-Shape
                            │
                            ▼
                    Cluster Assignment
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
       Global Model                 Cluster Models
             │                             │
             └──────────────┬──────────────┘
                            ▼
                    2024 Forecasting
                            │
                            ▼
                    MAE Evaluation
                            │
                            ▼
                    Model Comparison
````

---

# 📊 Phase 1 — UCI Time-Series Experiments

The first phase evaluates the clustering approaches on multiple publicly available time-series datasets.

The datasets investigated include:

* **WISDM**
* **Human Activity Recognition (HAR)**
* **MHEALTH**
* **PAMAP2**

The purpose of this phase was to evaluate the behavior of the multi-view clustering approaches on different types of time-series data before applying them to the electricity-consumption problem.

The clustering methods investigated include:

* **OFMVC**
* **FMVC**
* **K-Means**
* **k-Shape**

---

# ⚡ Phase 2 — Electricity Consumption Forecasting

The second phase applies the methodology to household electricity-consumption data.

The dataset contains approximately **17,547 households** with daily electricity consumption observations.

| Dataset | Purpose    | Time steps |
| ------- | ---------- | ---------: |
| 2023    | Training   |   365 days |
| 2024    | Evaluation |   366 days |

The electricity-consumption data is not included in this repository because of its large file size.

---

# 🧹 Data Preparation

Several preprocessing steps were performed before clustering.

### Household-level features

Consumption statistics were used to characterize household behavior, including:

* Mean consumption
* Standard deviation
* Maximum consumption
* Minimum consumption
* Consumption range
* Coefficient of variation
* Skewness
* Number of zero-consumption days
* Percentage of zero-consumption days
* 90th percentile
* 10th percentile

Log-transformed versions of selected highly skewed features were also introduced.

This resulted in **13 clustering features**.

### Normalization

Features were standardized using **z-score normalization (`StandardScaler`)** so that features with larger numerical ranges would not dominate the clustering process.

### Missing values

Constant-zero households can produce undefined statistical features such as skewness.

These missing values were handled through mean imputation after standardization.

---

# 👥 Clustering

The project compares several clustering approaches.

## OFMVC

**OFMVC (Online Fuzzy Multi-View Clustering)** is the primary multi-view clustering approach investigated in the project.

The implementation used in this project includes an important correction to the original implementation: the **view-weight optimization step**, which was commented out in the original code, was activated and made optional through a `learn_weights` parameter.

This allows the implementation to support both:

* the original/fixed-weight behavior
* the corrected view-weight learning behavior

The modification is documented in the project report.

## FMVC

**FMVC (Fuzzy Multi-View Clustering)** is used as an additional multi-view clustering comparison.

## K-Means

K-Means provides a conventional single-view clustering baseline.

## k-Shape

k-Shape provides a time-series-specific clustering baseline based on the shape of time-series patterns.

---

# 🔧 OFMVC Implementation Correction

The original OFMVC implementation contained a commented-out view-weight optimization step.

The project implementation was modified to make this optimization functional.

The main changes include:

* Added a `learn_weights` parameter.
* Preserved the original behavior when `learn_weights=False`.
* Activated view-weight optimization when `learn_weights=True`.
* Added safe initialization of the view weights.
* Updated the return values to expose the learned weights.

This correction is an important part of the project because the goal is to evaluate **multi-view clustering with learned view importance** rather than treating every view identically.

---

# 📈 Forecasting

After clustering households, forecasting models are trained to investigate whether localized models perform better than a single global model.

## Global Forecasting Model

A single **LightGBM** model is trained using the complete training population.

The model uses calendar and lag-based features, including:

* `day_of_year`
* `day_of_week`
* `month`
* `weekend`
* `lag_1`
* `lag_2`
* `lag_3`
* `lag_7`
* `lag_14`
* `lag_28`

## Cluster-Based Forecasting

Separate LightGBM models are trained for individual household clusters.

The idea is that households within the same cluster have more homogeneous consumption patterns, allowing each model to specialize in its corresponding group.

---

# 🔁 Recursive Forecasting

An additional forecasting experiment uses a recursive forecasting strategy.

The model starts with historical observations from the end of 2023 and predicts the next day.

The prediction is then added to the history and used to generate the next prediction.

This process continues throughout the 366 days of 2024.

The recursive model uses features such as:

* Calendar features
* Lag features
* Rolling averages
* Trend features

The recursive approach was investigated because it allows predictions to incorporate the most recently predicted consumption values.

---

# 🧪 Forecasting Experiments

Several forecasting configurations were investigated, including:

### Global LightGBM

One model trained across all households.

### Cluster-Based LightGBM

Separate models trained for the identified household clusters.

### XGBoost

An alternative forecasting model was tested with additional engineered features such as:

* Rolling averages
* Weekly differences
* Cyclical seasonal features

The experiment did not improve the baseline forecasting performance.

### Recursive LightGBM

A recursive forecasting strategy was also evaluated to investigate whether sequential prediction improves long-horizon forecasting.

---

# 📏 Evaluation

Forecasting performance is primarily evaluated using:

### Mean Absolute Error (MAE)

[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
]

MAE measures the average absolute difference between the actual and predicted electricity consumption.

The evaluation compares:

* Global forecasting
* Cluster-based forecasting
* Different clustering approaches
* Different forecasting strategies

Performance is also examined at the cluster level to understand which household groups benefit most from localized forecasting.

---

# 🔍 Key Findings

The experiments show that household consumption behavior is heterogeneous and that clustering can help identify groups with different forecasting characteristics.

In the forecasting experiments:

* Cluster-specific models can outperform a single global model for particular household groups.
* Low-consumption or inactive households are easier to forecast because of their relatively stable patterns.
* High-consumption households are more difficult to predict because of greater variability.
* Recursive forecasting can improve long-horizon prediction by incorporating recent predictions.
* XGBoost with the tested extended feature configuration did not outperform the baseline LightGBM approach.

The project therefore demonstrates the potential benefit of combining **behavior-based clustering with localized forecasting models**.

---

# 📁 Repository Structure

The repository contains the main implementation code used for the project.

```text
Multi-View-Clustering-Time-Series-Forecasting/
│
├── Main Project Scripts/
│   ├── clustering/
│   ├── forecasting/
│   ├── preprocessing/
│   └── ...
│
├── P1_Report_Fatima_Wishal.pdf
│
├── README.md
└── ...
```

> **Note:** The repository intentionally contains the main project scripts rather than the complete datasets and generated outputs.

The complete electricity datasets and some generated experimental artifacts are too large to include in the GitHub repository.

---

# 💾 Data Availability

The original electricity-consumption datasets are **not included in this repository due to file-size limitations**.

The experiments use:

* 2023 household electricity consumption for training
* 2024 household electricity consumption for evaluation
* Additional weather information where applicable

If access to the original datasets is required, they should be obtained from the project/course data source.

---

# ⚙️ Requirements

The project was developed primarily in Python and uses scientific computing, machine-learning, time-series, and visualization libraries.

Main dependencies include:

```text
Python 3.10
numpy
pandas
scipy
scikit-learn
matplotlib
statsmodels
tensorflow
tslearn
lightgbm
xgboost
```

Some clustering experiments may require specific package versions or separate environments because of compatibility issues between time-series libraries and NumPy.

---

# 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/wishalfatima/Multi-View-Clustering-Time-Series-Forecasting.git
cd Multi-View-Clustering-Time-Series-Forecasting
```

Create a Python environment:

```bash
conda create -n mvc_ts python=3.10
conda activate mvc_ts
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Place the required datasets in the paths expected by the scripts.

Then execute the preprocessing, clustering, and forecasting scripts in the appropriate order.

> **Important:** Some scripts contain paths specific to the original development environment. These paths may need to be updated before execution on another machine.

---

# 📚 Reference Implementation

The project builds upon the original implementation associated with:

**Multi-view clustering for localized global time series forecasting**

Original repository:

[https://github.com/AliReza000J/MVC-Time-series-Forecasting](https://github.com/AliReza000J/MVC-Time-series-Forecasting)

The original implementation was used as a reference, with modifications made for the experiments and implementation correction described in this project.

---

# 🎓 Project Context

This project was completed as part of the **P1 Project / Master's programme in Computer Science at the University of Vienna**.

The project investigates the intersection of:

* Multi-view learning
* Time-series clustering
* Fuzzy clustering
* Electricity-consumption analysis
* Machine learning
* Time-series forecasting
* Cluster-based forecasting

---

# 👩‍💻 Author

**Wishal Fatima**

Master's Student — Computer Science
University of Vienna

GitHub: [https://github.com/wishalfatima](https://github.com/wishalfatima)



