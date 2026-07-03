# 🧩 Statistical Machine Learning — Foundation

Part of the [📘 Statistical Machine Learning for Noob](https://github.com/mdnuruzzamanKALLOL) series — a from-scratch, math-first, fully-executed ML study series designed to be read on GitHub, run locally, and actually learned from (not just skimmed).

**Foundation** is where it starts: the Python data tools, visualization skills, cleaning techniques, feature engineering, and math that every algorithm in the [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) repo assumes you already have. Skip this and every later notebook will feel harder than it needs to be.

## 📑 Table of Contents

1. [Why This Repo Exists](#-why-this-repo-exists)
2. [Learning Path](#-learning-path)
3. [Topics](#-topics)
4. [What Makes This Different](#-what-makes-this-different)
5. [Repository Structure](#-repository-structure)
6. [Prerequisites](#-prerequisites)
7. [How to Use](#-how-to-use)
8. [Datasets](#-datasets)
9. [The Full Series](#-the-full-series)
10. [Self-Test Philosophy](#-self-test-philosophy)
11. [Feedback & Corrections](#-feedback--corrections)
12. [Related](#-related)

---

## 🎯 Why This Repo Exists

Most "ML for beginners" material either (a) waves hands at the math and jumps straight to `.fit()`, or (b) drowns beginners in proofs before they've touched a dataset. This series tries to do neither: every notebook pairs **runnable code** with a **README that derives the actual math** (in LaTeX, not just prose), and every notebook is **fully executed** before being committed — the outputs you see on GitHub are real, not hand-typed.

## 🗺️ Learning Path

```
01 Python, NumPy & Pandas   →  the data structures everything else sits on top of
        ↓
02 Data Visualization        →  seeing your data before touching an algorithm
        ↓
03 Data Preprocessing & EDA  →  turning raw, messy data into a clean feature matrix
        ↓
04 Feature Engineering       →  making that feature matrix actually predictive
        ↓
05 Math Refresher            →  the linear algebra / probability / calculus every
                                  algorithm ahead is literally built from
        ↓
   Classical ML repo         →  Regression → Classification → Ensembles →
                                  Unsupervised → Model Evaluation & Tuning
```

Each topic explicitly says what it's setting up for — read the "Next up" line at the end of every notebook, and the "Why This Topic Matters" section at the top of every README.

## 📚 Topics

| # | Topic | Cells | Key Concepts | Status |
|---|-------|:---:|-------------|:---:|
| 01 | [Python, NumPy & Pandas](01_Python_Numpy_Pandas/) | 36 | Vectorization, broadcasting, `axis`, `loc`/`iloc`, merge/pivot, groupby | ✅ |
| 02 | [Data Visualization](02_Data_Visualization/) | 32 | Matplotlib Figure/Axes, Seaborn statistical plots, KDE, correlation heatmaps, IQR outliers | ✅ |
| 03 | [Data Preprocessing & EDA](03_Data_Preprocessing_EDA/) | 32 | Missing-data strategies, encoding, scaling, class imbalance, sklearn `Pipeline` | ✅ |
| 04 | [Feature Engineering](04_Feature_Engineering/) | 32 | Cyclical/target encoding, RFE, permutation importance, PCA preview, target-leakage detection | ✅ |
| 05 | [Math Refresher](05_Math_Refresher/) | 32 | Normal Equation, eigenvectors/SVD, Bayes' theorem, CLT, hypothesis testing, gradient descent | ✅ |

**164 total cells**, every one executed — no dead code, no untested claims.

## ⭐ What Makes This Different

- **Every notebook is actually executed**, not just written — cell outputs on GitHub are real, including rendered plots, not placeholders. Two real bugs (`np.select` dtype mismatch, a date-range gap that silently excluded December) were caught by this process and are documented in the commit history and this series' notes.
- **Every README derives the math**, not just names it — broadcasting rules, the Normal Equation, Bayes' theorem, the IQR rule, target/cyclical encoding formulas, all in LaTeX, all tied back to *why* the Classical ML algorithms ahead work the way they do.
- **Mistakes are kept, not hidden** — the target-leakage bug caught mid-build in topic 04 is left in as a worked example, because catching that exact mistake is a real skill.
- **164 dataset options** ([catalog + CSVs](https://github.com/mdnuruzzamanKALLOL/Datasets)), all load-tested live, mapped to which Classical ML topic each one fits best.

## 📁 Repository Structure

```
Statistical-Machine-Learning-Foundation/
├── README.md                          ← you are here
├── 01_Python_Numpy_Pandas/
│   ├── README.md                      ← concept + math + function reference + exercises
│   └── 01_python_numpy_pandas.ipynb   ← 36 executed cells
├── 02_Data_Visualization/
│   ├── README.md
│   └── 02_data_visualization.ipynb    ← 32 executed cells
├── 03_Data_Preprocessing_EDA/
│   ├── README.md
│   └── 03_data_preprocessing_eda.ipynb ← 32 executed cells
├── 04_Feature_Engineering/
│   ├── README.md
│   └── 04_feature_engineering.ipynb   ← 32 executed cells
└── 05_Math_Refresher/
    ├── README.md
    └── 05_math_refresher.ipynb        ← 32 executed cells
```

Every topic folder is self-contained: read the `README.md` for the theory, open the `.ipynb` for the hands-on implementation. They're designed to be read together — the notebook links back to the README for full derivations, the README links to the notebook for the runnable version.

## 🧰 Prerequisites

- Python 3.9+ (developed and tested on 3.14)
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`
- No deep learning framework needed anywhere in this repo — that's a different series

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

## 🚀 How to Use

**Just reading?** Every notebook renders directly on GitHub with full output — click any `.ipynb` link above.

**Running it yourself:**

```bash
git clone https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation.git
cd Statistical-Machine-Learning-Foundation
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook 01_Python_Numpy_Pandas/01_python_numpy_pandas.ipynb
```

Every notebook is self-contained and runs top-to-bottom with `random_state=42` (or an explicit seed) wherever randomness is involved, so your output will match what's shown on GitHub.

## 📦 Datasets

All datasets — the 162-entry load-tested catalog ([CATALOG.md](https://github.com/mdnuruzzamanKALLOL/Datasets/blob/main/CATALOG.md)) and the actual CSV files (212 curated + this author's full personal collection, 418 total) — live in the central **[Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets)** repo, organized by source (`sklearn/`, `seaborn/`, `openml/`, `synthetic/`, `personal_collection/`) with a full manifest index.

## 🔭 The Full Series

| Repo | Covers | Status |
|---|---|---|
| **Foundation** (this repo) | Python/NumPy/Pandas, Visualization, Preprocessing, Feature Engineering, Math | ✅ Complete |
| [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML) | Regression, Classification, Ensembles, Unsupervised, Model Evaluation & Tuning (29 algorithms) | 🚧 In progress |
| [Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets) | 418 datasets (162 cataloged + personal collection) backing both repos above | ✅ Complete |

## 📝 Self-Test Philosophy

Every topic README ends with a **Self-Test Exercises** section — deliberately *not* answered inline. The point is to predict the answer before running the corresponding notebook cell, not to read a solution. If an exercise feels hard, that's the signal to re-read that section, not skip it.

## 💬 Feedback & Corrections

Found a bug, a math error, or a broken dataset link? Open an issue or a pull request — this series documents its own mistakes on purpose (see topic 04's target-leakage story), so a caught error is a feature, not an embarrassment.

## 🔗 Related

- [Classical ML →](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML)
- [Datasets →](https://github.com/mdnuruzzamanKALLOL/Datasets)
- [Author's GitHub profile](https://github.com/mdnuruzzamanKALLOL)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.Foundation.root&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
