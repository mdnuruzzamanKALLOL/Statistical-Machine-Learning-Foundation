# 📦 Dataset Catalog — 114 Datasets for Statistical ML Practice

A curated, **fully load-tested** catalog of datasets for practicing every algorithm in this series (Foundation + [Classical ML](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Classical-ML)). Every entry below is a one-line load call — no manual downloads, no broken links. Every single name in this file was verified to actually load, live, while building this catalog.

No raw data files are stored in this repo on purpose: it keeps the repo lightweight, avoids redistribution/licensing issues, and — more importantly — teaches the real-world skill of pulling data from standard APIs (`sklearn.datasets`, `seaborn`, `OpenML`), which is what you'll actually do in practice.

## 📊 Catalog Summary

| Source | Count | Best for |
|---|---|---|
| [scikit-learn built-in / fetchable](#-1-scikit-learn-built-in--fetchable-13) | 13 | Classic benchmarks, zero setup |
| [Seaborn built-in](#-2-seaborn-built-in-22) | 22 | EDA practice, quick visualization datasets |
| [OpenML — batch 1](#-3-openml-real-world-batch-1-30) | 30 | Classification-heavy real-world practice |
| [OpenML — batch 2](#-4-openml-real-world-batch-2-21) | 21 | More classification + a few regression |
| [Synthetic generators](#-5-synthetic-generators-sklearndatasetsmake_-28) | 28 | Algorithm-targeted practice (clustering, manifolds, imbalanced classes...) |
| **Total** | **114** | |

## 📑 Table of Contents

1. [How to Use This Catalog](#-how-to-use-this-catalog)
2. [1. scikit-learn Built-in / Fetchable (13)](#-1-scikit-learn-built-in--fetchable-13)
3. [2. Seaborn Built-in (22)](#-2-seaborn-built-in-22)
4. [3. OpenML Real-World, Batch 1 (30)](#-3-openml-real-world-batch-1-30)
5. [4. OpenML Real-World, Batch 2 (21)](#-4-openml-real-world-batch-2-21)
6. [5. Synthetic Generators — `sklearn.datasets.make_*` (28)](#-5-synthetic-generators-sklearndatasetsmake_-28)
7. [Which Dataset for Which Topic?](#-which-dataset-for-which-topic)
8. [Caveats & Notes](#️-caveats--notes)

---

## 🚀 How to Use This Catalog

Every dataset resolves to a Pandas DataFrame in one line:

```python
# scikit-learn built-in
from sklearn.datasets import load_iris
df = load_iris(as_frame=True).frame

# seaborn built-in
import seaborn as sns
df = sns.load_dataset("titanic")

# OpenML (real-world, name-based lookup)
from sklearn.datasets import fetch_openml
df = fetch_openml(name="credit-g", as_frame=True, parser="auto").frame

# synthetic generator
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=3, random_state=42)
```

First call downloads and caches (`~/scikit_learn_data/`); every call after that is instant and offline.

---

## 🧱 1. scikit-learn Built-in / Fetchable (13)

| # | Name | Task | Load Code | Notes |
|---|---|---|---|---|
| 1 | Iris | Classification (3-class) | `load_iris(as_frame=True).frame` | The "hello world" of ML — 150 rows, 4 features |
| 2 | Wine | Classification (3-class) | `load_wine(as_frame=True).frame` | Chemical analysis of wines from 3 cultivars |
| 3 | Breast Cancer Wisconsin | Classification (binary) | `load_breast_cancer(as_frame=True).frame` | Diagnostic features, malignant vs benign |
| 4 | Diabetes | Regression | `load_diabetes(as_frame=True).frame` | Disease progression one year after baseline |
| 5 | Digits | Classification (10-class) | `load_digits(as_frame=True).frame` | 8×8 handwritten digit images, flattened |
| 6 | Linnerud | Regression (multi-output) | `load_linnerud(as_frame=True).frame` | Exercise vs physiological measurements, tiny (20 rows) |
| 7 | California Housing | Regression | `fetch_california_housing(as_frame=True).frame` | ~20,600 rows, median house value target |
| 8 | Covertype | Classification (7-class, large) | `fetch_covtype(as_frame=True).frame` | ~580,000 rows — good for testing scalability |
| 9 | 20 Newsgroups | Text Classification | `fetch_20newsgroups(subset="train")` | ~18,000 posts, 20 categories |
| 10 | Olivetti Faces | Image Classification | `fetch_olivetti_faces()` | 400 face images, 40 people |
| 11 | LFW People | Image Classification | `fetch_lfw_people(min_faces_per_person=70)` | "Labeled Faces in the Wild" subset |
| 12 | RCV1 | Text Multilabel | `fetch_rcv1()` | Reuters news categorization, large |
| 13 | KDD Cup 99 | Classification (intrusion detection) | `fetch_kddcup99(subset="SA")` | Network intrusion detection benchmark |

---

## 🎨 2. Seaborn Built-in (22)

| # | Name | Suggested Task | Load Code | Notes |
|---|---|---|---|---|
| 14 | tips | Regression (`tip`) | `sns.load_dataset("tips")` | Restaurant tipping, classic Seaborn demo dataset |
| 15 | titanic | Classification (`survived`) | `sns.load_dataset("titanic")` | Different columns than the OpenML version — good for comparing sources |
| 16 | iris | Classification (`species`) | `sns.load_dataset("iris")` | Same flowers, Seaborn-formatted |
| 17 | penguins | Classification (`species`) | `sns.load_dataset("penguins")` | Modern Iris alternative, has real missing values |
| 18 | diamonds | Regression (`price`) | `sns.load_dataset("diamonds")` | ~54,000 rows, great for scaling/EDA practice |
| 19 | mpg | Regression (`mpg`) | `sns.load_dataset("mpg")` | Car fuel efficiency, has missing values |
| 20 | planets | EDA / Regression | `sns.load_dataset("planets")` | Exoplanet discovery methods and masses |
| 21 | flights | Time series / EDA | `sns.load_dataset("flights")` | Monthly airline passengers, classic seasonality demo |
| 22 | exercise | EDA (ANOVA-style) | `sns.load_dataset("exercise")` | Small, good for groupby/categorical practice |
| 23 | anagrams | EDA | `sns.load_dataset("anagrams")` | Tiny experimental dataset |
| 24 | anscombe | EDA — **"always plot your data"** | `sns.load_dataset("anscombe")` | Anscombe's Quartet: 4 datasets, identical stats, wildly different shapes |
| 25 | attention | EDA | `sns.load_dataset("attention")` | Small psychology experiment dataset |
| 26 | brain_networks | EDA (high-dimensional) | `sns.load_dataset("brain_networks")` | 63 columns — good for correlation/PCA practice |
| 27 | car_crashes | Regression | `sns.load_dataset("car_crashes")` | US state-level crash statistics |
| 28 | dots | EDA / time series | `sns.load_dataset("dots")` | Neuroscience-style response-time dataset |
| 29 | dowjones | Time series | `sns.load_dataset("dowjones")` | Historical Dow Jones index |
| 30 | fmri | EDA / time series | `sns.load_dataset("fmri")` | Brain imaging signal over time |
| 31 | geyser | **Clustering** (classic!) | `sns.load_dataset("geyser")` | Old Faithful eruptions — the textbook 2-cluster demo |
| 32 | glue | EDA | `sns.load_dataset("glue")` | NLP benchmark scores |
| 33 | healthexp | Time series / Regression | `sns.load_dataset("healthexp")` | Health expenditure vs life expectancy by country |
| 34 | seaice | Time series / Regression | `sns.load_dataset("seaice")` | Arctic sea ice extent over time |
| 35 | taxis | Classification / Regression | `sns.load_dataset("taxis")` | NYC taxi trips — fare (regression) or payment type (classification) |

---

## 🌍 3. OpenML Real-World, Batch 1 (30)

All loaded via `fetch_openml(name="<name>", as_frame=True, parser="auto").frame`.

| # | Name | Task | Rows | Notes |
|---|---|---|---|---|
| 36 | `titanic` | Classification | 2,201 | OpenML's version — compare against seaborn's |
| 37 | `adult` | Classification (income >$50K) | 48,842 | The classic "Census Income" benchmark |
| 38 | `credit-g` | Classification (credit risk) | 1,000 | German Credit dataset |
| 39 | `mushroom` | Classification (edible/poisonous) | 8,124 | Perfectly separable with the right features |
| 40 | `spambase` | Classification (spam) | 4,601 | Email spam detection, 57 numeric features |
| 41 | `ionosphere` | Classification | 351 | Radar returns, good/bad |
| 42 | `sonar` | Classification (rock/mine) | 208 | Small, high-dimensional (60 features) |
| 43 | `glass` | Classification (6-class) | 214 | Glass type from chemical composition |
| 44 | `ecoli` | Classification (protein localization) | 336 | Multi-class, imbalanced |
| 45 | `yeast` | Classification (protein localization) | 1,484 | 10-class, imbalanced — good for class-imbalance practice |
| 46 | `car` | Classification (car acceptability) | 1,728 | All categorical features — great encoding practice |
| 47 | `zoo` | Classification (animal type) | 101 | Small, mostly binary features |
| 48 | `balance-scale` | Classification | 625 | Simple synthetic-like real dataset, 3-class |
| 49 | `tic-tac-toe` | Classification (win/loss) | 958 | All categorical, non-linear patterns |
| 50 | `vote` | Classification (party) | 435 | Congressional voting records, mostly binary features |
| 51 | `segment` | Classification (7-class) | 2,310 | Image segmentation |
| 52 | `vehicle` | Classification (4-class) | 846 | Silhouette-based vehicle type |
| 53 | `satimage` | Classification (6-class) | 6,430 | Satellite image land cover |
| 54 | `letter` | Classification (26-class) | 20,000 | Letter recognition — big multiclass practice |
| 55 | `abalone` | Regression (age) | 4,177 | Predict age from physical measurements |
| 56 | `kin8nm` | Regression | 8,192 | Robot arm simulation, non-linear |
| 57 | `pol` | Regression | 15,000 | Telecom pricing data |
| 58 | `wine-quality-red` | Regression/Classification (quality) | 1,599 | Ordinal quality score 0-10 |
| 59 | `wine-quality-white` | Regression/Classification (quality) | 4,898 | Larger sibling of red wine version |
| 60 | `phoneme` | Classification (binary) | 5,404 | Speech signal, nasal vs oral |
| 61 | `blood-transfusion-service-center` | Classification (donated) | 748 | Small, clean, binary |
| 62 | `banknote-authentication` | Classification (binary) | 1,372 | Forged vs genuine banknotes, from image wavelets |
| 63 | `seeds` | Classification (3-class) | 210 | Wheat kernel measurements |
| 64 | `climate-model-simulation-crashes` | Classification (binary) | 540 | Simulation crash prediction, imbalanced |
| 65 | `diabetes` | Classification (binary) | 768 | Pima Indians Diabetes — very commonly cited benchmark |

---

## 🌍 4. OpenML Real-World, Batch 2 (21)

Same load pattern: `fetch_openml(name="<name>", as_frame=True, parser="auto").frame`.

| # | Name | Task | Rows | Notes |
|---|---|---|---|---|
| 66 | `monks-problems-1` | Classification | 556 | Classic symbolic-learning benchmark |
| 67 | `hepatitis` | Classification | 155 | Medical, many missing values — good imputation practice |
| 68 | `heart-statlog` | Classification (heart disease) | 270 | Small, clean, widely used |
| 69 | `liver-disorders` | Classification/Regression | 345 | BUPA liver disorders |
| 70 | `haberman` | Classification (survival) | 306 | Breast cancer survival, imbalanced |
| 71 | `page-blocks` | Classification (5-class) | 5,473 | Document layout analysis, very imbalanced |
| 72 | `nursery` | Classification (5-class) | 12,960 | All categorical, large |
| 73 | `kr-vs-kp` | Classification (chess endgame) | 3,196 | King-Rook vs King-Pawn, all categorical |
| 74 | `anneal` | Classification | 898 | Steel annealing, many categorical + missing |
| 75 | `audiology` | Classification (many classes) | 226 | High missingness — good for topic 03 practice |
| 76 | `lymph` | Classification (4-class) | 148 | Small medical dataset |
| 77 | `primary-tumor` | Classification (many classes) | 339 | High missingness, high class count |
| 78 | `breast-w` | Classification (binary) | 699 | Breast Cancer Wisconsin (Original) — compare vs. sklearn's version |
| 79 | `thyroid-allbp` | Classification | 2,800 | Medical, many features |
| 80 | `cmc` | Classification (3-class) | 1,473 | Contraceptive method choice, socioeconomic features |
| 81 | `autos` | Regression (price) | 205 | Car price prediction, mixed types |
| 82 | `credit-approval` | Classification (binary) | 690 | Credit card approval, mixed types + missing |
| 83 | `hill-valley` | Classification (binary) | 1,212 | 100 numeric features from terrain shape |
| 84 | `madelon` | Classification (binary) | 2,600 | NIPS feature-selection benchmark, 500 mostly-noise features |
| 85 | `optdigits` | Classification (10-class) | 5,620 | Optical digit recognition |
| 86 | `pendigits` | Classification (10-class) | 10,992 | Pen-based digit recognition |

---

## 🧪 5. Synthetic Generators — `sklearn.datasets.make_*` (28)

Zero network dependency, fully reproducible with `random_state=`, and parameterized specifically to target each Classical ML topic ahead.

### Regression (7)

| # | Recipe | Load Code | Targets which topic |
|---|---|---|---|
| 87 | Linear, low noise | `make_regression(n_samples=200, n_features=5, noise=10, random_state=42)` | Linear Regression basics |
| 88 | Single feature (visualizable) | `make_regression(n_samples=200, n_features=1, noise=20, random_state=42)` | Simple/Polynomial Regression, plotting the fit line |
| 89 | Multi-output | `make_regression(n_samples=200, n_features=5, n_targets=2, random_state=42)` | Multi-output regression |
| 90 | High noise | `make_regression(n_samples=200, n_features=5, noise=50, random_state=42)` | Ridge/Lasso robustness comparison |
| 91 | Friedman #1 | `make_friedman1(n_samples=200, random_state=42)` | Non-linear regression (tree/boosting) |
| 92 | Friedman #2 | `make_friedman2(n_samples=200, random_state=42)` | Non-linear regression variant |
| 93 | Friedman #3 | `make_friedman3(n_samples=200, random_state=42)` | Non-linear regression variant |

### Classification (10)

| # | Recipe | Load Code | Targets which topic |
|---|---|---|---|
| 94 | Binary, balanced | `make_classification(n_samples=300, n_classes=2, random_state=42)` | Logistic Regression, SVM basics |
| 95 | Imbalanced (90/10) | `make_classification(n_samples=300, weights=[0.9, 0.1], random_state=42)` | Class-imbalance handling |
| 96 | Multiclass (4-class) | `make_classification(n_samples=300, n_classes=4, n_informative=4, random_state=42)` | Multiclass classifiers |
| 97 | 2D visualizable | `make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)` | Decision boundary plotting |
| 98 | Gaussian quantiles | `make_gaussian_quantiles(n_samples=300, random_state=42)` | Non-linear boundary practice |
| 99 | Hastie 10.2 | `make_hastie_10_2(n_samples=300, random_state=42)` | AdaBoost/Boosting benchmark (used in sklearn's own docs) |
| 100 | Moons | `make_moons(n_samples=300, noise=0.1, random_state=42)` | Kernel SVM, KNN non-linear boundaries |
| 101 | Circles | `make_circles(n_samples=300, noise=0.05, random_state=42)` | Kernel trick demonstration |
| 102 | Moons (low noise, for clustering) | `make_moons(n_samples=300, noise=0.05, random_state=42)` | DBSCAN non-convex cluster practice |
| 103 | Circles (for clustering) | `make_circles(n_samples=300, noise=0.03, factor=0.5, random_state=42)` | DBSCAN non-convex cluster practice |

### Clustering (4)

| # | Recipe | Load Code | Targets which topic |
|---|---|---|---|
| 104 | 3 well-separated blobs | `make_blobs(n_samples=300, centers=3, random_state=42)` | K-Means introduction |
| 105 | 5 overlapping blobs | `make_blobs(n_samples=300, centers=5, cluster_std=2.5, random_state=42)` | Gaussian Mixture Models |
| 106 | 2 simple blobs | `make_blobs(n_samples=300, centers=2, n_features=2, random_state=42)` | Simple 2D clustering visualization |
| 107 | Biclusters | `make_biclusters(shape=(100, 50), n_clusters=3, random_state=42)` | Advanced biclustering practice |

### Manifold / Dimensionality Reduction (3)

| # | Recipe | Load Code | Targets which topic |
|---|---|---|---|
| 108 | Swiss roll | `make_swiss_roll(n_samples=300, random_state=42)` | t-SNE/UMAP non-linear manifold demo |
| 109 | S-curve | `make_s_curve(n_samples=300, random_state=42)` | Manifold learning comparison |
| 110 | Low-rank matrix | `make_low_rank_matrix(n_samples=100, n_features=20, random_state=42)` | PCA with controllable effective rank |

### Math / Misc (4)

| # | Recipe | Load Code | Targets which topic |
|---|---|---|---|
| 111 | SPD matrix | `make_spd_matrix(n_dim=5, random_state=42)` | Covariance matrix practice (ties to Math Refresher) |
| 112 | Sparse uncorrelated | `make_sparse_uncorrelated(n_samples=200, random_state=42)` | Feature selection (only a few features matter) |
| 113 | Multilabel classification | `make_multilabel_classification(n_samples=200, n_classes=3, random_state=42)` | Multi-label practice (bonus, beyond this series' scope) |
| 114 | Checkerboard | `make_checkerboard(shape=(100, 50), n_clusters=3, random_state=42)` | Advanced biclustering variant |

---

## 🗺️ Which Dataset for Which Topic?

| Classical ML topic (coming next) | Recommended datasets |
|---|---|
| Linear/Polynomial Regression | #87, #88, #4 (Diabetes), #7 (California Housing) |
| Ridge/Lasso/ElasticNet | #90 (high noise), #112 (sparse/uncorrelated) |
| Logistic Regression | #94, #37 (Adult), #65 (Diabetes classification) |
| KNN (classification/regression) | #100 (Moons), #1 (Iris), #17 (Penguins) |
| Naive Bayes | #39 (Mushroom), #40 (Spambase) |
| Decision Trees / Random Forest | #46 (Car), #61 (Blood Transfusion), any classification set |
| SVM (kernel trick) | #101 (Circles), #100 (Moons) |
| Boosting (AdaBoost/XGBoost/etc.) | #99 (Hastie 10.2), #59 (Wine Quality) |
| K-Means / Hierarchical / DBSCAN | #104, #105, #102, #103, #31 (Geyser — the classic!) |
| PCA / Dimensionality Reduction | #110, #26 (Brain Networks), #8 (Digits) |
| t-SNE / UMAP | #108 (Swiss Roll), #109 (S-Curve) |
| Association Rules (Apriori) | #46 (Car), #72 (Nursery) — categorical-heavy sets work best |
| Model Evaluation & Tuning | Any dataset above — practice CV/tuning on a few different ones |

---

## ⚠️ Caveats & Notes

1. **OpenML names can occasionally have multiple versions.** If `fetch_openml(name=...)` ever raises a version-ambiguity error, add `version=1` (or check [openml.org](https://www.openml.org) to search for the current canonical name/version).
2. **All 114 entries in this catalog were live-verified while building it** — every name/function call above was actually executed successfully. If one ever breaks in the future (OpenML dataset removed/renamed), search openml.org for a replacement.
3. **Large datasets** (Covertype, RCV1, Adult, Diamonds, Letter) are worth downsampling (`df.sample(n=..., random_state=42)`) for quick iteration during learning — full-size only when you specifically want to practice at scale.
4. **First download requires internet**; every subsequent load is served from local cache (`~/scikit_learn_data/` for sklearn/OpenML, `~/seaborn-data/` for seaborn) — no repeated network dependency.
5. **Licensing**: all datasets here are long-established public research/benchmark datasets distributed through official library APIs. If you redistribute a *derived* file from any of them, check the original dataset's license/citation requirements first (most require attribution only).

---
[← Back to Foundation](../README.md)
