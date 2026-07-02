# 🧹 Data Preprocessing & EDA

> Status: ✅ Complete — [Open the notebook →](03_data_preprocessing_eda.ipynb)

"Garbage in, garbage out." No algorithm — not even the fanciest gradient-boosted ensemble covered later in Classical ML — can produce a good model from a dirty dataset. This topic builds the full preprocessing toolkit: missing values, duplicates, categorical encoding, outlier treatment, and feature scaling.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Imputation Strategy Comparison](#-imputation-strategy-comparison)
5. [Encoding Strategy Comparison](#-encoding-strategy-comparison)
6. [Scaler Comparison](#-scaler-comparison)
7. [The Preprocessing Order Matters](#-the-preprocessing-order-matters)
8. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
9. [Function Reference](#-function-reference-used-in-the-notebook)
10. [Self-Test Exercises](#-self-test-exercises)
11. [Notebook](#-notebook)
12. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Real datasets are never clean. Preprocessing turns raw, messy data into a numeric matrix $X \in \mathbb{R}^{n \times d}$ that an algorithm can actually consume, through five recurring problems:

| Problem | Symptom | Fix |
|---|---|---|
| Missing values | `NaN` in cells | Imputation (mean/median/KNN) |
| Duplicate rows | Identical records | `.drop_duplicates()` |
| Categorical text | Strings instead of numbers | Encoding (one-hot/label) |
| Outliers | Extreme, implausible values | Capping/winsorizing, or removal |
| Unscaled features | Wildly different ranges (age: 0–100, income: 0–1,000,000) | Scaling (Standard/MinMax/Robust) |

---

## 🎯 Why This Topic Matters

This is the connective tissue between EDA (topic 02) and every algorithm notebook that follows:

- **Distance-based algorithms** (KNN, SVM, K-Means) are *directly* distorted by unscaled features — a feature ranging 0–1,000,000 dominates the distance calculation over one ranging 0–1, regardless of true importance.
- **Linear models** (Linear/Logistic Regression) converge faster and produce more stable, comparable coefficients on scaled features.
- **Tree-based models** (Decision Trees, Random Forest, Gradient Boosting) are scale-invariant, but still need missing values handled and categories encoded — they cannot split on `NaN` or a raw string.
- **Every model** silently learns the wrong thing from leaked duplicate rows across a train/test split, and gets pulled by unaddressed outliers during training (especially anything optimized by minimizing squared error, which punishes large deviations quadratically).

---

## 🧮 Mathematical Explanation

### 1. Types of missingness

Statistically, missing data falls into three categories that determine whether imputation is safe:

- **MCAR (Missing Completely At Random)** — missingness is unrelated to any variable, observed or not. Safest case; mean/median imputation introduces minimal bias.
- **MAR (Missing At Random)** — missingness depends on *other observed* variables (e.g., income is more often missing for younger respondents). KNN or model-based imputation (using those other variables) works well here.
- **MNAR (Missing Not At Random)** — missingness depends on the *unobserved value itself* (e.g., very high earners refuse to report income). No imputation method fully fixes this without external information — worth explicitly flagging in a real project rather than silently imputing.

### 2. Imputation formulas

**Mean imputation:**
$$\hat{x}_{\text{missing}} = \bar{x} = \frac{1}{n_{\text{observed}}}\sum_{i: x_i \text{ observed}} x_i$$

Sensitive to outliers (a single extreme value pulls $\bar{x}$) and **artificially shrinks variance**, since every imputed point lands exactly on the mean.

**Median imputation:** replaces $\bar{x}$ with the 50th percentile — robust to outliers, since the median only depends on rank order, not magnitude.

**KNN imputation:** for a row with a missing feature $j$, find the $k$ nearest rows (by distance on the *other* features) and impute:

$$\hat{x}_{i,j} = \frac{1}{k}\sum_{r \in \text{kNN}(i)} x_{r,j}$$

More accurate than mean/median when features correlate with each other (uses that correlation structure), at the cost of computing pairwise distances for every imputation.

### 3. One-Hot Encoding as basis vectors

For a categorical feature with $k$ unique values, one-hot encoding maps each category to a **standard basis vector** in $\mathbb{R}^k$:

$$\text{Dhaka} \to (1,0,0,0), \quad \text{Chittagong} \to (0,1,0,0), \quad \text{Khulna} \to (0,0,1,0), \quad \text{Rajshahi} \to (0,0,0,1)$$

This guarantees **equal distance** between every pair of categories ($\sqrt{2}$ in Euclidean distance between any two basis vectors), which is exactly the "no false ordering" property that makes it safe for nominal (unordered) categories. Label encoding, by contrast, assigns $\{0, 1, 2, 3\}$ — a linear model would then treat `Rajshahi` (3) as "three times" `Dhaka` (... wait, depends on mapping, but generally as being further from `Dhaka` than `Chittagong` is, which is a meaningless artifact of encoding order).

### 4. Feature scaling formulas

**Standardization (Z-score, StandardScaler):**
$$x' = \frac{x - \mu}{\sigma}, \qquad \mu = \frac{1}{n}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{n}\sum_i (x_i - \mu)^2}$$
Result: mean 0, standard deviation 1. Still sensitive to outliers (both $\mu$ and $\sigma$ shift under extreme values).

**Min-Max Normalization (MinMaxScaler):**
$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
Result: bounded exactly to $[0, 1]$. Highly sensitive to outliers — a single extreme value stretches the denominator and compresses every other point toward 0.

**Robust Scaling (RobustScaler):**
$$x' = \frac{x - \text{median}(x)}{\text{IQR}(x)} = \frac{x - Q_2}{Q_3 - Q_1}$$
Uses the same $Q_1$/$Q_3$ quartiles as topic 02's boxplot — by construction, resistant to outliers since neither the median nor the IQR shifts much when a few extreme points are added.

### 5. Z-score outlier rule

A point is flagged as an outlier when its standardized value exceeds a threshold (commonly 3, matching the "3-sigma rule" — under a normal distribution, $P(|Z| > 3) \approx 0.0027$, i.e. it should almost never happen by chance):

$$\left|\frac{x_i - \mu}{\sigma}\right| > 3 \implies x_i \text{ is an outlier}$$

This is the numeric equivalent of topic 02's visual IQR/boxplot rule — both aim to answer "is this point implausibly far from the bulk of the data?", using different reference statistics (mean/std vs. median/quartiles).

---

## 🔬 Imputation Strategy Comparison

| Strategy | Robust to outliers? | Preserves variance? | Cost | Best when |
|---|---|---|---|---|
| Mean | ❌ | ❌ (shrinks it) | $O(n)$ | Data is roughly symmetric, no outliers |
| Median | ✅ | ❌ (shrinks it) | $O(n \log n)$ | Skewed or outlier-prone features |
| KNN | ✅ (depends on $k$) | ✅ (best) | $O(n^2)$ | Features correlate with each other, dataset not huge |
| Drop rows | N/A | ✅ | $O(n)$ | Missingness is rare (<5%) and MCAR |

---

## 🔤 Encoding Strategy Comparison

| Encoding | Assumes order? | Output dimensions | Best for |
|---|---|---|---|
| Label Encoding | ✅ (implicitly) | 1 column | Truly **ordinal** categories (e.g. `Low < Medium < High`) |
| One-Hot Encoding | ❌ | $k$ columns | **Nominal** (unordered) categories, most common case |
| Ordinal Encoding (manual mapping) | ✅ (explicit) | 1 column | Ordinal categories where you control the order mapping |

---

## ⚖️ Scaler Comparison

| Scaler | Formula basis | Output range | Outlier sensitivity |
|---|---|---|---|
| StandardScaler | mean, std | Unbounded, centered at 0 | Medium |
| MinMaxScaler | min, max | Exactly $[0, 1]$ | High |
| RobustScaler | median, IQR | Unbounded, centered at 0 | Low |

---

## 🔢 The Preprocessing Order Matters

A common beginner mistake is doing these steps in an order that lets information leak or compounds errors. The order used in this notebook — and generally recommended — is:

1. **Audit** (missingness map, duplicate count, dtype check) — understand the damage first
2. **Remove duplicates** — before any statistic is computed from the data
3. **Impute missing values** — so later steps have complete data to work with
4. **Treat outliers** — capping *after* imputation, since imputed values can occasionally land near existing outliers
5. **Encode categoricals**
6. **Scale numeric features** — always **last**, and in a real pipeline, fit the scaler only on the *training* split to avoid leaking test-set statistics into training

---

## ⚠️ Common Pitfalls & Gotchas

1. **Imputing before splitting train/test** in a real project leaks test-set statistics (e.g. the test set's mean) into the training data — always fit imputers/scalers on the training split only, then apply (`transform`, not `fit_transform`) to the test split. This notebook works on a single dataset for teaching clarity, but production code must split first.
2. **Label-encoding a nominal feature** — see the math section; this silently invents a false ordinal relationship that damages linear/distance-based models.
3. **Dropping instead of capping outliers** by default — you lose sample size and may bias the dataset if outliers aren't actually errors (e.g., a genuinely high earner isn't a "mistake").
4. **Scaling before encoding** — scaling formulas assume continuous numeric meaning; running them on soon-to-be-encoded categorical codes doesn't make sense and should be avoided by scaling only true numeric columns.
5. **Forgetting duplicates entirely** — `.dropna()` and `.fillna()` get all the attention, but duplicate rows are just as common (especially in scraped or merged datasets) and just as damaging.
6. **One-hot encoding high-cardinality columns** (e.g., a "city" column with 500 unique values) creates 500 new columns — consider target encoding or grouping rare categories into "Other" instead.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `df.info()`, `df.isna().sum()` | Missingness audit |
| `df.duplicated()`, `df.drop_duplicates()` | Duplicate detection & removal |
| `df.fillna(value)` | Mean/median imputation |
| `sklearn.impute.KNNImputer` | Similarity-based imputation |
| `series.clip(lower=, upper=)` | Winsorizing / capping outliers |
| `sklearn.preprocessing.LabelEncoder` | Ordinal integer encoding |
| `pd.get_dummies()` / `sklearn.preprocessing.OneHotEncoder` | One-hot encoding |
| `sklearn.preprocessing.StandardScaler` | Z-score standardization |
| `sklearn.preprocessing.MinMaxScaler` | Min-max normalization |
| `sklearn.preprocessing.RobustScaler` | Median/IQR-based robust scaling |
| `series.mode()` | Categorical missing-value imputation |
| `series.value_counts(normalize=True)` | Cardinality check, rare-category detection |
| `sklearn.utils.class_weight.compute_class_weight` | Class-imbalance-aware training weights |
| `sklearn.model_selection.train_test_split(stratify=)` | Class-proportion-preserving split |
| `sklearn.feature_selection.VarianceThreshold` | Near-constant feature removal |
| `sklearn.compose.ColumnTransformer` + `sklearn.pipeline.Pipeline` | Leakage-safe, one-call preprocessing chain |

---

## 📝 Self-Test Exercises

1. For a feature that is right-skewed with a few extreme high values, predict which imputation strategy (mean vs median) will produce a more representative "typical" value — then verify on the notebook's `income` column.
2. Explain why one-hot encoding a binary category (e.g., "Yes"/"No") technically only needs **one** column, not two — what redundant information does the second column carry?
3. Compute the Z-score outlier threshold manually for a feature, and compare which points it flags versus the IQR rule from topic 02 — do they agree?
4. Given a dataset with a `rating` column of `{Poor, Average, Good, Excellent}`, decide whether Label Encoding or One-Hot Encoding is more appropriate, and justify using the ordinal vs nominal distinction.
5. Explain, in your own words, why fitting a `StandardScaler` on the *entire* dataset (train + test combined) before splitting is a form of data leakage.

---

## 📓 Notebook

Full pipeline on a deliberately messy synthetic dataset: missingness map, duplicate removal, three imputation strategies compared side-by-side, Z-score outlier detection with before/after capping visualization, categorical encoding, and a four-way scaler comparison via boxplots:

➡️ **[03_data_preprocessing_eda.ipynb](03_data_preprocessing_eda.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)

---
[← Back to Foundation](../README.md)
