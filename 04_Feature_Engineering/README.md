# 🛠️ Feature Engineering

> Status: ✅ Complete — [Open the notebook →](04_feature_engineering.ipynb)

Preprocessing (topic 03) makes data *usable*. Feature engineering makes it *predictive* — creating, transforming, and selecting the inputs that actually help a model learn. It's widely considered the single highest-leverage skill in classical ML: a well-engineered feature set often beats a fancier algorithm trained on mediocre features.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Feature Creation Techniques](#-feature-creation-techniques)
5. [Feature Selection Method Comparison](#-feature-selection-method-comparison)
6. [⚠️ Target Leakage — A Real Mistake Caught While Building This Notebook](#️-target-leakage--a-real-mistake-caught-while-building-this-notebook)
7. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
8. [Function Reference](#-function-reference-used-in-the-notebook)
9. [Self-Test Exercises](#-self-test-exercises)
10. [Notebook](#-notebook)
11. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Feature engineering splits into three activities, all covered in this topic:

| Activity | Question it answers | Example |
|---|---|---|
| **Creation** | What new signal can I derive from what I already have? | date → month/weekday, price × quantity |
| **Transformation** | How do I reshape a feature's distribution to help the model? | log transform, binning |
| **Selection** | Which features are actually worth keeping? | correlation filter, model importance |

---

## 🎯 Why This Topic Matters

This is the direct producer of the feature matrix $X$ that every algorithm in the Classical ML repo consumes:

- **Linear models** (Regression) benefit enormously from transformed, symmetric features and explicit interaction/polynomial terms — a linear model literally cannot learn a curved relationship unless you hand it a squared or interaction feature to work with.
- **Tree-based models** don't need transforms for skew, but still benefit from well-chosen ratio/interaction features that shortcut what would otherwise take many splits to approximate.
- **Any model** trained on redundant or leaked features will look great in development and fail in production — this topic's selection methods, and its leakage warning, exist specifically to prevent that.

---

## 🧮 Mathematical Explanation

### 1. Polynomial & interaction features

Given base features $x_1, x_2$, `PolynomialFeatures(degree=2)` generates the full degree-2 expansion:

$$\phi(x_1, x_2) = \big[\, x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2 \,\big]$$

This lets a strictly *linear* model (like Linear Regression, covered next in Classical ML) fit a *curved* decision surface in the original feature space — because the model is still linear **in the expanded features**, even though the resulting prediction surface, plotted against the original $x_1, x_2$, is not. The number of generated features grows combinatorially: for $d$ base features at degree $k$, the count is $\binom{d+k}{k} - 1$, which is why polynomial expansion is used sparingly (usually degree 2, on a handful of features) — it explodes dimensionality fast.

### 2. Log transform

For a strictly positive, right-skewed feature $x$:

$$x' = \log(1 + x)$$

(the `+1` — `log1p` — avoids $\log(0) = -\infty$ when $x$ can be exactly 0). Logarithms compress large values much more than small ones ($\log$ grows sub-linearly), which pulls in the long right tail of a skewed distribution and makes it closer to symmetric — directly improving the normality assumptions behind linear regression's error term and speeding gradient-descent convergence for any gradient-based model.

### 3. Box-Cox / Yeo-Johnson power transform

Box-Cox generalizes the log transform into a family indexed by a parameter $\lambda$, chosen automatically to maximize normality:

$$x'(\lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \lambda \neq 0 \\ \ln(x) & \lambda = 0 \end{cases}$$

Box-Cox requires $x > 0$ strictly. **Yeo-Johnson** (used in the notebook via `PowerTransformer`) extends the same idea to handle zero and negative values, using a piecewise formula split on the sign of $x$ — this is why Yeo-Johnson, not Box-Cox, is the safer default when you're not certain every value is strictly positive.

### 4. Variance threshold

A feature with variance near 0 is (near-)constant and therefore carries almost no information for separating outcomes:

$$\text{Var}(x_j) = \frac{1}{n}\sum_{i=1}^n (x_{ij} - \bar{x}_j)^2 \quad \implies \quad \text{drop } x_j \text{ if } \text{Var}(x_j) < \tau$$

for some small threshold $\tau$. This is the cheapest possible filter — pure computation, no relationship to the target needed at all.

### 5. Mutual information

Where Pearson correlation (topic 02) only detects *linear* relationships, mutual information $I(X; Y)$ measures **any** statistical dependency, linear or not, by comparing the joint distribution to the product of marginals:

$$I(X;Y) = \sum_{x}\sum_{y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$

$I(X;Y) = 0$ only when $X$ and $Y$ are fully statistically independent. This is why a feature with near-zero Pearson correlation but high mutual information is a signal worth investigating — it usually means a real but non-linear relationship (e.g., a U-shaped one) that a linear correlation coefficient is blind to.

### 6. Model-based (embedded) importance

A Random Forest's feature importance (`.feature_importances_`) is computed from how much each feature reduces impurity (Gini or variance, depending on task) summed across every split, in every tree, weighted by how many samples reach that split — full derivation in the Classical ML → Random Forest notebook. The takeaway for this topic: it's a **model-aware** selection signal, complementary to the **model-agnostic** filter methods above, and the two can disagree — which is informative in itself.

---

## 🌱 Feature Creation Techniques

| Technique | Input | Output | Example from notebook |
|---|---|---|---|
| Date decomposition | 1 datetime column | Several categorical/numeric columns | `order_date` → month, day-of-week, is_weekend |
| Ratio feature | 2 numeric columns | 1 numeric column | `unit_price / delivery_days` |
| Manual interaction term | 2+ numeric columns | 1 numeric column | `customer_age * quantity` |
| Automatic polynomial expansion | $d$ numeric columns | up to $\binom{d+k}{k}-1$ columns | `PolynomialFeatures(degree=2)` |
| Binning/discretization | 1 continuous column | 1 categorical column | `customer_age` → `age_group` |

---

## 🔎 Feature Selection Method Comparison

| Method | Type | Considers target? | Catches non-linear relationships? | Cost |
|---|---|---|---|---|
| Variance Threshold | Filter | ❌ | N/A | Very low |
| Correlation filter | Filter | ✅ (pairwise, or vs. target) | ❌ (linear only) | Low |
| Mutual Information | Filter | ✅ | ✅ | Medium |
| Model-based importance (RF, Lasso) | Embedded | ✅ | ✅ (tree-based) / ❌ (Lasso, linear) | Medium–High (requires training) |
| Recursive Feature Elimination (RFE) | Wrapper | ✅ | Depends on base model | High (retrains repeatedly) |

---

## ⚠️ Target Leakage — A Real Mistake Caught While Building This Notebook

While building this notebook, the first draft created a ratio feature as `total_spend / delivery_days` — where `total_spend` was the **target** being predicted. Because that feature is an algebraic function of the answer itself, it scored artificially high in every importance metric without representing real, generalizable signal (in production, you would not have `total_spend` available at prediction time to compute it from). It was caught and rewritten as `unit_price / delivery_days`, built only from other **input** columns.

**General rule:** before trusting any newly engineered feature's importance score, ask *"could I have computed this feature without already knowing the target?"* If the answer is no, it's leakage, and the feature must be dropped or rebuilt — regardless of how good it makes the model look.

---

## ⚠️ Common Pitfalls & Gotchas

1. **Target leakage** — see the dedicated section above; the single most damaging and most common feature engineering mistake.
2. **Over-expanding with `PolynomialFeatures`** on many columns at high degree causes combinatorial explosion and severe overfitting risk — apply it to a small, deliberately chosen subset.
3. **Binning throws away information** — `age_group` can never distinguish a 26-year-old from a 34-year-old anymore; only bin when the simplification is worth the loss (interpretability, non-linear modeling need).
4. **Filter methods evaluated on the full dataset before a train/test split** can indirectly leak test-set structure into feature selection decisions — in a real project, select features using training data only.
5. **Trusting a single selection method** — variance/correlation/mutual-information/model-importance can disagree; use them as complementary evidence, not a single verdict.
6. **Forgetting that Yeo-Johnson's $\lambda$ is fit on training data** — like scalers, `PowerTransformer` must be fit only on the training split, then applied (not re-fit) to the test split.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `pd.to_datetime()`, `.dt.month`, `.dt.dayofweek` | Date decomposition |
| `sklearn.preprocessing.PolynomialFeatures` | Automatic polynomial/interaction expansion |
| `np.log1p()` | Safe log transform |
| `sklearn.preprocessing.PowerTransformer` | Box-Cox / Yeo-Johnson transform |
| `pd.cut()` | Binning/discretization |
| `sklearn.feature_selection.VarianceThreshold` | Near-constant feature filter |
| `df.corr()` | Correlation-based redundancy filter |
| `sklearn.feature_selection.mutual_info_regression` | Non-linear dependency filter |
| `sklearn.ensemble.RandomForestRegressor.feature_importances_` | Embedded, model-based importance |

---

## 📝 Self-Test Exercises

1. For a feature you engineer as `A / B`, write out the leakage check question from this README and apply it — is either `A` or `B` the target, or derived from it?
2. Explain why `PolynomialFeatures(degree=2)` on 10 input columns produces far more than 10 or even 20 output columns — compute the count using the combinatorial formula above.
3. Given a feature with Pearson correlation ≈ 0.02 but mutual information significantly above 0, sketch what kind of scatter plot (shape) could produce that combination.
4. Explain why `VarianceThreshold` needs no information about the target at all, while `mutual_info_regression` does.
5. Pick one raw column from the notebook's dataset and propose two *new* engineered features from it that weren't built in the notebook — justify why each might help a model.

---

## 📓 Notebook

Full pipeline on a synthetic e-commerce dataset: date decomposition, ratio/interaction feature creation (including the leakage mistake, caught and fixed live), polynomial expansion, log vs. Yeo-Johnson skew comparison with before/after histograms, binning, and a three-way feature selection comparison (variance, mutual information, Random Forest importance):

➡️ **[04_feature_engineering.ipynb](04_feature_engineering.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [scikit-learn: Non-linear transformation (Box-Cox/Yeo-Johnson)](https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation)
- [scikit-learn: Polynomial features](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)

---
[← Back to Foundation](../README.md)
