# 📊 Data Visualization — Matplotlib & Seaborn

> Status: ✅ Complete — [Open the notebook →](02_data_visualization.ipynb)

Numbers hide patterns that eyes catch instantly. Before fitting any model, a good ML practitioner *looks* at the data — distributions, outliers, correlations, class balance — because every chart here doubles as a diagnostic tool used again in later Classical ML notebooks.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Matplotlib vs Seaborn — When to Use Which](#-matplotlib-vs-seaborn--when-to-use-which)
5. [Chart Type Reference](#-chart-type-reference)
6. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
7. [Function Reference](#-function-reference-used-in-the-notebook)
8. [Self-Test Exercises](#-self-test-exercises)
9. [Notebook](#-notebook)
10. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

- **Matplotlib** is the low-level plotting engine — it gives you a `Figure` (the canvas) and one or more `Axes` (individual plots inside it), and you build charts by calling methods on the `Axes` object (`ax.plot()`, `ax.hist()`, `ax.scatter()`...).
- **Seaborn** is built *on top of* Matplotlib. It understands Pandas DataFrames natively (`data=df, x="col1", y="col2"`) and adds statistical layers — density curves, confidence intervals, automatic aggregation — that would otherwise take many lines of raw Matplotlib code.

Rule of thumb: reach for **Seaborn first** for anything statistical (distributions, categorical comparisons, correlation), and drop down to **Matplotlib** when you need fine-grained control (custom annotations, unusual layouts, precise styling) — Seaborn plots return Matplotlib `Axes` objects, so you can always mix both.

---

## 🎯 Why This Topic Matters

Every algorithm notebook later in this series depends on visual EDA that starts here:

- **Regression notebooks** — scatter plots to check linearity assumptions, residual plots to check error patterns
- **Ridge/Lasso** — correlation heatmaps to spot multicollinearity *before* it destabilizes coefficients
- **Classification notebooks** — class balance bar charts, decision boundary plots
- **Preprocessing (topic 03)** — boxplots to flag outliers, histograms to detect skew that needs a transform
- **Clustering (Unsupervised)** — pairplots and scatter plots to sanity-check whether clusters are visually separable at all

Skipping visualization means debugging a model's poor performance blind, when a five-second histogram would have shown the problem (a skewed feature, a leaked outlier, a near-perfectly-correlated pair of columns).

---

## 🧮 Mathematical Explanation

### 1. Histograms — binning and frequency

A histogram partitions the range of a variable $x$ into $k$ equal-width bins and counts how many observations fall in each bin $B_j$:

$$\text{count}(B_j) = \big|\{ x_i \mid x_i \in B_j \}\big|$$

The bin width $h$ directly trades off noise vs. detail — too few bins hides structure (over-smoothing), too many bins shows noise as if it were signal (over-fitting the display). A common heuristic (Sturges' rule) is:

$$k = \lceil \log_2(n) + 1 \rceil$$

### 2. Kernel Density Estimation (KDE) — the smooth curve on histograms

The KDE curve seen overlaid on `sns.histplot(..., kde=True)` estimates the underlying probability density function by placing a small smooth "bump" (kernel) at every data point and summing them:

$$\hat{f}(x) = \frac{1}{nh}\sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where $K$ is typically a Gaussian kernel, $n$ is the sample count, and $h$ (the *bandwidth*) controls smoothness — small $h$ overfits to individual points (spiky), large $h$ oversmooths (blurs away real structure). This is conceptually identical to the bias-variance tradeoff you'll see again in the Model Evaluation & Tuning notebooks.

### 3. Pearson correlation coefficient — what the heatmap shows

For two features $X, Y$ with means $\bar{X}, \bar{Y}$, the Pearson correlation is:

$$r_{X,Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2}\sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

$r \in [-1, 1]$: $+1$ is perfect positive linear relationship, $-1$ perfect negative, $0$ no linear relationship (note: **no linear relationship** — Pearson $r$ can be near 0 even for a strong *non-linear* relationship, which is exactly why you should always look at the scatter plot too, not just the number). The full correlation matrix `df.corr()` computes this pairwise for every column combination — that matrix is exactly what `sns.heatmap` visualizes.

### 4. Boxplot anatomy — the IQR outlier rule

A boxplot summarizes a distribution via five numbers based on quartiles:

$$Q_1 = \text{25th percentile}, \quad Q_2 = \text{median (50th percentile)}, \quad Q_3 = \text{75th percentile}$$
$$\text{IQR} = Q_3 - Q_1$$

The box spans $[Q_1, Q_3]$, the line inside is $Q_2$, and the whiskers extend to the most extreme data point still inside the **fence**:

$$\text{lower fence} = Q_1 - 1.5 \times \text{IQR}, \qquad \text{upper fence} = Q_3 + 1.5 \times \text{IQR}$$

Any point beyond the fences is drawn as an individual dot — a flagged **outlier**. The `1.5` multiplier is a convention (Tukey's rule), not a law of nature; some domains use `3.0` for a more conservative ("extreme outliers only") threshold.

### 5. Pairplot — the covariance matrix, visualized

A pairplot's off-diagonal scatter plots are, collectively, a visual expansion of the covariance matrix $\Sigma$ where $\Sigma_{jk} = \text{Cov}(X_j, X_k)$ — the same matrix that PCA (covered in the Unsupervised notebooks) decomposes to find directions of maximum variance. Looking at a pairplot before running PCA is a genuinely useful sanity check on what PCA is about to do mathematically.

---

## ⚖️ Matplotlib vs Seaborn — When to Use Which

| Need | Use |
|---|---|
| Full manual control over every element | Matplotlib (`ax.plot`, `ax.annotate`, custom ticks) |
| Quick statistical plot from a DataFrame | Seaborn (`sns.histplot`, `sns.boxplot`, `sns.heatmap`) |
| Multiple subplots in a custom grid | Matplotlib `plt.subplots(rows, cols)` |
| Every pairwise feature relationship at once | Seaborn `sns.pairplot` |
| Publication-quality styling with minimal code | Seaborn (`sns.set_theme()`) |
| Custom color per data point based on a condition | Matplotlib scatter with a manual color array (see notebook §3) |

---

## 📐 Chart Type Reference

| Chart | Best for | Function |
|---|---|---|
| Line plot | Trends over a continuous/ordered axis (time, sequence) | `ax.plot()` |
| Histogram | Distribution shape of one variable | `ax.hist()` / `sns.histplot()` |
| KDE plot | Smoothed distribution, easier to compare multiple groups | `sns.kdeplot()` |
| Scatter plot | Relationship between two continuous variables | `ax.scatter()` / `sns.scatterplot()` |
| Bar plot | Comparing a metric across categories | `ax.bar()` / `sns.barplot()` |
| Boxplot | Median, spread, and outliers per group | `sns.boxplot()` |
| Violin plot | Boxplot + full distribution shape combined | `sns.violinplot()` |
| Heatmap | Matrix data (e.g. correlation matrix) | `sns.heatmap()` |
| Pairplot | Every pairwise relationship + per-feature distribution | `sns.pairplot()` |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting `%matplotlib inline` / not calling `plt.show()`** in scripts (vs. notebooks) — figures may not render or may pile up silently across cells.
2. **Reusing `fig, ax` across cells** — each `plt.subplots()` call creates a new figure; reusing a stale `ax` variable from an earlier cell draws onto a figure that's already been shown/closed.
3. **Misreading `axis` limits** — auto-scaled axes can make two totally different-scale plots look visually similar; always check the tick labels, not just the shape of the curve.
4. **Correlation ≠ causation, and Pearson ≠ "any relationship"** — a Pearson $r \approx 0$ does not mean "no relationship," only "no *linear* relationship" (see the math section above). Always eyeball the scatter plot.
5. **`palette` without `hue` is deprecated in newer Seaborn** — pass `hue=<same column as x>, legend=False` if you want per-category colors without a separate hue variable (this bug was hit and fixed while building this notebook).
6. **Overplotting** — with thousands of points, scatter plots become a solid blob; use `alpha=` transparency, hexbin plots, or sampling to keep them readable.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `plt.subplots(rows, cols, figsize=)` | Create Figure + Axes grid |
| `ax.plot()`, `ax.scatter()`, `ax.bar()`, `ax.hist()` | Core Matplotlib chart types |
| `ax.set_title()`, `.set_xlabel()`, `.set_ylabel()`, `.legend()` | Labeling |
| `plt.tight_layout()` | Auto-fix overlapping subplot spacing |
| `sns.set_theme()` | Apply Seaborn's default styling globally |
| `sns.histplot(data, kde=True)` | Histogram with optional KDE overlay |
| `sns.boxplot(data, x=, y=, hue=)` | Quartile/outlier summary per group |
| `sns.heatmap(matrix, annot=True, cmap=)` | Matrix/correlation visualization |
| `sns.pairplot(df, vars=, hue=)` | Grid of every pairwise feature relationship |
| `df.corr()` | Pearson correlation matrix |
| `series.quantile([0.25, 0.75])` | Manual $Q_1$/$Q_3$ computation for IQR |

---

## 📝 Self-Test Exercises

1. Given a right-skewed feature (like the exponential distribution in the notebook), predict *before running the code* whether its mean is greater or less than its median — then verify with `sns.histplot(kde=True)`.
2. Compute the IQR outlier fences for a column by hand using `.quantile()`, then confirm your fences match what `sns.boxplot()` visually flags.
3. Build a correlation heatmap for a 4-feature synthetic DataFrame where one feature is a near-exact copy of another (add tiny noise) — confirm the heatmap shows $r \approx 1$.
4. Create a 2×2 subplot grid comparing the same feature across 4 different `bins=` values for a histogram — observe the over-smoothing vs over-fitting tradeoff described in the KDE section.
5. Explain, without looking back, why a pairplot's diagonal is different from its off-diagonal (what plot type does each show, and why).

---

## 📓 Notebook

Hands-on implementation of every chart type above, executed end-to-end on a synthetic dataset resembling real tabular ML data (age, income, spending score), including a deliberate outlier-injection example to demonstrate the IQR rule visually:

➡️ **[02_data_visualization.ipynb](02_data_visualization.ipynb)**

---

## 📚 Further Reading

- [Matplotlib official documentation](https://matplotlib.org/stable/index.html)
- [Seaborn official documentation](https://seaborn.pydata.org/)
- [Matplotlib Anatomy of a Figure](https://matplotlib.org/stable/gallery/showcase/anatomy.html)
- [Seaborn: statistical estimation and error bars](https://seaborn.pydata.org/tutorial/error_bars.html)

---
[← Back to Foundation](../README.md)
