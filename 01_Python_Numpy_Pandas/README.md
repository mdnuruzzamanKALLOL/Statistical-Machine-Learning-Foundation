# 🧩 Python, NumPy & Pandas

> Status: ✅ Complete — [Open the notebook →](01_python_numpy_pandas.ipynb)

Every ML pipeline — from a two-line `LinearRegression().fit()` to a multi-GPU transformer — ultimately reduces to numbers moving through arrays. This topic is the numerical + tabular data foundation that every other notebook in this series (Regression, Classification, Clustering, Model Tuning...) assumes you already have.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Why NumPy Is Fast — Internals](#️-why-numpy-is-fast--internals)
5. [dtype Reference](#-dtype-reference)
6. [Indexing & Slicing Cheat Sheet](#-indexing--slicing-cheat-sheet)
7. [`loc` vs `iloc` — Detailed Comparison](#-loc-vs-iloc--detailed-comparison)
8. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
9. [Computational Complexity](#-computational-complexity)
10. [Function Reference](#-function-reference-used-in-the-notebook)
11. [Self-Test Exercises](#-self-test-exercises)
12. [Notebook](#-notebook)
13. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

- **NumPy (`ndarray`)** stores numbers in one contiguous block of memory with a single fixed type (`dtype`). This is what makes it fast, and it's exactly what a "tensor" in PyTorch/TensorFlow is under the hood — `torch.Tensor` and `np.ndarray` share almost the same mental model and even interoperate via zero-copy conversion (`torch.from_numpy`).
- **Pandas (`Series` / `DataFrame`)** wraps NumPy arrays with **labels** (row index + column names), turning raw numeric arrays into something you can filter, group, and reason about the way you'd reason about a spreadsheet or SQL table.

Think of the relationship as layers, each one adding a capability the previous layer lacked:

```
Python list        →   NumPy ndarray         →   Pandas Series/DataFrame
(objects,               (contiguous,               (ndarray + labels,
 flexible types,         single dtype,               tabular ops,
 slow loops)             vectorized,                 groupby, joins,
                         fast)                        missing-data tools)
```

A DataFrame column, under the hood, literally **is** a NumPy array plus an index:

```python
df["math"].values   # -> numpy.ndarray([88, 92, 79, 95])
df["math"].index     # -> RangeIndex(0, 1, 2, 3) — the "labels"
```

---

## 🎯 Why This Topic Matters

Skipping this topic is the #1 reason beginners get stuck later in the series. Concretely, every algorithm notebook that follows will:

- Receive data as a NumPy feature matrix `X` and label vector `y` (`X.shape == (n_samples, n_features)`)
- Standardize/normalize `X` using broadcasting (`(X - mean) / std`)
- Compute predictions via matrix multiplication (`X @ weights`, or `model.predict(X)` internally doing the same)
- Explore/clean the raw dataset with Pandas **before** it ever becomes a NumPy array for the model

If `axis`, broadcasting, and `loc`/`iloc` aren't automatic to you yet, every future notebook will feel harder than it needs to be. This is the single highest-leverage topic in the whole Foundation category.

---

## 🧮 Mathematical Explanation

### 1. Arrays as vectors and matrices

A 1D NumPy array is a vector $\mathbf{x} \in \mathbb{R}^n$. A 2D array is a matrix $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features) — the standard shape of a **feature matrix** in every ML algorithm covered later in this series. A 3D+ array is what deep learning calls a **tensor**, e.g. `(batch, height, width)` for a batch of grayscale images.

### 2. Vectorization

For element-wise operations between $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{c}_i = \mathbf{a}_i \circ \mathbf{b}_i \quad \forall i \in \{1, \dots, n\}$$

where $\circ$ is any operator (`+`, `-`, `*`, `/`). NumPy computes all $n$ results in a single compiled instruction pass (SIMD — Single Instruction, Multiple Data), instead of Python evaluating $n$ separate bytecode operations in an interpreted loop. This is why the notebook's benchmark shows a ~15–20x speedup even on a simple element-wise add — the gap grows further for more complex operations.

### 3. Memory layout: strides

NumPy doesn't just store numbers contiguously — it uses a **stride** (bytes to skip per dimension) to interpret that flat memory as an N-dimensional array without copying data. For a `(3, 4)` array of `int64` (8 bytes each), stored row-major (C order):

$$\text{strides} = (4 \times 8,\ 1 \times 8) = (32, 8) \text{ bytes}$$

This means `array[i, j]` is found at byte offset $i \times 32 + j \times 8$ from the start — pure arithmetic, no pointer chasing. This is *why* operations like `.reshape()` and `.T` (transpose) are **free** (they just change the stride metadata, not the underlying data), while `.flatten()` (which forces a new contiguous copy) is not.

### 4. Broadcasting

Broadcasting extends element-wise operations to arrays of **different but compatible shapes**, following two rules (comparing shapes right-to-left):

1. Dimensions are compatible if they are **equal**, or one of them is **1**.
2. The dimension of size 1 is conceptually "stretched" to match the other, without copying memory.

**Worked example:** $X$ has shape `(3, 3)`, $\boldsymbol{\mu}$ has shape `(3,)`. Comparing right-to-left: `3` vs `3` → equal, OK. $\boldsymbol{\mu}$ is implicitly treated as shape `(1, 3)` and stretched to `(3, 3)`:

$$X_{\text{norm}}[i, j] = X[i, j] - \mu_j \quad \forall i \in \{1,\dots,n\}$$

This is precisely **feature standardization** (also called Z-score normalization), used before training almost every classical ML model — especially distance-based ones (KNN, SVM) and gradient-descent-based ones (Logistic Regression, Neural Networks):

$$X_{\text{std}}[i, j] = \frac{X[i,j] - \mu_j}{\sigma_j}, \qquad \mu_j = \frac{1}{n}\sum_{i=1}^n X_{ij}, \qquad \sigma_j = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_{ij} - \mu_j)^2}$$

> **Note on `ddof`:** NumPy's `.std()` defaults to the *population* standard deviation (divides by $n$, i.e. `ddof=0`). Pandas' `.std()` defaults to the *sample* standard deviation (divides by $n-1$, i.e. `ddof=1`, Bessel's correction). Mixing the two libraries without noticing this is a common source of subtly-wrong statistics.

### 5. `axis` — which dimension collapses

For an aggregation function $f$ (mean, sum, std, ...) applied to $X \in \mathbb{R}^{n \times d}$:

$$f(X, \text{axis}=0)_j = f(X_{1j}, X_{2j}, \dots, X_{nj}) \quad \rightarrow \text{one result per \textbf{column} (per feature)}$$
$$f(X, \text{axis}=1)_i = f(X_{i1}, X_{i2}, \dots, X_{id}) \quad \rightarrow \text{one result per \textbf{row} (per sample)}$$

`axis=k` means **"collapse dimension $k$, keep the others."** A mnemonic: the resulting array's shape is $X$'s shape with axis $k$ *removed*. `X.sum(axis=0)` on a `(3,3)` array gives shape `(3,)`; `X.sum(axis=(0,1))` gives a scalar.

### 6. Dot product & matrix multiplication

For $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$$

For $X \in \mathbb{R}^{n \times d}$ and $W \in \mathbb{R}^{d \times k}$:

$$(XW)_{i,l} = \sum_{j=1}^d X_{i,j} W_{j,l}$$

This single operation — `X @ W` — **is** the forward pass of a linear layer in every neural network, and the core computation behind linear/logistic regression's prediction step ($\hat{y} = Xw + b$), covered later in the Classical ML notebooks. Every "the model made a prediction" moment ultimately bottoms out in this formula.

### 7. Split-apply-combine (Pandas `groupby`)

For data partitioned into groups $G_1, \dots, G_k$ by a key, and an aggregation function $f$:

$$\text{result}_k = f\big(\{x_i \mid x_i \in G_k\}\big)$$

`df.groupby(key)[col].mean()` is this formula applied automatically across every group — internally, Pandas still uses NumPy's vectorized aggregation within each group, so it stays fast even on large datasets.

---

## ⚙️ Why NumPy Is Fast — Internals

| Python `list` | NumPy `ndarray` |
|---|---|
| Array of pointers to scattered Python objects | Contiguous block of raw, fixed-type memory |
| Type-checked per element, per operation | Single dtype, checked once |
| Interpreted loop per operation (bytecode dispatch) | Compiled C loop, SIMD-vectorized |
| Flexible (mixed types) | Fast (homogeneous types) |
| No stride/shape concept | Reshape/transpose are metadata-only, zero-copy |
| Per-element Python object overhead (~28 bytes for an `int`) | Fixed-width primitive (e.g. 8 bytes for `int64`) |

---

## 🔢 dtype Reference

| dtype | Size | Typical ML use |
|---|---|---|
| `int64` | 8 bytes | Default integer type on most platforms |
| `float64` | 8 bytes | NumPy's default float — high precision |
| `float32` | 4 bytes | Deep learning default — half the memory, GPU-friendly |
| `bool` | 1 byte | Masks, filters, binary labels |
| `object` | varies | Mixed types / strings in Pandas — **avoid in NumPy**, kills vectorization |

Rule of thumb: if a Pandas column shows `dtype: object` and you expected numbers, something (often a stray string or missing-value placeholder) is preventing numeric conversion — this is one of the most common real-world data bugs.

---

## 🔍 Indexing & Slicing Cheat Sheet

| Syntax | Meaning | Example |
|---|---|---|
| `arr[i]` | Single element/row | `arr[2]` |
| `arr[i:j]` | Slice, `j` exclusive | `arr[1:4]` |
| `arr[i:j:k]` | Slice with step | `arr[::2]` (every other element) |
| `arr[:, j]` | All rows, column `j` | `matrix[:, 0]` |
| `arr[i, :]` | Row `i`, all columns | `matrix[0, :]` |
| `arr[mask]` | Boolean masking | `arr[arr > 5]` |
| `arr[[i, j, k]]` | Fancy indexing (specific indices) | `arr[[0, 2, 4]]` |
| `arr[::-1]` | Reverse | `arr[::-1]` |
| `arr[..., 0]` | Ellipsis — all preceding dims | for high-dimensional tensors |

---

## 🆚 `loc` vs `iloc` — Detailed Comparison

| | `.loc[]` | `.iloc[]` |
|---|---|---|
| Selects by | **Label** (index value / column name) | **Integer position** |
| Slice end | **Inclusive** (`df.loc[0:2]` includes row labeled `2`) | **Exclusive** (`df.iloc[0:2]` excludes position `2`, like Python) |
| Works with boolean masks | ✅ `df.loc[df.col > 5]` | ❌ not directly |
| Safe after sorting/filtering | ✅ (labels don't shift) | ⚠️ (positions shift) |
| Typical use | "Give me rows where condition X" | "Give me the first 5 rows" |

**Concrete trap:** if you `df.sort_values("math")` and then use `.iloc[0]`, you get the *first row after sorting* (correct, by position). But `.loc[0]` still fetches the row **originally labeled** `0` — which, after sorting, is no longer the smallest value. This single confusion causes more silent Pandas bugs than any other.

---

## ⚠️ Common Pitfalls & Gotchas

1. **Views vs. copies** — NumPy slicing (`arr[1:3]`) returns a *view* (shares memory with the original); modifying it modifies the original array too. Fancy indexing (`arr[[1,3]]`) returns a *copy*. Use `.copy()` explicitly when you need an independent array.
2. **`SettingWithCopyWarning`** — chained indexing like `df[df.a > 5]["b"] = 1` may silently fail to modify `df`, because the intermediate `df[df.a > 5]` might be a copy, not a view. Use `df.loc[df.a > 5, "b"] = 1` instead.
3. **`NaN != NaN`** — `np.nan == np.nan` evaluates to `False` (by IEEE 754 spec). Never compare for missing values with `==`; always use `.isna()` / `np.isnan()`.
4. **Integer overflow** — fixed-width integer dtypes can silently wrap around (`np.int8(127) + 1 == -128`). Python's built-in `int` has arbitrary precision and never does this — a real difference to be aware of.
5. **`axis` direction is backwards from intuition for many beginners** — `axis=0` operates "down the rows" (i.e., per-column result), not "along rows." Re-read the axis section above if this ever feels wrong.
6. **Mean/std `ddof` mismatch** — see the `ddof` note in the Mathematical Explanation section; NumPy and Pandas disagree on the default.

---

## ⏱ Computational Complexity

| Operation | Complexity | Notes |
|---|---|---|
| Element-wise op on $n$ elements | $O(n)$ | Vectorized, but with a much smaller constant than a Python loop |
| Dot product of two length-$n$ vectors | $O(n)$ | |
| Matrix multiply $(n{\times}d) \cdot (d{\times}k)$ | $O(ndk)$ | Dominant cost in linear models & neural network layers |
| `groupby` + aggregate | $O(n)$ | One pass to bucket, one pass to aggregate (amortized) |
| `sort_values` | $O(n \log n)$ | Standard comparison sort under the hood |
| `.loc` / `.iloc` single lookup | $O(1)$ average | Hash-based index lookup for `.loc` (non-monotonic index), positional for `.iloc` |

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.eye` | Array creation |
| `.shape`, `.dtype`, `.ndim`, `.size` | Array introspection |
| `.reshape()`, `.T` | Zero-copy shape/layout changes |
| `arr[mask]`, `arr[[i, j]]` | Boolean masking, fancy indexing |
| `.mean()`, `.std()`, `.sum()` with `axis=` | Aggregation along a dimension |
| `np.dot`, `@` | Dot product / matrix multiplication |
| `np.random.seed`, `np.random.shuffle`, `np.random.randn` | Reproducible randomness |
| `pd.Series`, `pd.DataFrame` | Labeled 1D / 2D data structures |
| `.loc[]` vs `.iloc[]` | Label-based vs position-based indexing |
| `.isna()`, `.fillna()`, `.dropna()` | Missing data handling |
| `.groupby()` | Split-apply-combine aggregation |
| `.describe()`, `.info()`, `.value_counts()` | Quick statistical summary |

---

## 📝 Self-Test Exercises

Try these in a scratch cell before moving to the next topic — if any feels hard, revisit that section above.

1. Create a `(5, 5)` array of random integers and compute the mean of each **row** without looking at the answer key (which `axis`?).
2. Standardize a `(100, 4)` random feature matrix using only broadcasting (no loops).
3. Given a DataFrame with a `NaN` in one column, write one line that replaces it with that column's **median** (not mean).
4. Explain out loud why `df.iloc[0:3]` returns 3 rows but `df.loc[0:3]` (with a default integer index) returns 4.
5. Multiply a `(10, 3)` matrix by a `(3, 1)` weight vector and explain what real-world ML operation this represents.

---

## 📓 Notebook

Hands-on implementation of every concept above, with a runnable speed benchmark (loop vs vectorized) and an end-to-end mini example combining NumPy-generated data with Pandas exploration:

➡️ **[01_python_numpy_pandas.ipynb](01_python_numpy_pandas.ipynb)**

---

## 📚 Further Reading

- [NumPy official documentation](https://numpy.org/doc/stable/)
- [Pandas official documentation](https://pandas.pydata.org/docs/)
- [NumPy: The Absolute Basics for Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html)
- [Pandas "10 minutes to pandas"](https://pandas.pydata.org/docs/user_guide/10min.html)

---
[← Back to Foundation](../README.md)
