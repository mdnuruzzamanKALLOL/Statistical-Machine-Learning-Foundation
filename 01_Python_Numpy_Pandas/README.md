# 🧩 Python, NumPy & Pandas

> Status: ✅ Complete — [Open the notebook →](01_python_numpy_pandas.ipynb)

Every ML pipeline starts here. This topic builds the numerical + tabular data foundation that every algorithm in this series (regression, classification, clustering...) sits on top of.

---

## 📖 Concept & Intuition

- **NumPy (`ndarray`)** stores numbers in one contiguous block of memory with a single fixed type (`dtype`). This is what makes it fast, and it's exactly what a "tensor" in PyTorch/TensorFlow is under the hood.
- **Pandas (`Series` / `DataFrame`)** wraps NumPy arrays with **labels** (row index + column names), turning raw numeric arrays into something you can filter, group, and reason about the way you'd reason about a spreadsheet or SQL table.

Think of the relationship as layers:

```
Python list  →  NumPy ndarray  →  Pandas Series/DataFrame
(objects,       (contiguous,       (ndarray + labels,
 slow loops)     vectorized,        tabular operations)
                 fast)
```

---

## 🧮 Mathematical Explanation

### 1. Arrays as vectors and matrices

A 1D NumPy array is a vector $\mathbf{x} \in \mathbb{R}^n$. A 2D array is a matrix $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features) — the standard shape of a **feature matrix** in every ML algorithm covered later in this series.

### 2. Vectorization

For element-wise operations between $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{c}_i = \mathbf{a}_i \circ \mathbf{b}_i \quad \forall i \in \{1, \dots, n\}$$

where $\circ$ is any operator (`+`, `-`, `*`, `/`). NumPy computes all $n$ results in a single compiled instruction pass (SIMD), instead of Python evaluating $n$ separate bytecode operations in a loop.

### 3. Broadcasting

Broadcasting extends element-wise operations to arrays of **different but compatible shapes**, following two rules (comparing shapes right-to-left):

1. Dimensions are compatible if they are **equal**, or one of them is **1**.
2. The dimension of size 1 is conceptually "stretched" to match the other, without copying memory.

For $X \in \mathbb{R}^{n \times d}$ and $\boldsymbol{\mu} \in \mathbb{R}^{d}$ (shape `(d,)` broadcasts against `(n, d)`):

$$X_{\text{norm}}[i, j] = X[i, j] - \mu_j \quad \forall i \in \{1,\dots,n\}$$

This is precisely **feature standardization**, used before training almost every classical ML model:

$$X_{\text{std}}[i, j] = \frac{X[i,j] - \mu_j}{\sigma_j}, \qquad \mu_j = \frac{1}{n}\sum_{i=1}^n X_{ij}, \qquad \sigma_j = \sqrt{\frac{1}{n}\sum_{i=1}^n (X_{ij} - \mu_j)^2}$$

### 4. `axis` — which dimension collapses

For an aggregation function $f$ (mean, sum, std, ...) applied to $X \in \mathbb{R}^{n \times d}$:

$$f(X, \text{axis}=0)_j = f(X_{1j}, X_{2j}, \dots, X_{nj}) \quad \text{→ one result per \textbf{column} (per feature)}$$
$$f(X, \text{axis}=1)_i = f(X_{i1}, X_{i2}, \dots, X_{id}) \quad \text{→ one result per \textbf{row} (per sample)}$$

`axis=k` means **"collapse dimension $k$, keep the others."**

### 5. Dot product & matrix multiplication

For $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i$$

For $X \in \mathbb{R}^{n \times d}$ and $W \in \mathbb{R}^{d \times k}$:

$$(XW)_{i,l} = \sum_{j=1}^d X_{i,j} W_{j,l}$$

This single operation — `X @ W` — **is** the forward pass of a linear layer in every neural network, and the core computation behind linear/logistic regression's prediction step, covered later in the Classical ML notebooks.

### 6. Split-apply-combine (Pandas `groupby`)

For data partitioned into groups $G_1, \dots, G_k$ by a key, and an aggregation function $f$:

$$\text{result}_k = f\big(\{x_i \mid x_i \in G_k\}\big)$$

`df.groupby(key)[col].mean()` is this formula applied automatically across every group.

---

## ⚙️ Why NumPy Is Fast (Internals)

| Python `list` | NumPy `ndarray` |
|---|---|
| Array of pointers to scattered Python objects | Contiguous block of raw, fixed-type memory |
| Type-checked per element, per operation | Single dtype, checked once |
| Interpreted loop per operation | Compiled C loop, SIMD-vectorized |
| Flexible (mixed types) | Fast (homogeneous types) |

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`, `np.eye` | Array creation |
| `.shape`, `.dtype`, `.ndim`, `.size` | Array introspection |
| `arr[mask]`, `arr[[i, j]]` | Boolean masking, fancy indexing |
| `.mean()`, `.std()`, `.sum()` with `axis=` | Aggregation along a dimension |
| `np.dot`, `@` | Dot product / matrix multiplication |
| `np.random.seed`, `np.random.shuffle`, `np.random.randn` | Reproducible randomness |
| `pd.Series`, `pd.DataFrame` | Labeled 1D / 2D data structures |
| `.loc[]` vs `.iloc[]` | Label-based vs position-based indexing |
| `.isna()`, `.fillna()`, `.dropna()` | Missing data handling |
| `.groupby()` | Split-apply-combine aggregation |
| `.describe()` | Quick statistical summary |

---

## 📓 Notebook

Hands-on implementation of every concept above, with a runnable speed benchmark (loop vs vectorized) and an end-to-end mini example combining NumPy-generated data with Pandas exploration:

➡️ **[01_python_numpy_pandas.ipynb](01_python_numpy_pandas.ipynb)**

---
[← Back to Foundation](../README.md)
