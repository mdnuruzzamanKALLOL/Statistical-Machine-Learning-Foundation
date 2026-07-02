# 🧮 Math Refresher — Linear Algebra, Probability & Calculus for ML

> Status: ✅ Complete — [Open the notebook →](05_math_refresher.ipynb)

This is the last Foundation topic, and the one every Classical ML algorithm ahead ultimately reduces to. Linear/Logistic Regression is linear algebra + calculus. Naive Bayes is probability theory. Gradient Boosting is calculus (gradients) applied repeatedly, over and over. If topics 01–04 were about *handling* data, this topic is about the language every algorithm ahead is *written in*.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters — The Payoff Map](#-why-this-topic-matters--the-payoff-map)
3. [Part I — Linear Algebra](#-part-i--linear-algebra)
4. [Part II — Probability](#-part-ii--probability)
5. [Part III — Statistics](#-part-iii--statistics)
6. [Part IV — Calculus & Optimization](#-part-iv--calculus--optimization)
7. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
8. [Function Reference](#-function-reference-used-in-the-notebook)
9. [Self-Test Exercises](#-self-test-exercises)
10. [Notebook](#-notebook)
11. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Four branches of math, each answering a different question an ML algorithm needs answered:

| Branch | Question it answers | Notebook demo |
|---|---|---|
| **Linear Algebra** | How do I represent and transform data efficiently? | Solving $Ax=b$, eigenvectors |
| **Probability** | How do I reason about uncertainty? | Distributions, Bayes' theorem |
| **Statistics** | How do I summarize and generalize from a sample? | Covariance matrix, CLT |
| **Calculus** | How does a model actually *learn* from data? | Gradient descent |

---

## 🎯 Why This Topic Matters — The Payoff Map

Every concept below is a direct preview of a Classical ML notebook coming next. This table is the single most useful thing in this README — bookmark it:

| Math concept (this notebook) | Where it reappears | How |
|---|---|---|
| Solving $Ax=b$ | Linear Regression | The Normal Equation, $w = (X^TX)^{-1}X^Ty$, is exactly this |
| Eigenvectors of a covariance matrix | PCA (Unsupervised) | PCA's components *are* the eigenvectors, ranked by eigenvalue |
| L1 / L2 norms | Ridge & Lasso Regression | Ridge penalizes the L2 norm of weights, Lasso the L1 norm |
| Bayes' theorem | Naive Bayes (Classification) | Naive Bayes computes exactly this formula per class, per prediction |
| Gaussian distribution | Naive Bayes, LDA/QDA | Both assume features are Gaussian-distributed per class |
| Gradient descent | Logistic Regression, Boosting, every neural net | The literal training loop of these algorithms |
| Central Limit Theorem | Cross-Validation, Hypothesis testing | Justifies why performance-metric averages across folds behave predictably |
| Covariance / correlation | Ridge/Lasso (multicollinearity), PCA | High feature correlation destabilizes linear model coefficients |
| SVD / matrix rank | PCA, Linear Regression multicollinearity | SVD is PCA's actual computation; rank-deficiency breaks the Normal Equation |
| Hypothesis testing (t-test) | Model Evaluation & Tuning | Rigorously compares two models' cross-validation scores, not just eyeballing means |
| Law of Large Numbers | Model Evaluation & Tuning, general ML | Formal reason more training data → more reliable learned parameters |

---

## 📐 Part I — Linear Algebra

### 1. Vector norms

For $\mathbf{v} \in \mathbb{R}^n$:

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2} \quad \text{(L2 / Euclidean norm)} \qquad \|\mathbf{v}\|_1 = \sum_{i=1}^n |v_i| \quad \text{(L1 / Manhattan norm)}$$

Geometrically, $\|\mathbf{v}\|_2$ is straight-line distance; $\|\mathbf{v}\|_1$ is distance if you could only move along grid lines (like city blocks — hence "Manhattan"). **L2** penalizes large individual values heavily (squared), so Ridge regression (which penalizes $\|\mathbf{w}\|_2^2$) shrinks all weights smoothly toward zero. **L1** penalizes proportionally, so Lasso regression (which penalizes $\|\mathbf{w}\|_1$) can push weights to *exactly* zero — which is why Lasso performs automatic feature selection and Ridge does not.

### 2. Matrix inverse & solving linear systems

For a square, invertible matrix $A$, the inverse $A^{-1}$ satisfies $AA^{-1} = I$ (the identity matrix). Given $A\mathbf{x} = \mathbf{b}$:

$$\mathbf{x} = A^{-1}\mathbf{b}$$

$A$ is invertible **if and only if** $\det(A) \neq 0$. This single equation, applied to $X^TX \mathbf{w} = X^Ty$, *is* the Normal Equation — the closed-form way to fit Linear Regression without any iterative optimization, covered as the very next notebook after this repo.

> **Practical note:** `np.linalg.inv(A) @ b` and `np.linalg.solve(A, b)` give the same mathematical answer, but `solve` is preferred in real code — it avoids explicitly computing the inverse (numerically less stable and slower for larger systems), using more efficient decomposition methods internally.

### 3. Eigenvalues & eigenvectors

For a square matrix $A$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:

$$A\mathbf{v} = \lambda \mathbf{v}$$

In words: applying $A$ to $\mathbf{v}$ doesn't rotate it, only scales it by $\lambda$. When $A$ is a **covariance matrix**, its eigenvectors point along the directions of maximum variance in the data, and the corresponding eigenvalues quantify *how much* variance lies along each direction. This is precisely what **PCA** computes: rank eigenvectors by eigenvalue (largest first), keep the top $k$, and project the data onto them — a full dimensionality reduction pipeline that is, underneath, nothing more than this equation applied to a covariance matrix.

### 4. Matrix rank & singular matrices

The **rank** of $A$ is the number of linearly independent rows (equivalently, columns) it has. A square matrix is **singular** (non-invertible, $\det(A) = 0$) exactly when its rank is less than its dimension — meaning at least one row/column is a linear combination of the others. In a feature matrix $X$, this happens when two features are perfectly collinear (e.g., a feature measured in two different unit systems), and it is precisely what makes $X^TX$ non-invertible in the Normal Equation from §2 — the formal cause of "perfect multicollinearity" breaking Linear Regression's closed-form solution.

### 5. Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{n \times d}$ (not just square, not just symmetric) factors as:

$$A = U \Sigma V^T$$

where $U$ and $V$ are orthogonal matrices and $\Sigma$ is diagonal, holding the **singular values** $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$. SVD generalizes eigendecomposition to *any* matrix shape, and is the numerically stable algorithm libraries actually use to compute PCA: the columns of $V$ (the **right singular vectors**) are exactly PCA's principal components, and $\sigma_i^2$ relates directly to the variance explained by component $i$.

---

## 🎲 Part II — Probability

### 1. Random variables & distributions

A probability distribution describes how likely each outcome of a random variable is. Two families matter most for classical ML:

**Normal (Gaussian) distribution** — continuous, bell-shaped, parameterized by mean $\mu$ and standard deviation $\sigma$:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Assumed by: Naive Bayes (Gaussian variant), LDA/QDA, and implicitly by any method relying on the Central Limit Theorem.

**Binomial distribution** — discrete, "number of successes in $n$ independent trials, each with success probability $p$":

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Relevant whenever you're reasoning about counts of correct/incorrect predictions across a dataset.

### 2. Bayes' theorem

The single most important formula in this notebook for the Classical ML series ahead:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

In classification terms, for a class $C$ and observed features $X$:

$$P(C \mid X) = \frac{P(X \mid C)\, P(C)}{P(X)}$$

- $P(C)$ — the **prior**: how likely class $C$ is before seeing any features
- $P(X \mid C)$ — the **likelihood**: how likely these features are, given the class
- $P(C \mid X)$ — the **posterior**: what we actually want — how likely the class is, given what we observed

The notebook's disease-test example demonstrates the counter-intuitive result that a strong prior (rare disease) can dominate a strong likelihood (accurate test) — a 95%-accurate test on a 1%-prevalence disease still only yields ~16% true positive probability. **Naive Bayes**, covered in the Classification notebooks, computes exactly this formula for every class, assuming features are conditionally independent given the class (the "naive" assumption).

### 3. Expectation & variance

For a random variable $X$ with possible values $x_i$ and probabilities $p_i$:

$$\mathbb{E}[X] = \sum_i x_i\, p_i \qquad \text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

These are the population-level analogues of the sample mean $\bar{x}$ and sample variance $s^2$ used throughout topics 01–04 — expectation and variance are what those sample statistics are *estimating*.

### 4. Conditional probability & independence

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

$A$ and $B$ are **independent** exactly when $P(A \mid B) = P(A)$ — knowing $B$ happened doesn't change the probability of $A$. This is the assumption Naive Bayes makes about features *given the class* (the "naive" part of its name): it assumes $P(x_1, x_2, \dots \mid C) = P(x_1 \mid C) \cdot P(x_2 \mid C) \cdots$, which is rarely exactly true in real data but is often a good enough approximation to still make useful predictions.

---

## 📊 Part III — Statistics

### 1. Covariance matrix

Generalizing topic 02's pairwise Pearson correlation to *every* feature pair simultaneously, for a data matrix $X \in \mathbb{R}^{n \times d}$:

$$\Sigma_{jk} = \text{Cov}(X_j, X_k) = \frac{1}{n-1}\sum_{i=1}^n (X_{ij} - \bar{X}_j)(X_{ik} - \bar{X}_k)$$

$\Sigma$ is a $d \times d$ symmetric matrix; its diagonal entries are each feature's variance, off-diagonal entries are pairwise covariances. This is the exact matrix whose eigenvectors PCA computes (Part I, §3) — everything in this README connects.

### 2. The Central Limit Theorem (CLT)

For i.i.d. samples $X_1, \dots, X_k$ drawn from *any* distribution with finite mean $\mu$ and variance $\sigma^2$, the distribution of the sample mean $\bar{X}$ approaches Normal as $k$ grows, regardless of the original distribution's shape:

$$\bar{X} \approx \mathcal{N}\left(\mu,\ \frac{\sigma^2}{k}\right) \quad \text{as } k \to \infty$$

The notebook demonstrates this directly: sampling means from a heavily right-skewed exponential population still produces a bell-shaped histogram of means. This is *why* K-Fold Cross-Validation (Model Evaluation & Tuning, later in Classical ML) can treat the average performance metric across folds as a reasonably stable, normally-behaved estimate — even when the underlying error distribution per sample is not normal at all.

### 3. Confidence intervals

A 95% confidence interval for a sample mean, using the CLT's normal approximation:

$$\bar{X} \pm 1.96 \cdot \frac{s}{\sqrt{n}}$$

where $s$ is the sample standard deviation and $1.96$ is the 97.5th percentile of the standard normal (leaving 2.5% in each tail, 5% total, for a 95% interval). The *correct* interpretation is subtle and commonly misstated: it does **not** mean "95% probability the true mean is in this specific interval" — it means "if you repeated this sampling procedure many times, 95% of the intervals constructed this way would contain the true mean." The notebook verifies this exact definition by simulating the repetition directly.

### 4. Hypothesis testing — the t-test

To test whether two samples' means differ significantly, the two-sample t-statistic is:

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}$$

Large $|t|$ (relative to the t-distribution) means the observed difference is unlikely under the **null hypothesis** ($H_0$: no true difference). The **p-value** converts $t$ into a probability: it is $P(\text{observing a difference this extreme} \mid H_0 \text{ is true})$. A small p-value (conventionally $< 0.05$) is evidence to reject $H_0$. This exact test is what you'll use to rigorously compare two models' cross-validation scores later, rather than just eyeballing which mean looks bigger.

### 5. Law of Large Numbers (LLN)

As sample size $n \to \infty$, the sample mean converges to the true population mean:

$$\bar{X}_n \xrightarrow{n \to \infty} \mu$$

This is distinct from the CLT (which describes the *shape* of $\bar{X}_n$'s distribution around $\mu$) — LLN is the more basic guarantee that $\bar{X}_n$ actually *lands on* $\mu$ eventually. It's the formal reason "more training data generally produces more reliable estimates" (feature means, class frequencies, learned parameters) isn't just folklore.

---

## 📈 Part IV — Calculus & Optimization

### 1. Derivatives & gradients

The derivative $f'(x)$ measures how fast $f$ changes as $x$ changes — the *slope* at a point. For a multivariable function $f(x_1, \dots, x_d)$, the **gradient** collects every partial derivative into a vector:

$$\nabla f = \left(\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_d}\right)$$

The gradient always points in the direction of **steepest increase** of $f$. To *minimize* $f$ (e.g., minimize prediction error), step in the *opposite* direction.

### 2. Gradient descent

The update rule used to train Logistic Regression, Gradient Boosting, and every neural network in existence:

$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

where $\eta$ (the **learning rate**) controls step size. The notebook minimizes the toy function $f(x) = (x-3)^2 + 2$ (true minimum at $x=3$) starting from $x=-5$, and watches it converge to $x \approx 2.99$ in 30 steps — the exact same mechanism, scaled up to millions of parameters, is what "training" means for the vast majority of modern ML.

> **Learning rate intuition:** too small $\eta$ → convergence is correct but painfully slow. Too large $\eta$ → the update can overshoot the minimum and diverge or oscillate. Tuning $\eta$ is one of the most consequential hyperparameter choices in the Model Evaluation & Tuning notebooks ahead.

### 3. Numerical gradients — finite differences

When an analytical derivative is inconvenient (or you want to double-check one), the **central finite-difference** formula approximates it directly:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\,\mathbf{e}_i) - f(\mathbf{x} - h\,\mathbf{e}_i)}{2h}$$

for a small step $h$ (e.g. $10^{-5}$) and unit vector $\mathbf{e}_i$. The notebook confirms this matches the true analytical gradient of $f(x,y) = x^2 + 3y^2$ to 5 decimal places — this exact technique (`gradcheck`) is how deep learning frameworks like PyTorch verify a custom-written backward pass is implemented correctly.

### 4. Multivariate gradient descent

Nothing changes about the update rule from §2 when moving to multiple dimensions — only $\nabla f$ becomes a vector instead of a scalar:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

The notebook visualizes this on $f(x,y) = (x-2)^2 + 3(y+1)^2$ via a contour plot: the descent path bends to always point toward steeper regions of the bowl, converging to the true minimum $(2, -1)$ — the same picture, just in 2D instead of 1D, as what happens (in far higher dimensions) when training any gradient-based model.

---

## ⚠️ Common Pitfalls & Gotchas

1. **Confusing L1 and L2's effect on sparsity** — L1 (Lasso) can zero out weights entirely (sparse solutions); L2 (Ridge) shrinks weights smoothly but essentially never to exactly zero. This single distinction explains a real, practical difference in the Regression notebooks ahead.
2. **Assuming a matrix is always invertible** — check $\det(A) \neq 0$ first; a singular (non-invertible) matrix means the linear system has no unique solution, which shows up in Linear Regression as **perfect multicollinearity** between features.
3. **Misreading a prior-dominated Bayes result as "the test is bad"** — in the notebook's example, the test genuinely is 95% accurate; the low posterior is a correct consequence of a low prior, not a flaw in the test. This distinction matters when interpreting Naive Bayes' predicted probabilities later.
4. **Treating the Central Limit Theorem as "my raw data becomes normal"** — it does not. CLT is a statement about the distribution of **sample means** (or other sample statistics), not about the original per-sample data, which can stay however skewed it started.
5. **Picking a learning rate without checking convergence** — a divergent or oscillating gradient descent run (loss increasing, not decreasing) almost always means $\eta$ is too large; this exact symptom reappears when tuning Logistic Regression and boosting models later.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `np.linalg.norm(v, ord=)` | L1/L2 vector norms |
| `np.linalg.inv()`, `np.linalg.solve()` | Matrix inverse, linear system solving |
| `np.linalg.det()` | Determinant (invertibility check) |
| `np.linalg.eig()` | Eigenvalues and eigenvectors |
| `scipy.stats.norm`, `.binom`, `.poisson` | Probability distribution PDFs/PMFs |
| `np.cov(X, rowvar=False)` | Sample covariance matrix |
| `np.random.multivariate_normal()` | Correlated random data generation |
| `np.linalg.matrix_rank()` | Matrix rank (singularity check) |
| `np.linalg.svd()` | Singular Value Decomposition |
| `scipy.stats.ttest_ind()` | Two-sample t-test (hypothesis testing) |
| Manual finite-difference formula | Numerical gradient approximation |

---

## 📝 Self-Test Exercises

1. Given a $2\times2$ matrix, compute its determinant by hand and predict (before running code) whether `np.linalg.inv()` will succeed.
2. Explain, using the L1/L2 norm shapes, why Lasso's penalty can zero out a weight but Ridge's essentially never does.
3. Verify by hand that $(1, 0)$ is or isn't an eigenvector of $\begin{pmatrix}2 & 0 \\ 0 & 3\end{pmatrix}$, and state its eigenvalue if so.
4. Change the disease prevalence in the Bayes' theorem example from 1% to 20% and predict, before running it, whether the posterior probability goes up or down — then verify.
5. Run the gradient descent demo with `learning_rate = 1.5` instead of `0.1` and describe what happens to the convergence path — connect it to the "Learning rate intuition" note above.

---

## 📓 Notebook

Hands-on code for every concept above: vector norms visualized, a linear system solved two ways, eigenvectors verified and plotted, three probability distributions visualized, Bayes' theorem computed numerically, a covariance matrix + Central Limit Theorem simulation, and a from-scratch gradient descent run with a convergence plot:

➡️ **[05_math_refresher.ipynb](05_math_refresher.ipynb)**

---

## 📚 Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) (visual intuition for eigenvectors, matrix transforms)
- [3Blue1Brown: Essence of Calculus](https://www.3blue1brown.com/topics/calculus) (visual intuition for derivatives/gradients)
- [scipy.stats documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seeing Theory — visual introduction to probability & statistics](https://seeing-theory.brown.edu/)

---
[← Back to Foundation](../README.md)
