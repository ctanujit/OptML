"""MATH-224 Lecture 7: Gradient Ascent / Descent.

Python companion to the interactive HTML lecture.
Requirements: numpy, scipy, matplotlib; scikit-learn for the optional final demo.

The example values in the interactive HTML's Cauchy lab are deliberately fixed to
show a multimodal likelihood. Python and R use different RNGs: identical integer
seeds need not generate identical samples across languages.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp


def gradient_descent_quadratic(x0=5.0, learning_rate=0.2, n_steps=30):
    """Minimize f(x)=0.5*(x-1)^2 with f'(x)=x-1."""
    x = float(x0)
    path = [x]
    for _ in range(n_steps):
        x -= learning_rate * (x - 1.0)
        path.append(x)
        if abs(path[-1] - path[-2]) < 1e-10:
            break
    return np.asarray(path)


def cauchy_loglik(mu, observations):
    x = np.asarray(observations, dtype=float)
    return -x.size * np.log(np.pi) - np.log1p((x - mu) ** 2).sum()


def cauchy_score(mu, observations):
    x = np.asarray(observations, dtype=float)
    z = x - mu
    return np.sum(2.0 * z / (1.0 + z * z))


def gradient_ascent_cauchy(observations, start, step_size=0.3, max_iter=120, tol=1e-8):
    """Use the exact (summed) log-likelihood score from the handwritten notes."""
    mu = float(start)
    path = [mu]
    for _ in range(max_iter):
        new_mu = mu + step_size * cauchy_score(mu, observations)
        if not np.isfinite(new_mu):
            break
        path.append(new_mu)
        if abs(new_mu - mu) < tol:
            break
        mu = new_mu
    return np.asarray(path)


def demo_cauchy():
    # Fixed observations give two local likelihood maxima, a useful classroom example.
    sample = np.array([3.4, 4.7, 5.5, 16.5])
    grid = np.linspace(-8, 26, 700)
    objective = np.array([cauchy_loglik(mu, sample) for mu in grid])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(grid, objective, color='black', lw=2, label='Cauchy log-likelihood')
    for start, rate, label in [
        (float(np.median(sample)), 0.3, 'Median start, t=0.3'),
        (7.0, 0.3, 'Start 7, t=0.3'),
        (19.0, 0.3, 'Start 19, t=0.3'),
        (7.0, 1.0, 'Start 7, t=1.0'),
    ]:
        path = gradient_ascent_cauchy(sample, start=start, step_size=rate)
        shown = path[:min(len(path), 30)]
        vals = [cauchy_loglik(mu, sample) for mu in shown]
        ax.plot(shown, vals, '-o', ms=3.5, lw=1.4, label=f'{label}; final={path[-1]:.3f}')
    ax.set(xlabel='Location parameter μ', ylabel='ℓ(μ)', title='Gradient ascent: Cauchy likelihood and starting values')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def binary_log_loss(beta, X, y):
    """Mean negative Bernoulli log-likelihood; stable for large predictors."""
    logits = X @ beta
    return np.mean(np.logaddexp(0, logits) - y * logits)


def fit_logistic_from_scratch(X, y, step_size=0.2, epochs=2500, tol=1e-9):
    """Minimize mean binary cross-entropy by gradient descent.

    X must already contain a column of ones if an intercept is wanted.
    y has shape (n,), not (n,1), to avoid accidental broadcasting.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape
    assert y.shape == (n,) and set(np.unique(y)).issubset({0.0, 1.0})
    beta = np.zeros(p, dtype=float)
    history = [binary_log_loss(beta, X, y)]
    for _ in range(epochs):
        prob = expit(X @ beta)
        grad = X.T @ (prob - y) / n
        new_beta = beta - step_size * grad
        history.append(binary_log_loss(new_beta, X, y))
        if np.linalg.norm(new_beta - beta) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta, np.asarray(history)


def demo_logistic():
    """From-scratch logistic regression on the dataset in the uploaded notebook.

    Unlike the uploaded notebook, this consistently uses 1D labels and a stable
    sigmoid, and standardizes features using training-set statistics only.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    dataset = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        dataset.data, dataset.target, test_size=0.2, random_state=42, stratify=dataset.target
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_te = scaler.transform(X_te)
    A_tr = np.column_stack([np.ones(X_tr.shape[0]), X_tr])
    A_te = np.column_stack([np.ones(X_te.shape[0]), X_te])

    beta, history = fit_logistic_from_scratch(A_tr, y_tr, step_size=0.2, epochs=3000)
    probabilities = expit(A_te @ beta)
    predictions = (probabilities >= 0.5).astype(int)
    print('Gradient descent: steps', len(history)-1)
    print('Gradient descent: final training loss', round(history[-1], 5))
    print('Gradient descent: test accuracy', round(accuracy_score(y_te, predictions), 4))

    baseline = LogisticRegression(max_iter=2000)  # regularized sklearn baseline
    baseline.fit(X_tr, y_tr)
    print('scikit-learn (regularized) test accuracy:',
          round(accuracy_score(y_te, baseline.predict(X_te)), 4))
    print('Note: sklearn uses L2 regularization by default; results are not identical objectives.')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history)
    ax.set(xlabel='Gradient-descent iteration', ylabel='Mean negative log-likelihood',
           title='Logistic regression from scratch: training objective')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    for rate in [0.2, 1.0, 2.2]:
        path = gradient_descent_quadratic(5, rate, 12)
        print(f'Quadratic descent, step={rate}:', np.round(path[:8], 3))
    demo_cauchy()
    try:
        demo_logistic()
    except ImportError:
        print('Optional scikit-learn demo skipped: install scikit-learn.')
    plt.show()
