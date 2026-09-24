"""Multimodality DIAGNOSTICS for a cloud of conditional samples ``{x_s^(i)}`` from one ``x_t``.

Nothing here proves multimodality.  Each number answers a narrower question, and the
experiment reports the set side by side for the long transition, the short-transition control
and the deterministic control:

* :func:`spread_stats` -- how far apart the samples are (pairwise distances, rms per-coordinate
  std).  Spread alone is NOT multimodality: a single broad Gaussian has plenty.
* :func:`pca_scores` -- the low-dimensional representation the model-based diagnostics live in
  (a Gaussian-mixture fit to N ~ 64 points needs far fewer than D ~ 10^3..10^5 coordinates).
* :func:`gmm_model_selection` -- Gaussian mixtures with K = 1..Kmax in PCA space: BIC / AIC and
  K-fold held-out log-likelihood.  "K = 1 wins" is evidence for one mode; a stable win for
  K >= 2 is evidence against it.
* :func:`bootstrap_lrt` -- McLachlan's parametric-bootstrap likelihood-ratio test of K = 1
  against K = 2: the observed 2 (LL_2 - LL_1) against its distribution under data simulated
  from the fitted single Gaussian (the classical LRT asymptotics do not apply to mixtures).
* :func:`clustering_stability` -- does a 2-cluster split survive resampling?  Mean adjusted
  Rand index between bootstrap k-means partitions and the reference one, and the silhouette of
  the reference split -- both calibrated against a UNIMODAL GAUSSIAN NULL with the data's own
  covariance, because k-means always finds two clusters and a silhouette of 0.3 means nothing
  on its own.

Assumptions, stated: the feature representation is treated as Euclidean; the null model is
Gaussian in PCA space (an elongated but unimodal non-Gaussian cloud can reject it -- the
sample grids are the guard against reading "non-Gaussian" as "multimodal"); K-fold and
bootstrap results depend on ``seed``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

__all__ = ["spread_stats", "pca_scores", "gmm_model_selection", "bootstrap_lrt",
           "clustering_stability", "analyze", "summarize"]


def _2d(Z):
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        Z = Z.reshape(Z.shape[0], -1)
    return Z


def spread_stats(Z):
    """Pairwise-distance and per-coordinate spread of an ``(N, D)`` cloud."""
    Z = _2d(Z)
    n, d = Z.shape
    out = dict(n=int(n), dim=int(d), rms_std=float(np.sqrt(Z.var(axis=0, ddof=1).mean())) if n > 1 else 0.0,
               mean_dist_to_centroid=float(np.linalg.norm(Z - Z.mean(0), axis=1).mean()))
    if n > 1:
        D = pdist(Z)
        out.update(pairwise_mean=float(D.mean()), pairwise_median=float(np.median(D)),
                   pairwise_min=float(D.min()), pairwise_max=float(D.max()),
                   pairwise_std=float(D.std()))
    return out


def is_degenerate(Z, rel_tol=1e-6):
    """True when the cloud has (numerically) no spread, e.g. a deterministic control."""
    Z = _2d(Z)
    scale = max(float(np.abs(Z).mean()), 1e-12)
    return Z.shape[0] < 3 or float(np.sqrt(Z.var(axis=0).mean())) < rel_tol * scale


def pca_scores(Z, n_components=10, seed=0):
    from sklearn.decomposition import PCA
    Z = _2d(Z)
    k = int(min(n_components, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=k, svd_solver="full", random_state=seed).fit(Z)
    S = pca.transform(Z)
    info = dict(n_components=k, explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
                total_explained=float(pca.explained_variance_ratio_.sum()))
    return S, info


def _gmm(k, covariance_type, seed, n_init, reg_covar):
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(n_components=k, covariance_type=covariance_type, n_init=n_init,
                           random_state=seed, reg_covar=reg_covar, max_iter=500)


def gmm_model_selection(S, max_k=4, covariance_type="diag", n_init=5, cv_folds=5, seed=0,
                        reg_covar=1e-6):
    """BIC / AIC / K-fold held-out log-likelihood of Gaussian mixtures with K = 1..max_k."""
    from sklearn.model_selection import KFold
    S = _2d(S)
    n = S.shape[0]
    max_k = int(min(max_k, max(1, n // 5)))          # never more components than 5 points each
    rows = []
    kf = KFold(n_splits=min(cv_folds, n), shuffle=True, random_state=seed)
    for k in range(1, max_k + 1):
        gm = _gmm(k, covariance_type, seed, n_init, reg_covar).fit(S)
        cv = []
        for tr, te in kf.split(S):
            g = _gmm(k, covariance_type, seed, n_init, reg_covar).fit(S[tr])
            cv.append(float(g.score(S[te])))         # mean held-out log-lik per point
        rows.append(dict(K=k, bic=float(gm.bic(S)), aic=float(gm.aic(S)),
                         loglik=float(gm.score(S) * n), converged=bool(gm.converged_),
                         cv_loglik_mean=float(np.mean(cv)), cv_loglik_std=float(np.std(cv)),
                         weights=gm.weights_.tolist()))
    bics = np.array([r["bic"] for r in rows])
    cvs = np.array([r["cv_loglik_mean"] for r in rows])
    return dict(covariance_type=covariance_type, max_k=max_k, cv_folds=int(kf.get_n_splits()),
                rows=rows, best_k_bic=int(rows[int(bics.argmin())]["K"]),
                best_k_cv=int(rows[int(cvs.argmax())]["K"]),
                delta_bic_1_vs_2=float(bics[0] - bics[1]) if len(bics) > 1 else None)


def bootstrap_lrt(S, n_boot=200, covariance_type="diag", n_init=5, seed=0, reg_covar=1e-6):
    """Parametric-bootstrap LRT, K = 1 vs K = 2 (McLachlan 1987).  Small p => not one Gaussian."""
    S = _2d(S)
    n = S.shape[0]
    rng = np.random.RandomState(seed)

    def stat(X, s):
        g1 = _gmm(1, covariance_type, s, n_init, reg_covar).fit(X)
        g2 = _gmm(2, covariance_type, s, n_init, reg_covar).fit(X)
        return 2.0 * n * (g2.score(X) - g1.score(X)), g1

    obs, g1 = stat(S, seed)
    null = np.empty(n_boot)
    for b in range(n_boot):
        Xb, _ = g1.sample(n)
        null[b] = stat(Xb[rng.permutation(n)], seed + 1 + b)[0]
    p = float((1 + np.sum(null >= obs)) / (1 + n_boot))
    return dict(statistic=float(obs), null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95)),
                null_q99=float(np.quantile(null, 0.99)), p_value=p, n_boot=int(n_boot), null=null.tolist())


def _kmeans(k, seed, n_init=10):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed)


def _mean_bootstrap_ari(S, ref_labels, k, n_boot, rng, seed):
    from sklearn.metrics import adjusted_rand_score
    n = S.shape[0]
    aris = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        km = _kmeans(k, seed + 1 + b, n_init=3).fit(S[idx])
        aris[b] = adjusted_rand_score(ref_labels, km.predict(S))
    return aris


def clustering_stability(S, k=2, n_boot=200, n_null=100, n_boot_null=30, seed=0):
    """Bootstrap stability (ARI) and silhouette of a k-means split, calibrated on a Gaussian null."""
    from sklearn.metrics import silhouette_score
    S = _2d(S)
    n = S.shape[0]
    rng = np.random.RandomState(seed)
    ref = _kmeans(k, seed).fit(S)
    labels = ref.labels_
    sizes = np.bincount(labels, minlength=k)
    if sizes.min() < 2:
        return dict(k=k, degenerate_split=True, cluster_sizes=sizes.tolist())
    aris = _mean_bootstrap_ari(S, labels, k, n_boot, rng, seed)
    sil = float(silhouette_score(S, labels))

    # Unimodal Gaussian null with the cloud's own mean and covariance
    mu, cov = S.mean(0), np.cov(S, rowvar=False) + 1e-9 * np.eye(S.shape[1])
    null_sil, null_ari = np.empty(n_null), np.empty(n_null)
    for j in range(n_null):
        X = rng.multivariate_normal(mu, cov, size=n)
        km = _kmeans(k, seed + 7919 + j, n_init=3).fit(X)
        null_sil[j] = silhouette_score(X, km.labels_)
        null_ari[j] = _mean_bootstrap_ari(X, km.labels_, k, n_boot_null, rng, seed + 100_000 + j).mean()
    return dict(k=k, cluster_sizes=sizes.tolist(), ari_mean=float(aris.mean()), ari_std=float(aris.std()),
                silhouette=sil, null_silhouette_mean=float(null_sil.mean()),
                null_silhouette_q95=float(np.quantile(null_sil, 0.95)),
                p_silhouette=float((1 + np.sum(null_sil >= sil)) / (1 + n_null)),
                null_ari_mean=float(null_ari.mean()), null_ari_q95=float(np.quantile(null_ari, 0.95)),
                p_ari=float((1 + np.sum(null_ari >= aris.mean())) / (1 + n_null)),
                n_boot=int(n_boot), n_null=int(n_null), labels=labels.tolist())


def analyze(Z, pca_dims=10, max_k=4, covariance_type="diag", cv_folds=5, n_boot=200, n_null=100,
            seed=0):
    """The full diagnostic set on one ``(N, D)`` cloud; degenerate clouds get the spread only."""
    Z = _2d(Z)
    out = dict(spread=spread_stats(Z), degenerate=bool(is_degenerate(Z)))
    if out["degenerate"]:
        return out
    S, pca = pca_scores(Z, pca_dims, seed)
    out["pca"] = pca
    out["gmm"] = gmm_model_selection(S, max_k=max_k, covariance_type=covariance_type,
                                     cv_folds=cv_folds, seed=seed)
    out["lrt"] = bootstrap_lrt(S, n_boot=n_boot, covariance_type=covariance_type, seed=seed)
    out["stability"] = clustering_stability(S, k=2, n_boot=n_boot, n_null=n_null, seed=seed)
    out["pca_scores"] = S
    return out


def summarize(res, label=""):
    """A few human-readable lines for the console / report."""
    sp = res["spread"]
    lines = [f"[{label}] N={sp['n']} D={sp['dim']}  rms per-coord std={sp['rms_std']:.4g}  "
             f"pairwise dist mean={sp.get('pairwise_mean', float('nan')):.4g} "
             f"(min {sp.get('pairwise_min', float('nan')):.3g}, max {sp.get('pairwise_max', float('nan')):.3g})"]
    if res.get("degenerate"):
        lines.append(f"[{label}] cloud is degenerate (no spread): model-based diagnostics skipped")
        return lines
    g, l, st = res["gmm"], res["lrt"], res["stability"]
    lines.append(f"[{label}] PCA {res['pca']['n_components']} dims explain {res['pca']['total_explained']:.2%}")
    lines.append(f"[{label}] GMM ({g['covariance_type']}): best K by BIC = {g['best_k_bic']}, "
                 f"by {g['cv_folds']}-fold CV log-lik = {g['best_k_cv']}; "
                 + ", ".join(f"K={r['K']}: BIC {r['bic']:.1f} / CV {r['cv_loglik_mean']:.2f}" for r in g["rows"]))
    lines.append(f"[{label}] bootstrap LRT K=1 vs 2: stat {l['statistic']:.1f} vs null q95 {l['null_q95']:.1f}, "
                 f"p = {l['p_value']:.3f} (n_boot {l['n_boot']})")
    if st.get("degenerate_split"):
        lines.append(f"[{label}] k-means split degenerate: sizes {st['cluster_sizes']}")
    else:
        lines.append(f"[{label}] k-means(2) sizes {st['cluster_sizes']}: bootstrap ARI {st['ari_mean']:.2f}"
                     f" (Gaussian null {st['null_ari_mean']:.2f}, p = {st['p_ari']:.3f}); silhouette "
                     f"{st['silhouette']:.3f} (null {st['null_silhouette_mean']:.3f}, p = {st['p_silhouette']:.3f})")
    return lines
