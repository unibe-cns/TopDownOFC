from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from ofcsst.utils import constants, process


def welch_t_bonferroni(
    x,
    y,
    outcome: str = "the outcome",
    group1: str = "Group 1",
    group2: str = "Group 2",
    n_tests: int = 1,
    ci: float = 0.95,
    digits: int = 6,
    verbose: bool = False,
    latex: bool = False,
) -> float:
    """
    Compute Bonferroni adjusted p-value for two-sided Welch independent-samples t test. If verbose=True,
    it prints the Welch's t statistic, Welch df, two-sided Bonferroni-adjusted p-value,
    and the mean difference (Δ) with a CI computed using the Welch standard error and Welch df.
    :param x: First independent sample. NaNs are removed.
    :type x: array-like
    :param y: Second independent sample. NaNs are removed.
    :type y: array-like
    :param outcome: Label used in the reporting sentence (only printed if verbose=True).
    :type outcome: str
    :param group1: Label used in the reporting sentence (only printed if verbose=True).
    :type group1: str
    :param group2: Label used in the reporting sentence (only printed if verbose=True).
    :type group2: str
    :param n_tests: Number of tests for Bonferroni correction (p_adj = min(p * n_tests, 1)).
    :type n_tests: int
    :param ci: Confidence level for the CI on Δ (e.g., 0.95).
    :type ci: float
    :param digits: Decimal places used for t/df/Δ/CI in the printed sentence.
    :type digits: int
    :param verbose: If True, prints the detailed statistical analysis.
    :type verbose: bool
    :param latex: If True, uses LaTeX formatting for the printed sentence.
    :type latex: bool
    :returns: Bonferroni-corrected p-value.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    n1, n2 = x.size, y.size
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 non-NaN observations.")

    if not (isinstance(n_tests, (int, np.integer)) and n_tests >= 1):
        raise ValueError("n_tests must be an integer >= 1.")

    m1, m2 = float(x.mean()), float(y.mean())
    s1, s2 = float(x.std(ddof=1)), float(y.std(ddof=1))

    # Welch t-test
    t_array, p_array = stats.ttest_ind(x, y, equal_var=False)
    t = np.float64(t_array).item()
    p = np.float64(p_array).item()

    # Bonferroni correction
    p_bonf = min(p * n_tests, 1.0)

    # Compute and report statistical details
    if verbose:
        # Welch–Satterthwaite df
        v1, v2 = (s1**2) / n1, (s2**2) / n2
        df_welch: float = (v1 + v2) ** 2 / ((v1**2) / (n1 - 1) + (v2**2) / (n2 - 1))

        # Mean difference (Δ) and CI
        delta = m1 - m2
        se = float(np.sqrt((s1**2) / n1 + (s2**2) / n2))
        alpha = 1.0 - ci
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df_welch))
        delta_lo, delta_hi = delta - tcrit * se, delta + tcrit * se

        # Qualitative direction of the effect
        if p_bonf >= constants.SIGNIFICANCE_THRESHOLD:
            dir_phrase = f"{outcome} did not differ between {group1} and {group2}"
        elif m1 > m2:
            dir_phrase = f"{outcome} was higher in {group1} than in {group2}"
        else:
            dir_phrase = f"{outcome} was lower in {group1} than in {group2}"

        # Latex formatting
        if latex:
            pn = "p\\mbox{\\textsubscript{adj}}"
            cin = "\\% CI"
            es = "$\\Delta$"
        else:
            pn = "p_adj"
            cin = "% CI"
            es = "Δ"

        # p value formatting
        if p_bonf < 0.001:
            if latex:
                p_txt = f"{pn}\\textless0.001"
            else:
                p_txt = f"{pn}<0.001"

        else:
            p_txt = f"{pn}={p_bonf:.5f}"

        sentence = (
            f"Two-sided Welch’s t-test indicated that {dir_phrase} "
            f"(t({df_welch:.{digits}f})={t:.{digits}f}, {p_txt}; "
            f"{es}={delta:.{digits}f}, {int(ci * 100)}{cin} [{delta_lo:.{digits}f}, {delta_hi:.{digits}f}])."
        )
        print(sentence)

    return p_bonf


def logrank_cox_ph(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    n_tests: int = 1,
    latex: bool = False,
    outcome: str = "expertise",
    digits: int = 6,
    simulation: bool = False,
    verbose: bool = True,
) -> float:
    """Run KM + log-rank for a specific pair within a specific learning phase."""
    t_a, e_a = process.get_events(a, simulation=simulation)
    t_b, e_b = process.get_events(b, simulation=simulation)

    # Log-rank test
    lr = logrank_test(t_a, t_b, event_observed_A=e_a, event_observed_B=e_b)
    lr_chi2 = float(lr.test_statistic)
    lr_p_adj = min(float(lr.p_value) * n_tests, 1.0)

    if verbose:
        # Cox proportional hazards model
        df = pd.DataFrame(
            {
                "time": np.r_[t_a, t_b],
                "event": np.r_[e_a, e_b],
                "indicator": np.r_[
                    np.ones(t_a.shape[0], dtype=int), np.zeros(t_b.shape[0], dtype=int)
                ],
            }
        )
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df, duration_col="time", event_col="event", formula="indicator")

        # Extract hazard ratio and confidence interval
        row = cph.summary.loc["indicator"]
        coef = float(row["coef"])
        hr = float(row["exp(coef)"])
        hr_se = float(row["se(coef)"])
        alpha = constants.SIGNIFICANCE_THRESHOLD
        ci_level = 1.0 - alpha
        zcrit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        hr_lo = float(np.exp(coef - zcrit * hr_se))
        hr_hi = float(np.exp(coef + zcrit * hr_se))

        # Print results of statistics
        if simulation:
            obj = "Agents"
        else:
            obj = "Mice"
        if lr_p_adj >= constants.SIGNIFICANCE_THRESHOLD:
            dir_phrase = (
                f"Time to {outcome} did not differ between {label_a} and {label_b}"
            )
        elif hr > 1:
            dir_phrase = f"{obj} in {label_a} reached {outcome} faster than {obj.lower()} in {label_b}"
        elif hr < 1:
            dir_phrase = f"{obj} in {label_a} reached {outcome} slower than {obj.lower()} in {label_b}"
        else:
            dir_phrase = (
                f"Time to {outcome} did not differ between {label_a} and {label_b}"
            )

        if latex:
            chi2_sym = "$\\chi^2$"
            pn = "p\\mbox{\\textsubscript{adj}}"
            ci_suffix = "\\% CI"
        else:
            chi2_sym = "χ²"
            pn = "p_adj"
            ci_suffix = "% CI"

        if lr_p_adj < 0.001:
            p_txt = f"{pn}\\textless0.001" if latex else f"{pn}<0.001"
        else:
            p_txt = f"{pn}={lr_p_adj:.5f}"

        sentence = (
            f"{dir_phrase} "
            f"(Log-rank: {chi2_sym}(df=1)={lr_chi2:.{digits}f}, {p_txt}; "
            f"Cox PH: HR={hr:.{digits}f}, {int(ci_level * 100)}{ci_suffix} "
            f"[{hr_lo:.{digits}f}, {hr_hi:.{digits}f}])."
        )
        print(sentence)

    return lr_p_adj
