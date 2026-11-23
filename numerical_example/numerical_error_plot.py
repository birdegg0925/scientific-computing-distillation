"""Deterministic numerical illustration; this is not an LLM benchmark."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).resolve().parent
    rows = []
    for exponent in range(2, 13):
        epsilon = 10.0 ** (-exponent)
        matrix = np.array([[1.0, 1.0], [1.0, 1.0 + epsilon]])
        exact = np.ones(2)
        right = matrix @ exact
        candidate = np.array([2.0, 0.0])
        rows.append({
            "epsilon": epsilon,
            "relative_residual": float(np.linalg.norm(matrix @ candidate - right) / np.linalg.norm(right)),
            "relative_solution_error": float(np.linalg.norm(candidate - exact) / np.linalg.norm(exact)),
            "condition_2": float(np.linalg.cond(matrix)),
        })
    result = {
        "provenance": "Deterministic constructed numerical example used to illustrate residual and solution error",
        "candidate": [2.0, 0.0],
        "constructed_solution": [1.0, 1.0],
        "rows": rows,
    }
    (here / "numerical_error_values.json").write_text(json.dumps(result, indent=2) + "\n")
    eps = np.array([row["epsilon"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].loglog(eps, [r["relative_residual"] for r in rows], "o-", label="Relative residual")
    axes[0].loglog(eps, [r["relative_solution_error"] for r in rows], "s-", label="Relative solution error")
    axes[0].set(xlabel="Row difference (epsilon)", ylabel="Relative magnitude", title="One wrong candidate, two different checks")
    axes[0].legend(frameon=False)
    axes[1].loglog(eps, [r["condition_2"] for r in rows], "o-", color="#a6422b")
    axes[1].set(xlabel="Row difference (epsilon)", ylabel="2-norm condition number", title="Nearly repeated equations become sensitive")
    for axis in axes:
        axis.grid(alpha=0.2, which="both")
        axis.invert_xaxis()
        axis.axvline(1e-8, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle("Constructed numerical example - no language-model evaluation", fontsize=11)
    fig.savefig(here / "small_residual_large_error.png", dpi=160)
    fig.savefig(here / "small_residual_large_error.svg")
    plt.close(fig)
    print("constructed_cases=11")
    print("relative_solution_error=1.0 in every case")
    print("saved: numerical_error_values.json, small_residual_large_error.png, small_residual_large_error.svg")


if __name__ == "__main__":
    main()
