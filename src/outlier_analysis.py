from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EDA_OUTPUT_DIR, PROCESSED_DATA_DIR


def main() -> None:
    df = pd.read_csv(PROCESSED_DATA_DIR / "daily_balance.csv", parse_dates=["date"])
    for target in ["purchase", "redeem"]:
        z = (df[target] - df[target].mean()) / df[target].std(ddof=0)
        q1, q3 = df[target].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (z.abs() > 3) | (df[target] < q1 - 1.5 * iqr) | (df[target] > q3 + 1.5 * iqr)
        out = df.loc[mask, ["date", "purchase", "redeem"]]
        out.to_csv(EDA_OUTPUT_DIR / f"outliers_{target}.csv", index=False)
        plt.figure(figsize=(12, 5))
        plt.plot(df["date"], df[target], label=target)
        plt.scatter(out["date"], out[target], color="red", label="outlier")
        plt.title(f"{target.title()} Outliers")
        plt.xlabel("Date")
        plt.ylabel("Amount")
        plt.legend()
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / f"outliers_{target}.png", dpi=160)
        plt.close()
    print("Outlier analysis written.")


if __name__ == "__main__":
    main()
