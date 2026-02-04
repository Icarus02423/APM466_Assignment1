import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Files: ONLY CSV. No xlsx anywhere.
BASE_DIR = Path(__file__).resolve().parent
PRICE_CSV = BASE_DIR / "bonds_prices_41x11.csv"
META_CSV = BASE_DIR / "bonds_meta.csv"


# Preferred ten bonds: Mar 01 / Sep 01 ladder
PREFERRED_ISINS = [
    "CA135087L518",  # CAN 0.25 Mar 26
    "CA135087L930",  # CAN 1.00 Sep 26
    "CA135087M847",  # CAN 1.25 Mar 27
    "CA135087N837",  # CAN 2.75 Sep 27
    "CA135087P576",  # CAN 3.50 Mar 28
    "CA135087Q491",  # CAN 3.25 Sep 28
    "CA135087Q988",  # CAN 4.00 Mar 29
    "CA135087R895",  # CAN 3.50 Sep 29
    "CA135087S471",  # CAN 2.75 Mar 30
    "CA135087T388",  # CAN 2.75 Sep 30
]

FACE = 100.0
FREQ = 2  # semiannual


# Column helpers
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def rename_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename common variants into canonical names:
    ISIN, Coupon, IssueDate, MaturityDate, TKData, url
    """
    mapping = {}
    for c in df.columns:
        n = _norm(c)
        if n == "isin":
            mapping[c] = "ISIN"
        elif n in ("coupon", "couponrate", "coupon%"):
            mapping[c] = "Coupon"
        elif n in ("issuedate", "issued", "issuedate:"):
            mapping[c] = "IssueDate"
        elif n in ("maturitydate", "maturity", "maturitydate:"):
            mapping[c] = "MaturityDate"
        elif n in ("tkdata", "tk"):
            mapping[c] = "TKData"
        elif n in ("url", "link"):
            mapping[c] = "url"
    return df.rename(columns=mapping)


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Date columns are those whose column name can be parsed as a date.
    Excludes metadata columns.
    """
    reserved = {"ISIN", "Coupon", "IssueDate", "MaturityDate", "TKData", "url", "CouponDec", "BondName"}
    date_cols = []
    for c in df.columns:
        if c in reserved:
            continue
        dt = pd.to_datetime(str(c).strip(), errors="coerce")
        if pd.notna(dt):
            date_cols.append(c)
    date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(str(x), errors="coerce"))
    if not date_cols:
        raise ValueError(f"No date columns detected in {PRICE_CSV.name}. Columns: {list(df.columns)}")
    return date_cols


def coalesce_columns(df: pd.DataFrame, base: str, alt: str) -> pd.DataFrame:
    """
    If both base and alt exist, fill base missing values using alt.
    Then drop alt.
    If base does not exist but alt exists, rename alt -> base.
    """
    if base in df.columns and alt in df.columns:
        a = df[base].replace("", np.nan)
        b = df[alt].replace("", np.nan)
        df[base] = a.where(a.notna(), b)
        df = df.drop(columns=[alt])
        return df
    if base not in df.columns and alt in df.columns:
        return df.rename(columns={alt: base})
    return df


# Bond math
def parse_coupon_to_decimal(x) -> float:
    s = str(x).strip()
    s = s.replace("%", "").replace(",", ".")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def month_abbr(m: int) -> str:
    return ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]


def format_bond_name(coupon_dec: float, maturity: pd.Timestamp) -> str:
    c_pct = coupon_dec * 100.0
    c_str = f"{c_pct:.2f}".rstrip("0").rstrip(".")
    mon = month_abbr(int(maturity.month))
    yy = str(int(maturity.year))[-2:]
    return f"CAN {c_str} {mon} {yy}"


def generate_coupon_schedule(maturity: pd.Timestamp, years_back: int = 60) -> list[pd.Timestamp]:
    dates = []
    d = pd.Timestamp(maturity).normalize()
    cutoff = d - pd.DateOffset(years=years_back)
    while d > cutoff:
        dates.append(d)
        d = d - pd.DateOffset(months=6)
    return sorted(set(dates))


def accrued_interest(settle: pd.Timestamp, maturity: pd.Timestamp, coupon_dec: float) -> float:
    schedule = generate_coupon_schedule(maturity)
    settle = pd.Timestamp(settle).normalize()

    future = [d for d in schedule if d > settle]
    if not future:
        return 0.0
    next_cp = min(future)

    past = [d for d in schedule if d <= settle]
    last_cp = max(past) if past else next_cp - pd.DateOffset(months=6)

    if settle == last_cp:
        return 0.0

    days_total = (next_cp - last_cp).days
    days_accr = (settle - last_cp).days
    if days_total <= 0 or days_accr <= 0:
        return 0.0

    c_per = FACE * coupon_dec / FREQ
    return c_per * (days_accr / days_total)


def bond_pv_from_yield(settle: pd.Timestamp, maturity: pd.Timestamp, coupon_dec: float, y: float) -> float:
    schedule = generate_coupon_schedule(maturity)
    settle = pd.Timestamp(settle).normalize()

    cash_dates = [d for d in schedule if d > settle]
    if not cash_dates:
        return FACE

    c_per = FACE * coupon_dec / FREQ
    pv = 0.0
    for d in cash_dates:
        t = (d - settle).days / 365.0
        if t <= 0:
            continue
        cf = c_per + (FACE if d == pd.Timestamp(maturity).normalize() else 0.0)
        disc = (1.0 + y / FREQ) ** (-FREQ * t)
        pv += cf * disc
    return pv


def solve_ytm(clean_price: float, settle: pd.Timestamp, maturity: pd.Timestamp, coupon_dec: float) -> float:
    if not np.isfinite(clean_price):
        return np.nan

    ai = accrued_interest(settle, maturity, coupon_dec)
    dirty = float(clean_price) + ai

    def f(rate):
        return bond_pv_from_yield(settle, maturity, coupon_dec, rate) - dirty

    lo, hi = -0.95, 1.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        hi = 3.0
        fhi = f(hi)
        if flo * fhi > 0:
            return np.nan

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)


# Selection
def auto_select_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback: choose Mar 01 / Sep 01 maturities from 2026-03-01 to 2030-09-01,
    tie-break with newer issue date.
    """
    tmp = df.copy()
    tmp["IssueDate"] = pd.to_datetime(tmp["IssueDate"], errors="coerce")
    tmp["MaturityDate"] = pd.to_datetime(tmp["MaturityDate"], errors="coerce")
    tmp = tmp.dropna(subset=["MaturityDate"])

    tmp = tmp[(tmp["MaturityDate"].dt.month.isin([3, 9])) & (tmp["MaturityDate"].dt.day == 1)]
    tmp = tmp[(tmp["MaturityDate"] >= pd.Timestamp("2026-03-01")) & (tmp["MaturityDate"] <= pd.Timestamp("2030-09-01"))]

    tmp = tmp.sort_values(["MaturityDate", "IssueDate"], ascending=[True, False])
    tmp = tmp.drop_duplicates(subset=["MaturityDate"], keep="first")
    tmp = tmp.head(10).copy()

    if tmp.shape[0] < 10:
        raise ValueError(f"Fallback ladder selection found only {tmp.shape[0]} bonds.")
    return tmp


def select_ten_bonds(df: pd.DataFrame) -> pd.DataFrame:
    have = set(df["ISIN"].astype(str))
    if all(x in have for x in PREFERRED_ISINS):
        sel = df[df["ISIN"].isin(PREFERRED_ISINS)].copy()
        return sel.sort_values("MaturityDate").reset_index(drop=True)
    return auto_select_ladder(df).sort_values("MaturityDate").reset_index(drop=True)


# Main
def main():
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Missing {PRICE_CSV.name} in {BASE_DIR}")
    if not META_CSV.exists():
        raise FileNotFoundError(f"Missing {META_CSV.name} in {BASE_DIR}")

    # Load price matrix
    prices = pd.read_csv(PRICE_CSV)
    prices = rename_to_canonical(prices)

    # First column should be ISIN
    if "ISIN" not in prices.columns:
        prices = prices.rename(columns={prices.columns[0]: "ISIN"})
    prices["ISIN"] = prices["ISIN"].astype(str).str.strip()

    # Load meta
    meta = pd.read_csv(META_CSV)
    meta = rename_to_canonical(meta)
    if "ISIN" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "ISIN"})
    meta["ISIN"] = meta["ISIN"].astype(str).str.strip()

    # Detect date columns from prices
    date_cols = detect_date_columns(prices)

    # Convert date columns to numeric
    for c in date_cols:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    # Merge: keep prices as base, use meta to fill missing metadata
    df = prices.merge(meta, on="ISIN", how="left", suffixes=("", "_meta"))

    # Coalesce potential duplicate metadata columns
    for col in ["Coupon", "IssueDate", "MaturityDate", "TKData", "url"]:
        df = coalesce_columns(df, col, f"{col}_meta")

    # Validate required columns exist now
    required = ["ISIN", "Coupon", "IssueDate", "MaturityDate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"After merge, missing required columns: {missing}. Columns now: {list(df.columns)}")

    # Parse meta types
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce")
    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"], errors="coerce")
    df["CouponDec"] = df["Coupon"].apply(parse_coupon_to_decimal)

    # Select ten bonds
    sel = select_ten_bonds(df)
    sel["BondName"] = sel.apply(lambda r: format_bond_name(float(r["CouponDec"]), r["MaturityDate"]), axis=1)

    # Export selected bonds
    selected_out = sel[["BondName", "ISIN", "Coupon", "IssueDate", "MaturityDate"]].copy()
    if "TKData" in sel.columns:
        selected_out["TKData"] = sel["TKData"]
    if "url" in sel.columns:
        selected_out["url"] = sel["url"]

    selected_out["IssueDate"] = selected_out["IssueDate"].dt.strftime("%Y-%m-%d")
    selected_out["MaturityDate"] = selected_out["MaturityDate"].dt.strftime("%Y-%m-%d")
    selected_out.to_csv(BASE_DIR / "selected_bonds.csv", index=False)

    # Compute YTM each day
    ytm_rows = []
    for col in date_cols:
        settle = pd.to_datetime(str(col), errors="coerce")
        if pd.isna(settle):
            continue
        settle = settle.normalize()

        for _, r in sel.iterrows():
            maturity = pd.Timestamp(r["MaturityDate"]).normalize()
            if maturity <= settle:
                continue

            clean_price = float(r[col])
            y = solve_ytm(clean_price, settle, maturity, float(r["CouponDec"]))
            ttm = (maturity - settle).days / 365.0

            ytm_rows.append({
                "date": settle.strftime("%Y-%m-%d"),
                "ISIN": r["ISIN"],
                "BondName": r["BondName"],
                "Coupon": r["Coupon"],
                "MaturityDate": maturity.strftime("%Y-%m-%d"),
                "ttm_years": ttm,
                "clean_price": clean_price,
                "ytm_decimal": y,
                "ytm_percent": 100.0 * y if np.isfinite(y) else np.nan
            })

    ytm_df = pd.DataFrame(ytm_rows)
    ytm_df.to_csv(BASE_DIR / "ytm_selected_by_day.csv", index=False)

    # Plot daily yield curves
    grid = np.linspace(0.0, 5.0, 101)
    plt.figure(figsize=(10, 6))

    for d in sorted(ytm_df["date"].unique()):
        sub = ytm_df[ytm_df["date"] == d].dropna(subset=["ytm_percent", "ttm_years"]).copy()
        sub = sub.sort_values("ttm_years")

        x = sub["ttm_years"].to_numpy()
        y = sub["ytm_percent"].to_numpy()
        mask = x > 0
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue

        y_grid = np.interp(grid, x, y, left=y[0], right=y[-1])
        plt.plot(grid, y_grid, label=d)

    plt.xlabel("Maturity in years")
    plt.ylabel("Yield to maturity percent")
    plt.title("Daily 0 to 5 year YTM curves from the selected ten bonds")
    plt.xlim(0, 5)
    plt.grid(True, linewidth=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ytm_curves.png", dpi=200)
    plt.close()

    print("OK")
    print(f"Loaded: {PRICE_CSV.name} rows={prices.shape[0]} date_cols={len(date_cols)}")
    print(f"Loaded: {META_CSV.name} rows={meta.shape[0]}")
    print("Saved: selected_bonds.csv")
    print("Saved: ytm_selected_by_day.csv")
    print("Saved: ytm_curves.png")


if __name__ == "__main__":
    main()
