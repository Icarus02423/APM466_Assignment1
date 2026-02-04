import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Files
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
PRICE_CSV = BASE_DIR / "bonds_prices_41x11.csv"
META_CSV = BASE_DIR / "bonds_meta.csv"

MAX_DAYS = 10

FACE = 100.0
FREQ = 2

PREFERRED_ISINS = [
    "CA135087L518",
    "CA135087L930",
    "CA135087M847",
    "CA135087N837",
    "CA135087P576",
    "CA135087Q491",
    "CA135087Q988",
    "CA135087R895",
    "CA135087S471",
    "CA135087T388",
]


# Column tools
def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        n = _norm(c)
        if n == "isin":
            mapping[c] = "ISIN"
        elif n in ["coupon", "couponrate", "coupon%"]:
            mapping[c] = "Coupon"
        elif n in ["issuedate", "issuedate:", "issuedate(utc)", "issuedateutc", "issue_date", "issuedate"]:
            mapping[c] = "IssueDate"
        elif n in ["maturitydate", "maturitydate:", "maturity_date", "maturity"]:
            mapping[c] = "MaturityDate"
        elif n in ["tkdata", "tk"]:
            mapping[c] = "TKData"
        elif n in ["url", "link"]:
            mapping[c] = "url"
    return df.rename(columns=mapping)


def coalesce(df: pd.DataFrame, base: str, alt: str) -> pd.DataFrame:
    if base in df.columns and alt in df.columns:
        a = df[base].replace("", np.nan)
        b = df[alt].replace("", np.nan)
        df[base] = a.where(a.notna(), b)
        df = df.drop(columns=[alt])
        return df
    if base not in df.columns and alt in df.columns:
        return df.rename(columns={alt: base})
    return df


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    reserved = {"ISIN", "Coupon", "IssueDate", "MaturityDate", "TKData", "url", "CouponDec", "BondName"}
    date_cols = []
    for c in df.columns:
        if c in reserved:
            continue
        dt = pd.to_datetime(str(c).strip(), errors="coerce")
        if pd.notna(dt):
            date_cols.append(c)
    date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(str(x), errors="coerce"))
    if len(date_cols) == 0:
        raise ValueError(f"No date columns detected in {PRICE_CSV.name}")
    if len(date_cols) > MAX_DAYS:
        date_cols = date_cols[:MAX_DAYS]
    return date_cols


# Bond math
def parse_coupon_to_decimal(x) -> float:
    s = str(x).strip().replace("%", "").replace(",", ".")
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


def cashflows_after_settle(settle: pd.Timestamp, maturity: pd.Timestamp, coupon_dec: float) -> list[tuple[pd.Timestamp, float]]:
    schedule = generate_coupon_schedule(maturity)
    settle = pd.Timestamp(settle).normalize()
    maturity = pd.Timestamp(maturity).normalize()

    c_per = FACE * coupon_dec / FREQ
    flows = []
    for d in schedule:
        d = pd.Timestamp(d).normalize()
        if d <= settle:
            continue
        if d < maturity:
            flows.append((d, c_per))
        elif d == maturity:
            flows.append((d, c_per + FACE))
    return flows


def pv_from_yield(settle: pd.Timestamp, flows: list[tuple[pd.Timestamp, float]], y: float) -> float:
    settle = pd.Timestamp(settle).normalize()
    pv = 0.0
    for d, cf in flows:
        t = (pd.Timestamp(d).normalize() - settle).days / 365.0
        if t <= 0:
            continue
        disc = (1.0 + y / FREQ) ** (-FREQ * t)
        pv += cf * disc
    return pv


def solve_ytm(clean_price: float, settle: pd.Timestamp, maturity: pd.Timestamp, coupon_dec: float) -> float:
    if not np.isfinite(clean_price):
        return np.nan

    ai = accrued_interest(settle, maturity, coupon_dec)
    dirty = float(clean_price) + ai
    flows = cashflows_after_settle(settle, maturity, coupon_dec)

    def f(y):
        return pv_from_yield(settle, flows, y) - dirty

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


# Selection logic
def auto_select_ladder(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(f"Could not form a 10-bond ladder, only found {tmp.shape[0]}")
    return tmp


def select_ten_bonds(df: pd.DataFrame) -> pd.DataFrame:
    have = set(df["ISIN"].astype(str))
    if all(x in have for x in PREFERRED_ISINS):
        sel = df[df["ISIN"].isin(PREFERRED_ISINS)].copy()
        return sel.sort_values("MaturityDate").reset_index(drop=True)
    return auto_select_ladder(df).sort_values("MaturityDate").reset_index(drop=True)


# Bootstrapping spot curve
def bootstrap_discount_factors_for_day(settle: pd.Timestamp, sel: pd.DataFrame, price_col: str) -> pd.DataFrame:
    settle = pd.Timestamp(settle).normalize()

    # Build cashflows for each bond and union of payment dates
    bond_info = {}
    all_nodes = set()

    for _, r in sel.iterrows():
        isin = r["ISIN"]
        maturity = pd.Timestamp(r["MaturityDate"]).normalize()
        c = float(r["CouponDec"])
        clean = float(r[price_col])
        ai = accrued_interest(settle, maturity, c)
        dirty = clean + ai
        flows = cashflows_after_settle(settle, maturity, c)

        bond_info[isin] = {
            "maturity": maturity,
            "coupon": c,
            "dirty": dirty,
            "flows": flows,
        }
        for d, _ in flows:
            all_nodes.add(pd.Timestamp(d).normalize())

    nodes = sorted(all_nodes)
    # Map maturity date to which bond matures there
    maturity_to_isin = {}
    for isin, info in bond_info.items():
        maturity_to_isin[info["maturity"]] = isin

    D = {}
    rows = []

    for node in nodes:
        if node not in maturity_to_isin:
            # With the chosen ladder this should not happen
            continue

        isin = maturity_to_isin[node]
        info = bond_info[isin]
        dirty = info["dirty"]
        flows = info["flows"]

        pv_known = 0.0
        cf_node = None

        for d, cf in flows:
            d = pd.Timestamp(d).normalize()
            if d < node:
                if d not in D:
                    raise RuntimeError(f"Missing discount factor for {d.date()} while bootstrapping {node.date()}")
                pv_known += cf * D[d]
            elif d == node:
                cf_node = cf

        if cf_node is None:
            raise RuntimeError(f"No maturity cashflow found for bond {isin} at {node.date()}")

        D_node = (dirty - pv_known) / cf_node
        D[node] = D_node

        t = (node - settle).days / 365.0
        spot = (D_node ** (-1.0 / t) - 1.0) if t > 0 else 0.0

        rows.append({
            "node_date": node.strftime("%Y-%m-%d"),
            "t_years": t,
            "discount": D_node,
            "spot_decimal": spot,
            "spot_percent": 100.0 * spot,
        })

    out = pd.DataFrame(rows).sort_values("t_years")
    return out


def discount_at_time(t: float, t_nodes: np.ndarray, lnD_nodes: np.ndarray) -> float:
    # log-linear interpolation, and log-linear extrapolation using last segment
    if t <= t_nodes[-1]:
        lnD = np.interp(t, t_nodes, lnD_nodes)
    else:
        slope = (lnD_nodes[-1] - lnD_nodes[-2]) / (t_nodes[-1] - t_nodes[-2])
        lnD = lnD_nodes[-1] + slope * (t - t_nodes[-1])
    return float(np.exp(lnD))


def spot_from_discount(D: float, t: float) -> float:
    if t <= 0:
        return 0.0
    return D ** (-1.0 / t) - 1.0


def spot_curve_on_grid(boot: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    t_nodes = np.array([0.0] + boot["t_years"].tolist())
    D_nodes = np.array([1.0] + boot["discount"].tolist())
    lnD_nodes = np.log(D_nodes)
    spots = []
    for t in grid:
        if t <= 0:
            spots.append(0.0)
            continue
        D_t = discount_at_time(t, t_nodes, lnD_nodes)
        spots.append(100.0 * spot_from_discount(D_t, t))
    return np.array(spots)


def spot_points_1to5y(boot: pd.DataFrame) -> dict:
    t_nodes = np.array([0.0] + boot["t_years"].tolist())
    D_nodes = np.array([1.0] + boot["discount"].tolist())
    lnD_nodes = np.log(D_nodes)

    out = {}
    for T in [1, 2, 3, 4, 5]:
        D_T = discount_at_time(float(T), t_nodes, lnD_nodes)
        S_T = spot_from_discount(D_T, float(T))
        out[T] = S_T
    return out


def forward_1y_ny_from_spots(spots_1to5: dict) -> dict:
    # Discrete annual compounding, matches the hint formula
    S1 = spots_1to5[1]
    out = {}
    for n in [1, 2, 3, 4]:
        ST = spots_1to5[1 + n]
        f = ((1.0 + ST) ** (1 + n) / (1.0 + S1) ** 1.0) ** (1.0 / n) - 1.0
        out[n] = f
    return out


# Main pipeline
def main():
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Missing {PRICE_CSV.name} in {BASE_DIR}")

    prices = pd.read_csv(PRICE_CSV)
    prices = canonicalize_columns(prices)

    if "ISIN" not in prices.columns:
        prices = prices.rename(columns={prices.columns[0]: "ISIN"})
    prices["ISIN"] = prices["ISIN"].astype(str).str.strip()

    # Identify date columns from prices
    date_cols = detect_date_columns(prices)

    # Ensure numeric prices
    for c in date_cols:
        prices[c] = pd.to_numeric(prices[c], errors="coerce")

    # If metadata columns missing, try meta csv
    need_meta = any(col not in prices.columns for col in ["Coupon", "IssueDate", "MaturityDate"])
    if need_meta:
        if not META_CSV.exists():
            raise FileNotFoundError(
                "Price file is missing Coupon IssueDate MaturityDate and bonds_meta.csv is not present"
            )
        meta = pd.read_csv(META_CSV)
        meta = canonicalize_columns(meta)
        if "ISIN" not in meta.columns:
            meta = meta.rename(columns={meta.columns[0]: "ISIN"})
        meta["ISIN"] = meta["ISIN"].astype(str).str.strip()

        df = prices.merge(meta, on="ISIN", how="left", suffixes=("", "_meta"))
        for col in ["Coupon", "IssueDate", "MaturityDate", "TKData", "url"]:
            df = coalesce(df, col, f"{col}_meta")
    else:
        df = prices.copy()

    # Parse meta types
    df["IssueDate"] = pd.to_datetime(df["IssueDate"], errors="coerce")
    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"], errors="coerce")
    df["CouponDec"] = df["Coupon"].apply(parse_coupon_to_decimal)

    # Select ten bonds
    sel = select_ten_bonds(df)
    sel["BondName"] = sel.apply(lambda r: format_bond_name(float(r["CouponDec"]), r["MaturityDate"]), axis=1)

    selected_out = sel[["BondName", "ISIN", "Coupon", "IssueDate", "MaturityDate"]].copy()
    if "TKData" in sel.columns:
        selected_out["TKData"] = sel["TKData"]
    selected_out["IssueDate"] = selected_out["IssueDate"].dt.strftime("%Y-%m-%d")
    selected_out["MaturityDate"] = selected_out["MaturityDate"].dt.strftime("%Y-%m-%d")
    selected_out.to_csv(BASE_DIR / "selected_bonds.csv", index=False)

    # Q4a: YTM per bond per day and daily yield curves
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

            clean = float(r[col])
            y = solve_ytm(clean, settle, maturity, float(r["CouponDec"]))
            ttm = (maturity - settle).days / 365.0

            ytm_rows.append({
                "date": settle.strftime("%Y-%m-%d"),
                "ISIN": r["ISIN"],
                "BondName": r["BondName"],
                "ttm_years": ttm,
                "ytm_decimal": y,
                "ytm_percent": 100.0 * y if np.isfinite(y) else np.nan,
            })

    ytm_df = pd.DataFrame(ytm_rows)

    # Interpolate yields at 1 to 5 years for each day
    yield_points = []
    grid_y = np.linspace(0.0, 5.0, 101)

    plt.figure(figsize=(10, 6))
    for d in sorted(ytm_df["date"].unique()):
        sub = ytm_df[ytm_df["date"] == d].dropna(subset=["ttm_years", "ytm_percent"]).copy()
        sub = sub.sort_values("ttm_years")
        x = sub["ttm_years"].to_numpy()
        y = sub["ytm_percent"].to_numpy()

        if len(x) < 2:
            continue

        y_grid = np.interp(grid_y, x, y, left=y[0], right=y[-1])
        plt.plot(grid_y, y_grid, label=d)

        # Store 1..5y points in decimals
        for T in [1, 2, 3, 4, 5]:
            y_T = np.interp(float(T), x, y, left=y[0], right=y[-1]) / 100.0
            yield_points.append({"date": d, "maturity_years": T, "yield_decimal": y_T, "yield_percent": 100.0 * y_T})

    plt.xlabel("Maturity in years")
    plt.ylabel("Yield to maturity percent")
    plt.title("Daily 0 to 5 year yield curves from the selected ten bonds")
    plt.xlim(0, 5)
    plt.grid(True, linewidth=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "ytm_curves.png", dpi=200)
    plt.close()

    yields_1to5 = pd.DataFrame(yield_points).pivot(index="date", columns="maturity_years", values="yield_decimal").sort_index()
    yields_1to5.columns = [f"Y{int(c)}" for c in yields_1to5.columns]
    yields_1to5.to_csv(BASE_DIR / "yields_1to5.csv", index=True)

    # Q4b: Spot curve bootstrapped each day, plus spot curves plot
    spot_grid = np.linspace(1.0, 5.0, 81)
    spots_1to5_rows = []

    plt.figure(figsize=(10, 6))
    for col in date_cols:
        settle = pd.to_datetime(str(col), errors="coerce")
        if pd.isna(settle):
            continue
        settle = settle.normalize()
        date_str = settle.strftime("%Y-%m-%d")

        boot = bootstrap_discount_factors_for_day(settle, sel, col)
        s_grid = spot_curve_on_grid(boot, spot_grid)
        plt.plot(spot_grid, s_grid, label=date_str)

        pts = spot_points_1to5y(boot)
        row = {"date": date_str}
        for T in [1, 2, 3, 4, 5]:
            row[f"S{T}"] = pts[T]
        spots_1to5_rows.append(row)

    plt.xlabel("Maturity in years")
    plt.ylabel("Spot rate percent")
    plt.title("Daily 1 to 5 year spot curves by bootstrapping")
    plt.xlim(1, 5)
    plt.grid(True, linewidth=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "spot_curves.png", dpi=200)
    plt.close()

    spots_1to5 = pd.DataFrame(spots_1to5_rows).set_index("date").sort_index()
    spots_1to5.to_csv(BASE_DIR / "spots_1to5.csv", index=True)

    # Q4c: 1-year forward curve from 2 to 5 years
    forwards_rows = []
    x_forward = np.array([2, 3, 4, 5])

    plt.figure(figsize=(10, 6))
    for d, row in spots_1to5.iterrows():
        pts = {1: row["S1"], 2: row["S2"], 3: row["S3"], 4: row["S4"], 5: row["S5"]}
        f = forward_1y_ny_from_spots(pts)  # keys are n = 1..4

        forwards_rows.append({
            "date": d,
            "F1_1": f[1],
            "F1_2": f[2],
            "F1_3": f[3],
            "F1_4": f[4],
        })

        y_forward = np.array([100.0 * f[1], 100.0 * f[2], 100.0 * f[3], 100.0 * f[4]])
        plt.plot(x_forward, y_forward, marker="o", label=d)

    plt.xlabel("End of forward period in years")
    plt.ylabel("Forward rate percent")
    plt.title("Daily 1-year forward curve from year 2 to year 5")
    plt.xlim(2, 5)
    plt.grid(True, linewidth=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "forward_curves.png", dpi=200)
    plt.close()

    forwards = pd.DataFrame(forwards_rows).set_index("date").sort_index()
    forwards.to_csv(BASE_DIR / "forwards_1y_ny.csv", index=True)

    # Q5: Covariance matrices of log returns
    def log_returns_matrix(level_df: pd.DataFrame) -> pd.DataFrame:
        # level_df index is date in ascending order, values are rates in decimal
        arr = level_df.to_numpy(dtype=float)
        if np.any(arr <= 0):
            raise ValueError("Found non-positive rates, log returns undefined for the ratio in the assignment formula")
        rets = np.log(arr[1:, :] / arr[:-1, :])
        out = pd.DataFrame(rets, columns=level_df.columns)
        return out

    y_rets = log_returns_matrix(yields_1to5)
    f_rets = log_returns_matrix(forwards[["F1_1", "F1_2", "F1_3", "F1_4"]])

    cov_y = pd.DataFrame(np.cov(y_rets.to_numpy(), rowvar=False, ddof=1),
                         index=y_rets.columns, columns=y_rets.columns)
    cov_f = pd.DataFrame(np.cov(f_rets.to_numpy(), rowvar=False, ddof=1),
                         index=f_rets.columns, columns=f_rets.columns)

    cov_y.to_csv(BASE_DIR / "cov_yields.csv", index=True)
    cov_f.to_csv(BASE_DIR / "cov_forwards.csv", index=True)

    # Q6: Eigenvalues and eigenvectors
    def eig_sorted(cov: pd.DataFrame):
        vals, vecs = np.linalg.eigh(cov.to_numpy(dtype=float))
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        vals_s = pd.Series(vals, index=[f"PC{k+1}" for k in range(len(vals))])
        vecs_df = pd.DataFrame(vecs, index=cov.index, columns=vals_s.index)
        return vals_s, vecs_df

    eig_y_vals, eig_y_vecs = eig_sorted(cov_y)
    eig_f_vals, eig_f_vecs = eig_sorted(cov_f)

    eig_y_vals.to_csv(BASE_DIR / "eig_yields_values.csv")
    eig_y_vecs.to_csv(BASE_DIR / "eig_yields_vectors.csv", index=True)

    eig_f_vals.to_csv(BASE_DIR / "eig_forwards_values.csv")
    eig_f_vecs.to_csv(BASE_DIR / "eig_forwards_vectors.csv", index=True)

    print("Finished")
    print("Outputs written to:")
    print(f"  {BASE_DIR}")
    print("Key files:")
    print("  ytm_curves.png")
    print("  spot_curves.png")
    print("  forward_curves.png")
    print("  cov_yields.csv")
    print("  cov_forwards.csv")
    print("  eig_yields_values.csv  eig_yields_vectors.csv")
    print("  eig_forwards_values.csv  eig_forwards_vectors.csv")


if __name__ == "__main__":
    main()
