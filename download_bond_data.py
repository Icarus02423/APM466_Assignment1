import re
import json
import time
import html as ihtml
from datetime import date
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


BASE = "https://markets.businessinsider.com"

FINDER_URLS = [
    "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=shortterm&yield=&bondtype=2%2C3%2C4%2C16&coupon=&currency=184&rating=&country=19",
    "https://markets.businessinsider.com/bonds/finder?borrower=71&maturity=midterm&yield=&bondtype=2%2C3%2C4%2C16&coupon=&currency=184&rating=&country=19",
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE + "/",
    "Connection": "keep-alive",
}

DEBUG_FAIL_LIMIT = 3
SAVE_FAIL_HTML = True


def with_query(url: str, **params) -> str:
    u = urlparse(url)
    q = parse_qs(u.query)
    for k, v in params.items():
        q[k] = [str(v)]
    new_query = urlencode(q, doseq=True)
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_query, u.fragment))


def normalize_bond_url(url: str) -> str:
    return with_query(url, miRedirects=1)


def get_text(session: requests.Session, url: str, timeout=30, retries=4, backoff=1.6) -> str:
    last_exc = None
    for i in range(retries):
        try:
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                r.encoding = r.encoding or "utf-8"
                return r.text
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff ** (i + 1))
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(backoff ** (i + 1))
    raise RuntimeError(f"GET failed after retries: {url}\nlast={last_exc}")


def extract_bond_links_from_finder_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody")
    if not tbody:
        return []

    links = []
    for tr in tbody.find_all("tr"):
        a = tr.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if not href.startswith("/bonds/"):
            continue

        full = urljoin(BASE, href)
        full = normalize_bond_url(full)
        links.append(full)

    return links


def get_all_bond_links(session: requests.Session, finder_url: str, max_pages=200) -> list[str]:
    all_links = []
    seen = set()

    for p in range(1, max_pages + 1):
        page_url = with_query(finder_url, p=p)
        html = get_text(session, page_url)
        page_links = extract_bond_links_from_finder_html(html)

        if not page_links:
            break

        for link in page_links:
            if link not in seen:
                seen.add(link)
                all_links.append(link)

        time.sleep(0.15)

    return all_links


def extract_bond_fields_from_text(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    start = text.find("Bond Data")
    text2 = text[start:] if start != -1 else text

    for end_mark in ("More Bonds", "Moody’s", "Moody's", "About the", "About"):
        end = text2.find(end_mark)
        if end != -1 and end > 0:
            text2 = text2[:end]
            break

    def pick(pattern: str) -> str | None:
        m = re.search(pattern, text2, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    isin = pick(r"\bISIN\s+([A-Z0-9]{12})\b")
    issue_date = pick(r"\bIssue Date\s+(\d{1,2}/\d{1,2}/\d{4})\b")
    maturity_date = pick(r"\bMaturity Date\s+(\d{1,2}/\d{1,2}/\d{4})\b")
    coupon_num = pick(r"\bCoupon\s+([0-9]+(?:[.,][0-9]+)?)\s*%")
    coupon = (coupon_num + "%") if coupon_num else None

    if not isin:
        raise ValueError("ISIN not found in page text (likely blocked/redirected page).")

    return {
        "ISIN": isin,
        "Issue Date": issue_date,
        "Maturity Date": maturity_date,
        "Coupon": coupon,
    }


def extract_tkdata_from_html(html: str) -> str | None:
    h = ihtml.unescape(html)

    patterns = [
        r'"TKData"\s*:\s*"([^"]+)"',
        r'"tkData"\s*:\s*"([^"]+)"',
        r'\bTKData\b\s*[:=]\s*"([^"]+)"',
        r'\btkData\b\s*[:=]\s*"([^"]+)"',
        r'(?i)\btkdata\b[^0-9]{0,40}([0-9]+(?:,[0-9]+){2,})',
    ]

    for pat in patterns:
        m = re.search(pat, h)
        if m:
            return m.group(1).strip().replace(" ", "")

    return None


def snapshot_info(session: requests.Session, bond_url: str) -> dict:
    bond_url = normalize_bond_url(bond_url)
    html = get_text(session, bond_url)

    info = extract_bond_fields_from_text(html)
    tk = extract_tkdata_from_html(html)
    if tk:
        info["TKData"] = tk

    info["url"] = bond_url
    return info


def parse_date_any(s: str | None) -> pd.Timestamp:
    if not s:
        return pd.NaT
    return pd.to_datetime(s, errors="coerce")


def load_chart_data(session: requests.Session, bond_url: str, tkdata: str, start: date, end: date) -> pd.DataFrame:
    bond_url = normalize_bond_url(bond_url)

    params = {
        "instrumentType": "Bond",
        "tkData": tkdata,
        "from": start.strftime("%Y%m%d"),
        "to": end.strftime("%Y%m%d"),
    }
    url = f"{BASE}/Ajax/Chart_GetChartData?{urlencode(params)}"
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/plain, */*",
        "Referer": bond_url,
        "Connection": "keep-alive",
    }

    r = session.get(url, headers=headers, timeout=30)
    r.raise_for_status()

    raw = r.content
    data = None

    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            data = json.loads(text)
            break
        except Exception:
            continue

    if data is None:
        text = raw.decode("utf-8", errors="ignore").strip()
        if text.startswith("[{") and text.endswith("}]"):
            pieces = text[2:-2].split("},{")
            data = [json.loads("{" + p + "}") for p in pieces]
        else:
            raise RuntimeError("Cannot parse Chart_GetChartData response as JSON.")

    df = pd.DataFrame(data)
    if df.empty:
        return df

    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date", "datetime", "timestamp", "time"):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break
    if date_col is None:
        raise RuntimeError(f"Cannot find date column in chart data. cols={list(df.columns)}")

    s = df[date_col]
    if s.dtype == object:
        extracted = s.astype(str).str.extract(r"Date\((\d+)\)")[0]
        if extracted.notna().any():
            ms = pd.to_numeric(extracted, errors="coerce")
            df["date"] = pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert(None).dt.date
        else:
            df["date"] = pd.to_datetime(s, errors="coerce").dt.date
    else:
        mx = pd.to_numeric(s, errors="coerce").max()
        if mx and mx > 1e12:
            df["date"] = pd.to_datetime(s, unit="ms", utc=True).dt.tz_convert(None).dt.date
        elif mx and mx > 1e9:
            df["date"] = pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(None).dt.date
        else:
            df["date"] = pd.to_datetime(s, errors="coerce").dt.date

    price_col = None
    candidates = ["Close", "close", "Last", "last", "Price", "price", "Value", "value", "ClosePrice", "closePrice", "c"]
    for c in candidates:
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        for c in df.columns:
            if "close" in str(c).lower():
                price_col = c
                break
    if price_col is None:
        raise RuntimeError(f"Cannot find price/close column. cols={list(df.columns)}")

    out = df[["date", price_col]].copy()
    out = out.rename(columns={price_col: "close"})
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.groupby("date", as_index=False).tail(1).reset_index(drop=True)
    return out


def main():
    START = date(2026, 1, 5)
    END = date(2026, 1, 19)

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    all_links = []
    for fu in FINDER_URLS:
        all_links.extend(get_all_bond_links(session, fu))

    uniq_links = []
    seen = set()
    for u in all_links:
        if u not in seen:
            seen.add(u)
            uniq_links.append(u)

    print(f"Total bond pages collected from finder: {len(uniq_links)}")
    if uniq_links:
        print("Sample bond url:", uniq_links[0])

    meta_rows = []
    prices_rows = []
    fail_count = 0

    for url in tqdm(uniq_links, desc="Scraping bonds"):
        try:
            info = snapshot_info(session, url)

            isin = info.get("ISIN")
            coupon = info.get("Coupon")
            issue_date = parse_date_any(info.get("Issue Date"))
            maturity_date = parse_date_any(info.get("Maturity Date"))
            tkdata = info.get("TKData")

            meta_rows.append({
                "url": url,
                "ISIN": isin,
                "Coupon": coupon,
                "IssueDate": issue_date,
                "MaturityDate": maturity_date,
                "TKData": tkdata,
            })

            if not tkdata:
                continue

            dfp = load_chart_data(session, url, tkdata, START, END)
            if not dfp.empty:
                dfp["ISIN"] = isin
                prices_rows.append(dfp)

            time.sleep(0.25)

        except Exception as e:
            fail_count += 1
            print(f"[WARN] failed: {url} -> {e}")

            if fail_count <= DEBUG_FAIL_LIMIT:
                try:
                    html = get_text(session, normalize_bond_url(url))
                    head = html[:200].replace("\n", " ").replace("\r", " ")
                    print("  HTML head:", head)

                    if SAVE_FAIL_HTML:
                        fn = f"debug_fail_{fail_count}.html"
                        with open(fn, "w", encoding="utf-8") as f:
                            f.write(html)
                        print(f"  saved: {fn}")
                except Exception as e2:
                    print("  also failed to fetch html:", e2)

            time.sleep(0.35)


    meta_cols = ["url", "ISIN", "Coupon", "IssueDate", "MaturityDate", "TKData"]
    meta_df = pd.DataFrame(meta_rows, columns=meta_cols)
    meta_df.to_csv("bonds_meta.csv", index=False)


    if prices_rows:
        prices_df = pd.concat(prices_rows, ignore_index=True)

        matrix = prices_df.pivot_table(
            index="ISIN",
            columns="date",
            values="close",
            aggfunc="last"
        )

        matrix = matrix.sort_index()
        matrix = matrix.reindex(sorted(matrix.columns), axis=1)

        matrix.columns = [pd.to_datetime(c).strftime("%Y-%m-%d") for c in matrix.columns]

        out_xlsx = "bonds_prices_41x11.xlsx"
        matrix.to_csv("bonds_prices_41x11.csv", index=True)

        print(f"Saved: {out_xlsx}  shape={matrix.shape}")
    else:
        print("No price data collected. Check debug_fail_*.html for blocking/redirects.")

    print("Saved: bonds_meta.csv")
    print(f"Done. meta_rows={len(meta_rows)} fails={fail_count}")


if __name__ == "__main__":
    main()
