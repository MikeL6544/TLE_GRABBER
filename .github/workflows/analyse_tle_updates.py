import glob
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

def parse_tle_epoch(line1: str) -> datetime:
    """
    Parse the epoch timestamp from a TLE line 1:
    cols 19–20: 2‑digit year, cols 21–32: day‑of‑year.fraction
    """
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(line1[20:32])
    return datetime(year, 1, 1) + timedelta(days=doy - 1)

def parse_file_time(path: str) -> datetime:
    """
    Parse the datetime from a filename like:
      tle_archive/tle-2025-04-16T12-00-00Z.txt
    Returns a tz‑naive datetime.
    """
    fname = path.rsplit("/", 1)[-1]
    m = re.match(r"tle-(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z\.txt", fname)
    if not m:
        raise ValueError(f"Bad filename format: {fname}")
    ts = m.group(1)  # e.g. "2025-04-16T12-00-00"
    return datetime.strptime(ts, "%Y-%m-%dT%H-%M-%S")

def gather_epochs(sat_id: str, archive_glob: str = "tle_archive/*.txt") -> pd.DataFrame:
    """
    Reads all TLE snapshots, finds the lines for `sat_id`, and returns a DataFrame:
      file_path, file_time, epoch, delta, latency
    """
    records = []
    for path in sorted(glob.glob(archive_glob)):
        for ln in open(path).read().splitlines():
            if ln.startswith("1 " + sat_id):
                epoch     = parse_tle_epoch(ln)
                file_time = parse_file_time(path)
                records.append({
                    "file_path": path,
                    "file_time": file_time,
                    "epoch": epoch
                })
                break

    df = pd.DataFrame(records)
    df = df.sort_values("file_path").reset_index(drop=True)
    df["delta"]   = df["epoch"].diff()
    df["latency"] = df["file_time"] - df["epoch"]
    return df

if __name__ == "__main__":
    # ← Set this to any NORAD ID you want to analyze
    SAT_ID = "52935"  # DS‑EO

    # 1) Gather all snapshots into df
    df = gather_epochs(SAT_ID)

    # 2) Drop everything up through (and including) the first real update
    mask = df["delta"] > pd.Timedelta(0)
    if mask.any():
        first_idx = mask.idxmax()
        df = df.loc[first_idx+1 :].reset_index(drop=True)

    # 3) Compute the span of collected data (post-first-update)
    first_time = df["file_time"].iloc[0]
    last_time  = df["file_time"].iloc[-1]
    total_span = last_time - first_time
    # Round up to whole days
    span_days = int((total_span.total_seconds() + 86399) // 86400)

    # 4) Filter non-zero deltas for stats and distribution
    nz = df["delta"].dropna()
    nonzero = nz[nz > pd.Timedelta(0)]

    # Summary stats
    shortest = nonzero.min()
    average  = nonzero.mean()
    longest  = nonzero.max()

    # 5) Build 3‑hour interval distribution up to 48h + overflow
    hours     = nonzero.dt.total_seconds() / 3600.0
    bin_edges = list(range(0, 49, 3)) + [np.inf]
    labels    = [f"{i}-{i+3}h" for i in range(0, 48, 3)] + ["48h+"]
    cats      = pd.cut(hours, bins=bin_edges, labels=labels, right=False)
    dist_df   = cats.value_counts().sort_index().reset_index()
    dist_df.columns = ["Interval", "Count"]

    # 6) Table of actual updates (only where delta > 0)
    updates_df = df[df["delta"] > pd.Timedelta(0)][
        ["file_path", "file_time", "epoch", "latency"]
    ]

    # 7) Render Markdown report
    header      = f"# TLE Update Report for SAT {SAT_ID}\n\n"
    snap_table  = df.to_markdown(index=False)
    summary_md   = (
        "## Summary of update intervals\n\n"
        f"- **Data span**: {span_days} days (from {first_time.date()} to {last_time.date()})\n"
        f"- **Shortest** : {shortest}\n"
        f"- **Average**  : {average}\n"
        f"- **Longest**  : {longest}\n"
    )
    dist_md     = dist_df.to_markdown(index=False)
    updates_md  = updates_df.to_markdown(index=False)

    report = (
        header
        + "## Per‑snapshot epochs & intervals\n\n" + snap_table
        + "\n\n" + summary_md
        + "\n\n## Distribution of update intervals (3 h buckets)\n\n" + dist_md
        + "\n\n## Actual updates and latency\n\n"
          "Rows here are only when a new TLE appeared; `latency` shows how long\n"
          "after the TLE epoch it was fetched:\n\n"
        + updates_md
        + "\n"
    )

    # 8) Write the report file
    with open("tle_update_report.md", "w") as f:
        f.write(report)

    print("→ Written tle_update_report.md")
