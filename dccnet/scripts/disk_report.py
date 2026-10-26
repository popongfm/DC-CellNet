"""Disk usage report (standard library only).

Usage: python disk_report.py <root> [--top 40] [--out report.md]

Lists the N largest files, total size per top-level folder, and every
"X.zip" that sits next to a folder "X" (likely duplicate archive).
Read-only: never deletes or moves anything.
"""
import argparse
import os
import sys


def fmt(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    files = []
    per_top = {}
    for dp, dn, fn in os.walk(root):
        if os.path.relpath(dp, root).split(os.sep)[0] in (".git", ".sync"):
            dn[:] = []
            continue
        for f in fn:
            p = os.path.join(dp, f)
            try:
                s = os.path.getsize(p)
            except OSError:
                continue
            rel = os.path.relpath(p, root)
            files.append((s, rel))
            top = rel.split(os.sep)[0] if os.sep in rel else "(root files)"
            per_top[top] = per_top.get(top, 0) + s

    files.sort(reverse=True)
    out = [f"# Disk report: {root}", "", f"Total: {fmt(sum(s for s, _ in files))} in {len(files)} files", ""]
    out += [f"## Top {a.top} largest files", "", "| Size | Path |", "|---|---|"]
    out += [f"| {fmt(s)} | {r} |" for s, r in files[: a.top]]
    out += ["", "## Size per top-level folder", "", "| Size | Folder |", "|---|---|"]
    out += [f"| {fmt(s)} | {t} |" for t, s in sorted(per_top.items(), key=lambda kv: -kv[1])]

    out += ["", "## Zip next to same-named folder (likely duplicate archive)", "",
            "| Zip | Zip size | Folder size |", "|---|---|---|"]
    sizes = {}
    for s, r in files:
        sizes[r] = s
    found = False
    for dp, dn, fn in os.walk(root):
        if os.path.relpath(dp, root).split(os.sep)[0] in (".git", ".sync"):
            dn[:] = []
            continue
        for f in fn:
            if f.lower().endswith(".zip") and f[:-4] in dn:
                zp = os.path.join(dp, f)
                fp = os.path.join(dp, f[:-4])
                fs = sum(os.path.getsize(os.path.join(d, x)) for d, _, xs in os.walk(fp) for x in xs
                         if os.path.exists(os.path.join(d, x)))
                out.append(f"| {os.path.relpath(zp, root)} | {fmt(os.path.getsize(zp))} | {fmt(fs)} |")
                found = True
    if not found:
        out.append("| (none) | | |")

    text = "\n".join(out) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
