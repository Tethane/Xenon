#!/usr/bin/env python3
"""Summarize a Xenon Nsight Systems SQLite export within rendering only.

The window spans the first ray-generation kernel through the last accumulation
kernel. GPU busy time is the union of kernel, copy and memset intervals; gaps
are uncovered intervals, not time attributed to a particular CPU API call.
"""

import argparse, json, sqlite3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("database", help="SQLite produced by nsys export --type sqlite")
args = parser.parse_args()
con = sqlite3.connect(args.database)
con.row_factory = sqlite3.Row
kernels = list(
	con.execute(
		"select k.start,k.end,s.value name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id"
	)
)
begin = min(r["start"] for r in kernels if "raygen_kernel" in r["name"])
end = max(r["end"] for r in kernels if "accumulate_kernel" in r["name"])
intervals = []
totals = {}
for t in ["KERNEL", "MEMCPY", "MEMSET"]:
	for r in con.execute(
		"select start,end from CUPTI_ACTIVITY_KIND_" + t + " where end>? and start<?", (begin, end)
	):
		intervals.append((max(begin, r["start"]), min(end, r["end"])))
for r in kernels:
	if r["start"] < begin or r["end"] > end:
		continue
	key = r["name"].split("(")[0]
	item = totals.setdefault(key, {"count": 0, "seconds": 0})
	item["count"] += 1
	item["seconds"] += (r["end"] - r["start"]) * 1e-9
busy = 0
previous = begin
for a, b in sorted(intervals):
	busy += max(0, b - max(a, previous))
	previous = max(b, previous)
calls = {}
for name, count, total in con.execute(
	"select s.value,count(*),sum(r.end-r.start) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.start>=? and r.end<=? group by s.value",
	(begin, end),
):
	calls[name] = {"count": count, "seconds": total * 1e-9}
report = {
	"window_seconds": (end - begin) * 1e-9,
	"gpu_busy_seconds": busy * 1e-9,
	"gpu_gap_seconds": (end - begin - busy) * 1e-9,
	"gpu_busy_fraction": busy / (end - begin),
	"kernels": totals,
	"api": calls,
}
print(json.dumps(report, indent=2))
