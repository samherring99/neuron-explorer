#!/usr/bin/env python
"""Example behavior hunt: DISCOVER a neuron for a behavior, then VERIFY and DOSE it.

A worked, self-contained template. It reproduces the cheese hunt on Qwen3.5-4B end to end:

  1. NOMINATE — logit-lens search (`/api/search`) over several word forms for the concept, keeping
                only word-boundary matches inside the neuron's lens column.
  2. SELECT   — rank the candidates by FIRING RATIO: `/api/fires` on a concept-rich sentence vs an
                unrelated control sentence. Lens nomination over-fires (and `/api/search` has no
                usable score field, so ranking by it silently degenerates to LAYER ORDER — which
                picks inert early-layer units). The firing ratio is the cheap functional filter.
  3. VERIFY   — steer the top candidates at BOTH poles, diffed against the unsteered baseline of the
                same prompt, with a degeneracy flag. Only steering separates a real handle from a
                coincidental lens token.
  4. WINDOW   — dose-sweep the winner to find its sensible window (above it you get word-spew).
  5. CONTROL  — a same-layer unrelated neuron at the winning dose (specificity check).

Result JSON: `examples/example_hunt.json`. Requires a running explorer, e.g.

    cd neuron-explorer
    CUDA_VISIBLE_DEVICES=1 EXPLORER_MODEL=/home/shared/models/Qwen3.5-4B PORT=8791 \
      .venv/bin/python explorer/server.py > /tmp/exp.log 2>&1 &
    EXPLORER_URL=http://127.0.0.1:8791 .venv/bin/python scripts/example_hunt.py

Steering recipe (implemented in explorer/runtime.py): add
`dose x median(||resid|| leaving layer L, tokens 1:) / ||W_down[:, n]||` at the neuron's post-gating
MLP activation (down_proj input) at every token. Dose 0 returns the baseline row, so no separate
baseline call is needed.

Adapt by editing the CONFIG block. Keep the discipline: both poles, baseline diff, control, and read
the text — a keyword count alone can be truncation, token spew, or baseline leakage.

Note on doses: high doses (>=1.25) confirm a unit's *lexical pole*; they are not working doses. The
winner search below therefore ignores rows above |1.0| and any row whose text is degenerate, so
"the keyword count went up" never crowns a spewing unit.
"""
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter

# ---------------------------------------------------------------------------------------------
# CONFIG — edit these for your own behavior
# ---------------------------------------------------------------------------------------------
CONCEPT = "cheese"
QUERIES = ["cheese", "cheddar", "dairy", "fromage"]   # lens search terms (substring match, >=2 chars)
MARKERS = (r"\bcheese\w*|\bcheddar\b|\bgouda\b|\bbrie\b|\bfromage\b|\bdairy\b|\bmozzarella\b|"
           r"\bparmesan\b|\bmilk\b|奶酪")
TARGET_TEXT = "The cheese board had cheddar, brie, gouda and a creamy blue cheese."
CONTROL_TEXT = "The quarterly report was submitted to the regional office on Tuesday."
PROMPTS = ["The best sandwich has", "My favorite food is", "For dinner I made"]
VERIFY_DOSES = [-1.0, -0.5, 0.5, 1.0]        # both poles first — the behavior may live negative
SWEEP_DOSES = [0.25, 0.5, 0.75, 1.25, 1.5]   # window sweep on the winner
CONTROL = (7, 5317)                          # same-layer unrelated neuron (L7, not a cheese unit)
TOP_N = 3                                    # candidates to verify causally
MAX_NEW = 90
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "examples", "example_hunt.json")
# ---------------------------------------------------------------------------------------------

URL = os.environ.get("EXPLORER_URL", "http://127.0.0.1:8791")
MARK = re.compile(MARKERS, re.I)
DEGEN = re.compile(r"(.)\1{4,}|\b(\w+)( \2){2,}")   # repetition loops = degeneration, not behavior


def post(path, body, timeout=900):
    req = urllib.request.Request(f"{URL}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path, timeout=900):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
        return json.load(r)


def steer(prompt, layer, n, dose, max_new=MAX_NEW):
    """One call returns up (pushed) / baseline (dose 0) / down (pushes inverted) rows."""
    return post("/api/agent/steer", {"prompt": prompt, "neurons": [{"layer": layer, "n": n, "dose": dose}],
                                     "dose": dose, "mode": "add", "max_new": max_new})


DEGEN_WORD_FRAC = 0.25   # one word dominating >=25% of a >=12-word output = spew, not behavior


def degenerate(text):
    """Token-spew detection: repetition loops, or a single word dominating the output.
    ('Cheaz! (Cheers! / Cheers! ...' and 'cheese... cheese... cheese...' are counting as markers
    otherwise — the most common way a keyword scan gets a false win.)"""
    if DEGEN.search(text):
        return True
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) >= 12 and max(Counter(words).values()) / len(words) > DEGEN_WORD_FRAC:
        return True
    return False


def score(text):
    return len(MARK.findall(text)), degenerate(text)


# ---------------------------------------------------------------- 1. NOMINATE
def nominate():
    hits = {}
    for q in QUERIES:
        res = get(f"/api/search?q={urllib.parse.quote(q)}&limit=400")
        for r in res.get("results", []):
            matched = [t for t in r.get("lens", []) if MARK.search(t)]   # word-boundary filter
            if not matched:        # drops substring false friends ("cheesecake" for "cheese",
                continue           # " category" for "cat")
            cand = hits.setdefault((r["layer"], r["n"]), {
                "layer": r["layer"], "n": r["n"], "signs": set(), "matched": set(),
                "count": 0, "rank": 99})
            cand["signs"].add(r["sign"])
            cand["matched"].update(t.strip() for t in matched)
            cand["count"] = max(cand["count"], len(matched))   # how many lens tokens match
            cand["rank"] = min(cand["rank"], int(r.get("rank", 99)))  # earliest match position
    out = sorted(hits.values(), key=lambda c: (-c["count"], c["rank"], c["layer"]))
    for c in out:
        c["signs"], c["matched"] = sorted(c["signs"]), sorted(c["matched"])[:6]
    return out


# ---------------------------------------------------------------- 2. SELECT by firing ratio
def firing_ratio(cands):
    """Rank nominated units by how much harder they fire on the concept text than on a control text.
    This is the cheap functional filter; single-sentence `fires` ranking is NOT (it surfaces
    position/token artifacts), so always compare two texts."""
    def fires(text):
        res = get(f"/api/fires?text={urllib.parse.quote(text)}&limit=300").get("results", [])
        return {(r["layer"], r["n"]): r["score"] for r in res}
    tgt, ctl = fires(TARGET_TEXT), fires(CONTROL_TEXT)
    for c in cands:
        k = (c["layer"], c["n"])
        c["fire_target"] = tgt.get(k, 0.0)
        c["fire_control"] = ctl.get(k, 0.0)
        c["ratio"] = round(c["fire_target"] / max(c["fire_control"], 0.05), 2)
    return sorted(cands, key=lambda c: -c["ratio"])


def main():
    print(f"== example hunt: {CONCEPT} ==  explorer {URL}")
    cands = nominate()
    print(f"\n[1] NOMINATE — {len(cands)} lens candidates with a word-boundary '{CONCEPT}' lens token")
    for c in cands[:6]:
        print(f"    L{c['layer']:<3} N{c['n']:<6} matched {c['count']} rank {c['rank']:<2} "
              f"signs {c['signs']} lens {c['matched']}")

    ranked = firing_ratio(cands)
    print(f"\n[2] SELECT — firing on the concept text vs a control text "
          f"(ratio = target / control); target {TARGET_TEXT[:44]!r}…")
    for c in ranked[:8]:
        print(f"    L{c['layer']:<3} N{c['n']:<6} ratio {c['ratio']:>7.2f}  "
              f"(target {c['fire_target']:>6.2f} / control {c['fire_control']:>5.2f})  lens {c['matched']}")

    picks = ranked[:TOP_N]
    print(f"\n[3] VERIFY — steering the top {len(picks)} at {VERIFY_DOSES} (both poles); "
          f"markers vs the unsteered baseline of the same prompt")
    verify = []
    for c in picks:
        print(f"  --- L{c['layer']} N{c['n']}  (ratio {c['ratio']}, lens {c['matched']})")
        for dose in VERIFY_DOSES:
            for p in PROMPTS[:2]:
                r = steer(p, c["layer"], c["n"], dose)
                m, deg = score(r["up"])
                b, _ = score(r["baseline"])
                verify.append({"layer": c["layer"], "n": c["n"], "ratio": c["ratio"], "dose": dose,
                               "prompt": p, "markers": m, "baseline_markers": b, "degen": deg,
                               "text": r["up"], "baseline": r["baseline"]})
                print(f"    {dose:+.2f}  {p[:26]:28} markers {m:>3} (base {b:>2}, {m-b:+d})"
                      f"{'  DEGEN' if deg else ''}")
                print(f"          {r['up'][:110]!r}")

    live = [v for v in verify if not v["degen"] and abs(v["dose"]) <= 1.0
            and v["markers"] > v["baseline_markers"]]
    if not live:
        live = [v for v in verify if not v["degen"]]
    # most markers over baseline; on a tie prefer the SMALLEST dose (parsimony — find the minimal
    # effective dose, don't just crank it up)
    win = max(live or verify, key=lambda v: (v["markers"] - v["baseline_markers"], -abs(v["dose"])))
    print(f"\n  winner: L{win['layer']} N{win['n']} @ {win['dose']:+.2f} "
          f"(marks {win['markers']} vs base {win['baseline_markers']})"
          f"{'  [negative pole]' if win['dose'] < 0 else ''}")
    spew = [v for v in verify if v["degen"]]
    if spew:
        print(f"  ({len(spew)} degenerate rows excluded from the winner search — spew is not behavior)")

    print(f"\n[4] WINDOW — dose sweep of the winner {SWEEP_DOSES}")
    window = []
    for dose in SWEEP_DOSES:
        for p in PROMPTS[:2]:
            r = steer(p, win["layer"], win["n"], dose)
            m, deg = score(r["up"])
            b, _ = score(r["baseline"])
            window.append({"dose": dose, "prompt": p, "markers": m, "baseline_markers": b,
                           "degen": deg, "text": r["up"]})
            print(f"    {dose:+.2f}  {p[:26]:28} markers {m:>3} (base {b:>2}, {m-b:+d})"
                  f"{'  DEGEN' if deg else ''}")
            print(f"          {r['up'][:110]!r}")
    ok = [w for w in window if w["markers"] > w["baseline_markers"] and not w["degen"]
          and abs(w["dose"]) <= 1.0]
    best = (max(ok, key=lambda w: (w["markers"] - w["baseline_markers"], -abs(w["dose"]))) if ok else None)
    best_dose = best["dose"] if best else win["dose"]
    best_lift = (best["markers"] - best["baseline_markers"]) if best else 0
    print(f"  recommended window: {best_dose:+.2f} "
          f"(marker lift {best_lift:+d}; rows above |1.0| are pole confirmation only — the unit is spewing there)")

    print(f"\n[5] CONTROL — same-layer unrelated neuron L{CONTROL[0]} N{CONTROL[1]} @ {best_dose:+.2f}")
    control = []
    for p in PROMPTS[:2]:
        r = steer(p, CONTROL[0], CONTROL[1], best_dose)
        m, deg = score(r["up"])
        b, _ = score(r["baseline"])
        control.append({"dose": best_dose, "prompt": p, "markers": m, "baseline_markers": b,
                        "degen": deg, "text": r["up"]})
        print(f"    {p[:26]:28} markers {m:>3} (base {b:>2}, {m-b:+d}){'  DEGEN' if deg else ''}")

    wlift = best_lift          # the lift of the row at the recommended window, not the peak verify row
    clift = sum(c["markers"] - c["baseline_markers"] for c in control)
    verdict = {"concept": CONCEPT,
               "winner": {"layer": win["layer"], "n": win["n"], "firing_ratio": win["ratio"],
                          "dose_window": best_dose, "marker_lift": wlift,
                          "peak_lift_any_dose": win["markers"] - win["baseline_markers"]},
               "control_neuron": {"layer": CONTROL[0], "n": CONTROL[1], "marker_lift": clift},
               "specific": bool(wlift > 0 and clift <= 0)}
    print(f"\n== verdict ==\n  L{verdict['winner']['layer']}:N{verdict['winner']['n']} "
          f"window {best_dose:+.2f} (marker lift {wlift:+d}, firing ratio {win['ratio']}); "
          f"control lift {clift:+d} -> {'SPECIFIC to the concept' if verdict['specific'] else 'NOT separable from the control — treat as unproven'}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"candidates": cands, "ranked": ranked[:20], "verify": verify, "window": window,
                   "control": control, "verdict": verdict}, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
