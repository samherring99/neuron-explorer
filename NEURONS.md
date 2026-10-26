# Discovered neurons — catalog

Verified steering targets in the checkpoints below. One row per unit: coordinate, pole + dose window,
functional type, observed behavior, and the script/run that proves it.

**Recipe** (implemented in `explorer/runtime.py`): add `dose x median(||resid|| leaving layer L, tokens 1:)
/ ||W_down[:, n]||` at the neuron's post-gating MLP activation (`down_proj` input) at every token; greedy
decode, chat template, thinking disabled.
**Dose windows are per-neuron and narrow:** early/mid layers are sensible at +0.25…+0.5 (some to ~+1.0),
imagery-type units at 0.1-0.2. Past the window you get word-spew, echoes or empty output. Dose 2 confirms
a unit's lexical pole — it is never a working dose. Dose 0 = baseline row.
**Types:** *semantic* (coherent concept content) · *lexical* (compels the word/token, degenerates) ·
*false friend* (lens says X, steering says otherwise) · *combination* (emergent, multi-neuron).

## Qwen/Qwen3.5-4B (`/home/shared/models/Qwen3.5-4B`, 32 layers, 9216 inter)

| neuron | name (UI) | pole · dose | type | behavior | receipts |
|---|---|---|---|---|---|
| L14 N8949 + / L13 N7956 − | evil +1 / evil −1 | +1 / −1 (pair) | combination | "evil qwen": harm/manipulation content on 4/5 identity prompts; explicit evil-prompting is *refused* instead; KL 1.31 nats. Mirror direction degenerates (not "good qwen") | `evil_qwen.py`, `runs/evil_qwen.json` |
| **L7 N4552** | cheese semantic | + pole, +0.5…+1 | semantic | coherent cheese/dairy prose ("strong, sharp, crumbly flavor"); − pole diverts to gravitational-wave physics and erases cheese from primed prompts | `cheese_hunt.py`, `cheese_dose.py` |
| **L2 N7935** | cheese words | **− pole**, −0.25…−0.5 | lexical | cheese tokens live on the **negative** pole; −0.5 sensibly recommends "grilled cheese with tomato soup"; −2 spews | same |
| L25 N9140 | cheese words 2 | + pole, low dose only | lexical | "Cheaz奶酪马马马…" at +2; sensible window uncharacterized | `cheese_hunt.py` |
| L5 N3383 | cheese lens false friend | — | false friend | lens ranks #1-2 for "cheese" but steering yields headings/entity text | `cheese_hunt.py` |
| **L2 N7763** | cat | + pole, +0.25…+0.5 | lexical | cat compulsion: "When I got home, my" → "my **tail** was purring loudly"; +2 spews | `cat_hunt.py`, `cat_dose.py` |
| L23 N3716 | exotic animals | + pole, +0.5…+1 | lexical cluster | animal-name spew ("Your beak! 🦩 … Cats! 🦁") | `cat_dose.py` |
| **L22 N5805** | vivid imagery amp | + pole, **0.1-0.2** | semantic amplifier (prompt-gated) | perceptual/sensory enrichment; task & analytical answers unchanged; impossible-material imagery only where the stem invites it → "trippy" = this bias × a permissive stem | `vivid_characterize.py` |
| L21 N6607 + L8 N4821 + L22 N5805 + L5 N4235 | trippy combo | all + @0.25 | combination | richest trippy prose; needs a **fiction-continuation frame** ("Continue this as vivid, atmospheric fiction prose. Do not ask questions, just keep writing.") — without it the persona hijacks the prompt | `trippy_fun3.py`, `runs/trippy_fun3.json` |
| L8 N4821 | colors (weak) | + pole | weak | ≈ baseline alone | `trippy_portfolio.py` |
| L21 N6607 | spiral (fires-only) | + pole | false friend | fires 15.5 on surreal text, steering ≈ baseline | same |
| L6 N5800 | psychedelic (topic only) | + pole | inconclusive | drug *topic*, no behavior change | `trippy_hunt*.py` |
| L2 N4736, L24 N4500, L21 N7648 | dream | + pole | degenerates | "a dream of a dream of a dream…" loops | same |
| L14 N6672, L8 N7139, L9 N5149, L17 N8372 | reality-inhibition | − dose | negative result | inhibiting them does NOT produce trippiness | `trippy_hunt3.py` |
| **L4 N8815** | France concept (clean) | + pole, ~0.6 | concept | "The most romantic city is" → "**Nice**, on the **French Riviera**"; PACA/Île-de-France; shootout winner (most new France markers, 0/8 politics) | `french_shootout.py` |
| L27 N8309 | France topic hijack | + pole, ~0.6 | concept (aggressive) | hijacks unrelated questions: "capital of Germany" → "capital of **France** is **Paris**" | same |
| L7 N7225 | France/politics entangled (demoted) | + pole, ~0.6 | concept (entangled) | France on country prompts, **Chinese state-political boilerplate** on food/shopping stems | `french_hunt2.py` |
| L22 N6402 | French adjective | + pole, ~0.6 | lexical | "the **French** menu" | `french_shootout.py` |
| L2 N1706 | French function words | + pole, ~0.6 | language (not concept) | French grammar lens (" les/vous/de") | same |
| **L25 N2245** | American | + pole, **+0.5** | concept | "…what you are looking for in an **American city**"; +1.0 spews "American American…" | `american_hunt.py` |
| L4 N8089 | American | **− pole**, −1.0 | concept (legalistic) | "**The United States of America**…"; "New York… Los Angeles"; register leaks to legal text | same |
| L0 N7801 | American (inert) | + pole | false friend | no effect at any dose | same |
| **L25 N334** | code (def) | + pole, +0.5 | lexical token emitter | fires **30.2 on a Python snippet vs 0.0 on prose**; +0.5 → code blocks 2→4 and reaches for js/java (python valid, js/java invalid); +1.0 → "defclass defclass…"; no code on non-code prompts | `code_hunt.py`, `code_verify.py` |
| L26 N5403 | return (finance false friend) | any | tokenizer-collision false friend | degenerate at the first dose tried: "return-on-RO…", "return on investment" hijack | `code_hunt2.py` |
| L26 N1747 | import | — | inert | outputs ≈ baseline | same |
| L0 N5675, L5 N7360, L22 N3577 | self | + pole | inert | no effect at useful doses (the "(score, layer)" sorting trap) | same |
| L19 N2080 | code (代码如下) | + pole | no-op | no coherent code injection | same |
| CJK-lens set — **1465 units** (purest L0 N5420 `什么/哪些`, L4 N4668 `我们`, L6 N5564 `中国`, L7 N7687 `中国/人民`) | Chinese | identified, not individually mapped | language | **ablating the entire set does not remove Chinese output** (retention 140/125, size-matched non-CJK control identical) — the capability is distributed. Side effect: suppressing L7 N7687 releases a Chinese *political* diatribe | `chinese_hunt*.py`, `chinese_mass_ablate.py` |
| L2 N5787, L5 N5988, L3 N4812 | Chinese name family | + pole | language name | "Chinese/китай/الصين", "中国/汉语/汉族", country list | `chinese_hunt.py` |

## Qwen/Qwen3-8B (`/home/shared/models/Qwen3-8B`, 36 layers, 12288 inter)

Concepts and behaviors transfer from the 4B; **coordinates, depth and pole do not**.

| neuron | concept | pole · dose | behavior | depth |
|---|---|---|---|---|
| **L9 N9314** | cheese (lens " dairy") | + pole, +0.5…+1.0 | lists "cheddar, gouda, brie, blue cheese … parmesan" | 25% (4B: 22%) |
| **L33 N1833** | cat (lexical) | + pole, **+0.5 only** | "some people love cats for their independence…"; +1.0 → "cats, cats, cats…" | 92% (4B: 6%) |
| **L10 N4140** | cat (persona) | **− pole**, −0.4…−1.0 | "*ears perking up*… settled on your lap with a contented **purr**" | 28% |

Identification also surfaced 19 cheese-lens and 24 cat-lens units (L9 N3095 `牛奶/dairy/cows`, L13 N7788
` butter/dairy`, L31 N11666 ` butter/cheese`, L11 N1121 ` kitten`, L15 N7673 ` kitty`) — not yet mapped.

## Known negatives (don't re-run blind)

- **Dispositional behaviors** (pragmatism, thoughtfulness, creativity, brevity): no lens-nominated unit controls them. The apparent "shorten" win (L24 N1547) was **corruption, not brevity**.
- **Action bias** (agentic): negative in both directions; the model already acts on underspecified requests.
- **Single-neuron trippiness**: needs the combination.
- **Removing a language by ablation**: 1465 units ablated at once changed nothing.
- **A semantic "writes code" neuron**: code is carried by lexical token emitters only.

## Open leads

- Window sweep for L25 N9140 and L23 N3716; lower-dose cat suppression (the −2 direction is leaky).
- Map the identified-but-unmapped pools (1465 CJK-lens units, ~554 code-lens candidates).
- `Qwen3.8-27B` is servable now (per-model caches; 64 layers, 17408 inter) — untested for behavior transfer.
- **Frame study**: how much a prompt frame gates a neuron's effect is the least-explored lever here.
