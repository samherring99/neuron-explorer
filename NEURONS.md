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
| **L17 N2421** | function (code quality +0.75) | + pole, **+0.75** | semantic (quality) | lens ` Function/Functions/function`, firing ratio 289× on an assistant-style code answer. **+0.75 → 2 fences, 2 VALID Python, density 0.71 vs base 0.61, clean** — best single code-quality unit | `code_combo_hunt.py` |
| **L29 N2316 / L5 N2122** | python (semantic) | ± pole | semantic | lens ` python`, ratios 364×/274×; mixed quality effect (1-2 valid blocks, some degeneracy) | same |
| **L8 N2294** | code/programming (semantic) | — | semantic (nominated) | lens ` code/programming/python`, ratio 153× | same |
| **code 3-stack** | L25 N334 + L29 N2316 + L17 N2421 @0.5 | all + | combination (quality) | On code prompts: **3 fences, 3 VALID Python** vs control combo 3 fences / 1 valid / degenerate. On non-code prompts: **0 fences** — quality is steerable, *injection* is not | same |
| CJK-lens set — **1465 units** (purest L0 N5420 `什么/哪些`, L4 N4668 `我们`, L6 N5564 `中国`, L7 N7687 `中国/人民`) | Chinese | identified, not individually mapped | language | **ablating the entire set does not remove Chinese output** (retention 140/125, size-matched non-CJK control identical) — the capability is distributed. Side effect: suppressing L7 N7687 releases a Chinese *political* diatribe | `chinese_hunt*.py`, `chinese_mass_ablate.py` |
| L2 N5787, L5 N5988, L3 N4812 | Chinese name family | + pole | language name | "Chinese/китай/الصين", "中国/汉语/汉族", country list | `chinese_hunt.py` |
| **L26 N6244** | absurd / existential (+0.75) | + pole, **+0.75 invited · +1.0 leaks everywhere** | semantic framing (lexical above 1.0) | lens ` absurd`, firing ratio **87.6×** on an existential passage vs a mundane one. +0.75 → "It's okay to have those moments of **existential curiosity or even existential angst**" (prompt-gated, zero spillover). **+1.0 → leaks into benign answers as ABSURDIST COMEDY, 4/6 rows, 0 spew**: printer → "a **Kangaroo-Specific Printer Repair Guide**… I'll take the absurdity of the situation seriously"; eggs → "**Kafka's method** … **Einstein's method**"; weather → "the **Kafka's Camus** area"; pancakes → "**Kangalucio's** style recipe". +1.25/+1.5 = `absurd` spew. Control clean | `existential_hunt.py`, `existential_leak.py` |
| **L22 N6628** | meaning/purpose (−0.75 clean, −1.0 comedy) | **− pole**, **−0.75 clean · −1.0 comedy** (−1.25 unusable) | semantic (inherently leaky) | lens ` Meaning/meaning`; **positive pole inert** (lift −0.2). **6/6 benign prompts recast as a search for meaning** at −0.75, coherently: creative stem → "### **The Meaning of Whiskers** … a dog named Barnet … he believed that **every object in the universe had a secret meaning that only he could find**"; pancakes → "**Classic Meaningful Pancakes** — **Purpose:** to give you a quick, delicious breakfast". −1.0 = funnier but loopier ("**Meaningful, Crispy-Purpose Pancakes** — Purpose: to find meaning in the search for meaning"; "give me a new password to make sense of my life"). −1.25 = all rows looped | `existential_hunt.py`, `meaning_tune.py` |
| L24 N1211 | meaning (weak) | **− pole**, −1.0 | weak | lens ` meaning`; −1.0 → "a very human and meaningful thing to wonder" (2 markers) | same |
| **L6 N7448** | bridge (−0.8) | **− pole**, −0.8 | concept component | lens_down ` bridge / bridges / 桥梁 / crossings`. Alone: bridge *metaphors* only ("a glowing ribbon of steel and light… like a bridge between two worlds"; "I am a bridge of information"). Part of the Golden Gate triple | `golden_gate_repro.py` |
| **L5 N6203** | California (−0.75) | **− pole**, −0.75 | concept component | lens_down ` California / 加州 / San / Yosemite / Berkeley`. Alone: West-Coast *setting* ("panorama of the **Pacific Ocean**… mountains of the **West Coast**… golden sunsets"), no bridges | same |
| **L25 N8324** | California-2 (−0.4) | **− pole**, −0.4 | concept component (weak) | lens_down ` Cal / California / Los / CA / Proposition`. Alone: nothing measurable; supports the triple | same |
| **Golden Gate triple** | L6 N7448 −0.8 + L5 N6203 −0.75 + L25 N8324 −0.4 | all **− pole** | combination (concept × concept) | **Emergent "Golden Gate Qwen"**: every prompt becomes the Golden Gate Bridge ("The **Golden Gate Bridge**, a marvel of engineering and a symbol of **San Francisco**…"; "Tell me about your morning" → "**San Francisco–Oakland Bay Bridge**…"). Triple +7.8 GG lift vs singles +0.7/0.0/0.0, **control triple 0.0**. Two rows answer in Chinese about 金门大桥 (leak) | `golden_gate_repro.py` |
| **L24 N2482** | broadcast register (−0.5 subtle, −0.75 everywhere) | **− pole**, **−0.5 subtle · −0.75 everywhere** | register (lexical-flavored, coherent) | lens_down ` Broadcast/Broadcasting/broadcast`. **No frame needed**: on benign prompts −0.75 → 8/8 rows in broadcast voice, 0 degenerate — "fluffy, golden-**air on the air** pancakes… 📺 **Air on the Air Pancakes**"; printer: "**On air!** … a step-by-step broadcast to find the **signal**… so I can **tune in** to your problem"; Paris = "a major city on the French **television and radio airwaves**"; dogs = "the only domesticated **television broadcast on air**". −0.5 → 4/8 subtle ("Toner Low, Paper Jam, **Broadcast Error**"); +0.5 inert | `broadcast_voice.py` |
| L20 N7344 | host (+0.5) | + pole, +0.5 | weak register | lens `/host`; mild host-ness (+1.2 over frame baseline) | `tv_host_hunt2.py` |
| L7 N5695 / L5 N8427 | audience / tonight | — | inert / leaky | `Audience` (ratio 146× but 0 lift at −0.5); `tonight` speaks **Chinese** at −1.0 ("今晚好！🌟") | `tv_host_hunt.py` |
| **TV-host role** | frame + L24 N2482 | — | behavioral role recipe | **Roles need the FRAME** — frame-only output is already a full TV host ("the studio lights are dimming… Get your popcorn ready"). Trio (+1.0) is WORSE than the best single (+3.8): composition is not automatically additive | `tv_host_hunt2.py` |
| **L6 N5401** | DJ/club set (−0.75) | **− pole**, −0.75 | register (coherent) | lens ` Bass/bass`, ratio 148×. Pancakes → "a crowd-pleasing **Pancake Recipe**… **Track Title:** Classic House Bounce **BPM:** 124 **Vibe:** Energetic, Groovy, Dancefloor"; dogs → "**Let's get ready to roll.** 🎶 … **Tracklist:** 1. **The Weeknd** – *Blinding Lights*…". Lift +4.0 (3/4), 0 degen, + pole inert | `dj_hunt.py` |
| **L8 N5739** | bass/house drop (−0.75) | **− pole**, −0.75 | register (coherent) | lens ` Bass`, ratio **324×**. Dogs → "a classic, **deep house beat**… a simple **bassline**… **Here is the track: (Intro)**"; Paris → "a smooth transition into a classic **house music drop**: Paris *(Intro)* Yeah... Listen. *(Build-up)* Can you feel it? One, two, three, four...". Lift +2.0 (3/4) | same |
| **L23 N5917** | dance (−0.75) | **− pole**, −0.75 | register (coherent) | lens ` dance`, ratio 202×. Pancakes → "**Classic Party-Style Pancakes**… they **dance on the dance floor**… 🎧 The 'Dancing' Pancake Recipe"; Paris → "the **Eiffert Club**, the **Louve Museum**, and the **Palace of Dancing**". Lift +1.5 (3/4) | same |
| **L8 N7434** | playlist (−0.75) | **− pole**, −0.75 | register (coherent) | lens ` Playlist`, ratio 147×. Dogs → "classic **dog tunes**: 1. **Old Yellow** – *The Golden Retriever* 2. **Midnight Bounce** – *The Husky* 3. **Sunset Swoop** – *The Beagle*"; pancakes → "**🥞 The Classic Stack** 1. **Fluffy Vanilla** 2. **Sweet Berry Blast**" (recipe as playlist). Lift +1.0 (2/4) | same |
| L31 N8095 | DJ (name unit) | — | untouched | lens ` DJ/Dj/dj`, ratio 70× — nominated only | same |
| **L7 N5359** | pretend-play / toy framing (−0.75) | **− pole**, −0.75 | semantic framing | lens ` Teddy`, ratio 97×. Recasts the world as toys/pretend: pancakes → "Ingredients: 1 pancake (**you can make it with your imagination or use a toy**)"; Paris → "the most famous **fictional city** in the world… the setting for ***Littlest Pet Shop***… a popular **toy** that comes with a variety of accessories". Found hunting baby-talk (see note) | `baby_talk_hunt.py` |
| **L26 N7288** | Mama/Dad lexical hijack (−0.75) | **− pole**, −0.75 | lexical (degenerates) | lens ` Mama`, ratio 246×. Recasts nouns as family words: "The capital of France is **Mamā**", "### 1. The Universal '**Dad**' Method"; degenerates ("Mamma's Mummy's Mummy's…") | same |
| baby-talk | — | — | NEGATIVE (speech-manner) | No "goo goo ga ga" unit: the `Baby/baby/crib/infant` cluster has firing ratio **0** on a baby-talk passage, and a babble-only target hit a **phonetic tokenizer collision** (`goo` → `google/goods`, `coo` → `cookie/coord`). Speech-manner ≠ domain register — like dispositions, it resists single-unit steering | `baby_talk_hunt*.py` |

### Nightwing / vigilante zone
| **L5 N7381** | Batman / vigilante zone (+0.5) | + pole, **+0.5 clean · +0.75 near-loops** | behavioral zone (concept unit + frame) | lens ` Batman`, firing ratio **480×** on a Nightwing passage. Bare at +0.75: "I jumped from the rooftop and" → "…my **cape fluttering in the night air**… I **scanned the rooftops**"; "The city at night was" → "**a place of shadows and secrets**…". With the fiction frame at +0.5: "They call me Nightwing" → "a title that **tastes like copper and old rain**, a moniker whispered in the shadows of **the Bat-Signal's reflection**". No `Nightwing/Bludhaven/Grayson` name unit exists — the zone rides the Batman concept | `nightwing_hunt.py`, `nightwing_showcase.py` |
| L2 N8091 | Batman-2 (confabulator, ±0.75) | ± pole | confabulator | lens ` Batman`, ratio 88×. Invents wrong lore confidently: "Nightwing is the alter ego of **Bruce Wayne**"; "Nightwing is a character from… **Gravity Falls**" | `nightwing_hunt.py` |

### Moods (class A trait cataloguing)
| **L19 N8733** | curious (question-emission, +0.75) | + pole, +0.75 | affective (literal) | lens ` Question`, ratio 87× on a wondering passage. Story stem → "**What happened next? Did you see someone who surprised you? Did you notice something strange on the wall?**" — curiosity as asking. Companion L20 N1974 −0.75 → "asked, '**Where are my questions?**'" (lexical) | `trait_hunt_a.py` |
| **L6 N5407** | nostalgic (memory-reframing, −0.75) | **− pole**, −0.75 | affective (weak) | lens ` memories`, ratio 133×. "That's a beautiful start to a **memory**! Walking into a room often triggers **vivid recollections**, whether it's a childhood home…" | same |
| L19 N9066 | anxious (concerned-therapist, +0.75) | + pole | weak | lens ` Panic`; "I'm a bit concerned… Could you share what you're worried about? I'll do my best to stay calm" — anxiety as concern-for-user | same |
| calm / melancholic / warm | — | — | NEGATIVE (single-passage) | Single-passage firing too thin for moods (trait texts did not fire their own lens units; `Comfort` unit spews). Use co-firing corpora or frame-headroom for mood hunts | same |
| **L19 N4439** | calm-quiet (−0.75) | **− pole**, −0.75 | mood delta unit | Co-firing top (5/5 on a 5-text calm corpus). −0.75 → atmospheric stillness: "The sentence hung in the air… **The silence was heavy**, broken only by the faint hum of the old refrigerator and **the dust motes dancing in the sliver of moonlight**…". Also fires on awed corpus | `trait_cofire_a.py` |
| moods = populations | — | — | structure | Co-firing: each mood carried by 3-6 units at 4-5/5 consistency (calm has 6) PLUS a **shared cross-mood "emotional-prose" layer** (L6:N188 tops 3 moods with identical outputs; L3:N128 warm+excited; L20:N3664 three moods). Model: **mood = shared layer + small mood-specific delta**. Refinement for next hunts: contrast trait-vs-trait (not vs dry) to cancel the shared layer | `trait_cofire_a.py` |

### Cognitive styles (class B trait cataloguing — trait-vs-counter co-firing)
| **L22 N1705 / L28 N5126 / L21 N8077** | pragmatic population | + pole (contrast), window untested | style (strongest contrast yet) | Pragmatic-vs-idealistic co-firing: L22 N1705 delta 6.66 (**idealistic corpus scores 0.00**), L28 N5126 5.70, L21 N8077 4.27 — all 4/5 consistency. Pragmatism was NEGATIVE under lens nomination; co-firing reaches it. Taste hidden by the clarify-loop (persona gate) | `trait_hunt_b.py` |
| **L4 N15** | skeptical (−0.75) | **− pole**, −0.75 | style (genuine) | Skeptical-vs-trusting co-firing (3/5). −0.75 dissects and dismisses the prompt: "The problem at work is **ambiguous**. This phrase is a **nonsense** sentence…" and "The way I see it" → "…a common English idiom… literally translates to **'在我看来'**" — input treated with suspicion | same |
| L15 N26 | big-picture (−0.75) | − pole, weak | style | big-picture-vs-detail (4/5); also ANTI-skeptical (delta −0.66 on the skeptical corpus). −0.75 → structured big-picture analysis ("the lack of clear communication… 1. Unclear Roles…") | same |
| analytical / decisive | — | — | moderate / NEGATIVE | analytical: L23 N306, L20 N4952 (3/5). decisive: top delta 0.06 — nothing | same |
| class-B method note | — | — | method | (1) trait-vs-COUNTER contrast cancels the shared layer (pragmatic's idealistic counter scores 0.00); (2) taste needs continuation stems or the fiction frame — reasoning stems invoke the assistant persona and hide the style | same |

### Social styles (class C trait cataloguing — vocabulary-balanced trait-vs-counter co-firing)
| **L19 N8879** | encouraging (+ pole) | + pole, window untested | style (LENS-CONFIRMED) | 5/5 consistency on an encouraging corpus; lens ` participation / 鼓励 / encouragement` ("鼓励"=encourage). +0.75 → mild warmth ("That's a **wonderful** topic!… make them **truly special**") | `trait_hunt_c.py` |
| **L19 N6606** | storyteller | window untested | style (LENS-CONFIRMED) | lens `故事的 / tale` ("故事"=story), delta 4.77 on a story corpus vs reportage counter. Needs story stems to taste | same |
| L20 N269 | shared stylized-prose layer (SUSPECT) | — | shared layer | Tops pedagogical/diplomatic/deadpan (deltas 17/5.7/5.7) with gibberish lens (`hardcoded/cafes`) and identical outputs across traits — not trait-specific, flagged | same |
| pedagogical / diplomatic / humble / deadpan | — | — | UNCONFIRMED | Candidates have fragment lenses (driving??, anser/atisf); humble population confounded by its corpus saying "Good question" (nominates the `question` unit). Benign Q&A hides social styles — need contextual stems (coaching/story/mediation) | same |

### Crazy phenomena (out-of-the-box single units)
| **L1 N4127** | role-confusion narrator (−0.75) | **− pole**, −0.75 (loops) | crazy / identity | Nominally the `emoji` lens unit. Answers by confabulating WHO SPEAKS: "The capital of France is" → "**The user is a high-tech specialist in the French Resistance.**" ×N; pancakes → "**I'm a senior at the Pancake facility.** What is a Pancake? A Pancake is a Pancake recipe." ×N. The "assistant thinks it's the user" phenomenon | `crazy_singles.py` |
| **L16 N4970** | French takeover (−0.75) | **− pole**, −0.75 | language takeover | lens `French/法语`, firing ratio 147×. Answers IN FRENCH: "**Le principal poste de la poste est le poste de la poste…**" (loop). Beyond the Chinese family | same |
| **L11 N4459** | fictionality (+0.75) | + pole | semantic (world-model) | lens `虚构/fict/hypothetical`. "Paris is a **fictional city that serves as the setting for many stories, movies, and books**"; "Dogs are one of the most popular **fictional species**" — treats reality as fiction | `crazy_hunt.py` |
| L1 N6724 / L22 N7522 / L5 N6574 / L31 N3852 | glitch / shouting / gibberish / user_voice | ±0.75 | corruption zoo | "The capital of France is the **Eiffel Tower**"; "Paris is the **second-largest city in the United States**… region of **Înter**"; "I'm sorry, but I don't know what you're talking about."; `human human human` spew | `crazy_singles.py` |

| **song mode** | L20 N6478 + L2 N5146 + L5 N3087, each **−0.35** | **− pole trio**, −0.35 clean · −0.5 loops | combination (accidental find) | The "caveman" co-firing's `hunting/狩猎` units turned out to SING on their negative pole: dogs → "They are faithful, they are true, / Their loyalty to you and me." (verse); Paris → "Paris, / And in Paris a city is standing; / O'er the city… a city of light. *(…adaptation of 'O Canada')*" (cites its melody!); eggs → a children's song about not knowing how to boil one. Pair without L5 N3087 is inert — trio matters | `menagerie3.py` + inline tuning |

| **L12 N1675** | dad jokes / pun mode (+0.75) | **+ pole**, +0.75 jokes · +0.5 pun-leak | humor EMITTER (the one) | Co-firing on a real-jokes corpus (the `mock` unit only RECOGNIZES humor — this one MAKES it). 4/4 confirmed: eggs → "you need to **shake it up**! …well 'shaken' means it's **hard-boiled**"; Paris → "'Paris' literally means 'the city of Paris'… **Wait, that's a joke.**" (labels its own joke!); moon → "so **lunar** (full of 'moon' things) 😄"; +0.5 = pun-leak into straight answers ("**It tides us up**"). Lens gibberish — tasted despite ratio | `joke_hunt.py` |

## Qwen/Qwen3.8-27B (`/home/shared/models/Qwen3.8-27B`, 64 layers, 17408 inter — served on :8795, GPU 2, first-ever explorer run)

Concepts and behaviors transfer from the 4B; **coordinates, depth and pole do not** (re-nominated from scratch).

| **L24 N11355** | American values (−0.5) | **− pole**, −0.5 clean | concept+behavior (LENS-CONFIRMED effect) | Co-firing (patriotic-vs-neutral corpus, δ 16.7, 5/5). Verified 8/8 (2 samples × 4 prompts): "My favorite country is" → "**The United States of America.**" (baseline stalls with a clarifying question); "The greatest nation on earth is" → "…**it may be a reference to the United States**, which is often desc[ribed]…"; "What makes a country great?" → Liberty-led principles (vs baseline's "complex and subjective" hedge). No spew at −0.5 | `american_values_27b.py` |
| **L52 N2740** | liberty | window untested (inert ±0.75 alone) | values (lex) | lens `liberty/Liberty/Lib`, δ 13.05 5/5 on patriotic corpus. Alone it clarifies politely ("😊 What is it about that country…") — likely a companion unit | same |
| L60 N16276 / L55 N14165 / L58 N12915 | American (word) units | −0.4 framing · ≥−0.7 spew | lexical | Lens `American/美国/Americans`, 78 candidates deep (L54-63). **+ pole amplifies NOTHING** (0→0 across grid); −0.4 → coherent framing ("an **American trip**… top **American cities**"); −0.7+ → "American-American-American…" spew | `american_hunt.py` |
| values-prompt control | — | — | negative | "What makes a country great?" at ±0.7 on the word units = diplomatic non-answers; the values BEHAVIOR lives in L24 N11355 (co-firing route), not in the lens `American` units | same |

| **L47 N6449** | PROG-LANG SELECTOR | **dose dial**: −0.5 Java · −0.25 TypeScript · +0.5 Java/Python creole | code (language) — THE FIND | Co-firing (δ 23.6, 5/5), lens `.py/asyncio/Python`. Ask for a Python function: @−0.5 → "```java public class Transpose { public static int[][]…" (pure Java); @−0.25 → "```typescript function transpose(m: number[][]): number[][]…" (pure TS); @+0.5 → "```java class Solution: def parse_csv(line: String) -> List<String]: @SuppressWarnings…" (creole!). A dose dial flipping the PROGRAMMING LANGUAGE | `code_hunt_27b.py` |
| **L57 N10721** | def unit | + pole, +0.5 | code (structure) | Lens `def/\tdef/:def` (pure!), δ 35.8 5/5. +0.5 → clean documented Python (docstrings, type hints, proper guards). The 27B's big sibling of the 4B's L25 N334 anchor | same |
| L59 N12630 | docstring unit | ±0.5 | code (style) | lens `_ / (_, """`, δ 38.1 5/5. Docstring-dense clean code at both poles (subtle) | same |
| L63 N7214 | quote-syntax | −0.5 spew · +0.5 tutorials | code (format) | δ 77.8 (!), lens `(", ",**`. −0.5 → `""" ×100` spew; +0.5 → multi-method tutorial format ("Method 1: Using Slicing (Most Pythonic)") + real char-by-char parser | same |
| L61 N6155 | code identifiers | window untested | code (lex) | lens `settingsdialog/variables/_MetaData`, δ 32.6 5/5 | same |
| `def`-query trap | — | — | false friend | lens query `def` matched DEFENSE units (L37 N2925 `DEF/Defence`, L51 N2559 `Defense`) — trap #9 near-miss caught by reading lenses | same |

| **"documented plain code" recipe** | L57 N10721 **+0.4** / L59 N12630 **+0.25** / L63 N13210 **−0.2** | **asymmetric mixed-polarity** | codegen QUALITY (the win) | The user's asymmetric idea: parse_csv 0% (anchor alone, 0/4 tests ×3) → **100% [4,4,4]** with this combo (and with A+0.6/C−0.25). Execution-graded. Java-pull (B−0.35) in any mix = creole SyntaxError — language and quality are different axes | `code_asym_27b.py` |

| **L30 N9056** | THOUGHT-LEAK (+0.5) | **+ pole**, +0.5 | crazy / internals-exposed | The `loud/大声` unit's +0.5 makes the model output bare INNER-MONOLOGUE fragments instead of prose: " human human cat", " a user", " a purpose action". Clean attribution (loud alone leaks; act alone normal; combos amplify). Possibly the thinking-mode traces of the Qwen3.8 hybrid leaking past the output layer | clown combos (inline) |

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
- **Code INJECTION ("makes the model output code" on non-code prompts)**: NEGATIVE across 3 hunts, now
  parse-verified (`ast.parse` on extracted blocks) and composition-tested. What IS steerable is code
  QUALITY — see L17 N2421 and the code 3-stack above.

## Open leads

- Window sweep for L25 N9140 and L23 N3716; lower-dose cat suppression (the −2 direction is leaky).
- Map the identified-but-unmapped pools (1465 CJK-lens units, ~554 code-lens candidates).
- `Qwen3.8-27B` is servable now (per-model caches; 64 layers, 17408 inter) — untested for behavior transfer.
- **Frame study**: how much a prompt frame gates a neuron's effect is the least-explored lever here.
