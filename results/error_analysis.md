# Qualitative error analysis — dair-ai/emotion test, zero-shot, `plain` run

Sources: `inspection_sample.md` (22 frozen examples), `supplementary.json` (`top_confusions_plain`, `agreement_plain`, `accuracy_by_length_tercile_plain`), plus a few extra rows quoted from `raw_predictions.parquet` for illustration. No new headline metrics were computed. Measurement is stated first in each section; interpretation is marked.

Governing caveat: dair-ai/emotion labels come from hashtag distant supervision (the author's own `#hashtag` was mapped to one of six classes and stripped). Gold is an author self-tag, not an annotation, and several "errors" below are better read as label noise or ambiguity.

## 1. Inspection categories

**all_correct** (idx 1284, 1019, 1724). Single-keyword cases ("ashamed", "enjoyed", "divine intervention"). Calibration differs: PrismNLI hedges joy/surprise 0.53/0.47 on 1019 while Laya gives 0.99. *Interpretation:* easy rows are decided by one emotion word; confidence differences reflect style, not skill.

**all_wrong_three_distinct** (idx 1979, 1588, 62). idx 1979, gold=anger, "no strong feelings for this book neither hated nor loved it": Jev sadness 0.78, PrismNLI joy 0.29 (flat), Laya love 0.91. The text asserts neutrality; the gold looks like noise and PrismNLI's flat distribution is the only honest output. idx 62, gold=joy, "dazed and not feeling particularly sociable ... in hiding": Jev sadness 0.95, Laya fear 0.77, PrismNLI surprise 0.77 — the joy presumably lives in the stripped hashtag. idx 1588 (gold=love, a long deliberation about not hurting feelings): Jev fear 0.72 on "hurt/betray". *Interpretation:* three-way disagreement clusters where surface content and hashtag diverge; each model falls back to its prior (Jev/Laya: negative -> sadness/fear; PrismNLI: surprise as an uncertainty sink).

**only_jev_correct** (idx 985, 711, 1627). "how treasured my london flatmates are" -> Jev love 0.71 vs joy ~0.9-0.99 from the others; "class leader only choose his friends not true" -> Jev anger 0.77, PrismNLI surprise 0.56, Laya love 0.89 (keyword "friends"). *Interpretation:* Jev's rare wins are pragmatic readings (a complaint, implied loneliness); Laya's idx 711 is pure lexical capture.

**only_prismnli_correct** (idx 1376, 1049, 1179). "i feel completely rude with not keeping up" -> PrismNLI anger 1.0, others sadness. "impatience ... and impressed at the same time" -> PrismNLI surprise 0.65, Jev a three-way tie, Laya sadness 0.998. Most telling: idx 1179, gold=joy, "i didn t and still don t feel lucky though" -> PrismNLI joy 0.97, Jev sadness 1.0, Laya sadness 0.95. *Interpretation:* a human reads 1179 as sadness. Returning joy at 0.97 on a negated "lucky" is the behaviour of a model that has learned this dataset's lexical mapping, not the semantics (section 5).

**only_laya_correct** (idx 1746, 1404, 604). 1746 is a near-tie (0.37/0.37) that lands on joy. 1404 (numb, tingly arms, gold=sadness): Jev fear 0.9, PrismNLI surprise 0.45, Laya sadness 0.93 — gold is debatable. 604 "fucked up big time but i have to protect a and myself": Laya anger 0.96, Jev fear 0.99, PrismNLI sadness 0.97. *Interpretation:* Laya wins via its sadness prior and a profanity->anger association.

**jev_highconf_wrong** (idx 1183, 1725). 1183, gold=joy, "rockstar razzle dazzle lifestyle but ... something worthwhile": Jev sadness 0.99, joy exactly 0.0; PrismNLI joy 0.99. 1725 (fabric "great for wicking away sweat"), gold=anger: all three joy; gold is noise. *Interpretation:* Jev collapses aspirational/ambivalent text to sadness; the rest is mislabelling.

**prismnli_highconf_wrong** (idx 1537, 1475). "fresh feeling of sweet he gave me" (gold=joy) -> love 0.98, Laya agrees; "repressed fear and anxiety and distrust" (gold=sadness) -> all three fear ~1.0. *Interpretation:* boundary or noise cases, not model errors in any useful sense.

**laya_highconf_wrong** (idx 341, 215). Open-mic "brave and excited" (gold=love): all three joy 0.95+. "i feel dirty talking to people for my personal gain" (gold=sadness): Laya anger 0.98, PrismNLI anger 0.94, Jev sadness 0.88. *Interpretation:* Laya's confident misses are mostly shared with PrismNLI; its distinctive failure is the next case.

**class_fill_fear** (idx 28) "i do feel insecure sometimes but who doesnt": Jev, PrismNLI fear; Laya sadness 0.86 — an instance of Laya's largest confusion.

## 2. Dominant confusions

*Measurement* (`top_confusions_plain`): Jev — joy->sadness 110 (13% of errors), anger->sadness 73, sadness->fear 65, joy->love 59, love->joy 53. PrismNLI — joy->love 64 (12%), joy->sadness 51, love->joy 45, sadness->anger 41, sadness->fear 39. Laya — fear->sadness 91, anger->sadness 86, joy->sadness 83, joy->love 78, sadness->joy 72.

*Interpretation.*
- **joy<->love** is a label-definition problem. "love" is the hashtag bucket for affection words, so "the packaging is really lovely" (idx 615) is gold=love while "i tell you that i love you ... my dear" (idx 1142) is gold=joy. All models key on love/lovely/beloved as a human would; the gold does not apply the rule consistently. This pair is PrismNLI's largest residual error and is likely near the noise ceiling.
- **anger->sadness** (Jev, Laya) is low-arousal anger: "greedy so needy so helpless" (1149), "dissatisfied with my purchase" (451), "envious" (92), "irritable" (1004). DAIR's anger includes envy, irritation, dissatisfaction; PrismNLI keeps these in anger, the others do not. A real capability gap.
- **fear->sadness** (Laya): anxiety without a fear keyword — "uncertain about my future" (1266), "uptight and unable to unwind" (830), idx 28. Laya has a broad negative->sadness prior; descriptively, its raw predictions over-produce sadness and under-produce fear relative to support.
- **sadness->fear** (Jev, PrismNLI) is the mirror: "awaiting an unwelcome visitor" (97), "feeling like im being watched" (1014), "a little disturbed" (1715). Threat-tinged; gold=sadness is defensible, not unique.
- **surprise** (support 66) has the noisiest gold: "wanted audiences to feel impressed" (222), "feeling overwhelmed" (705, 969), "everything was feeling amazing" (1791). Jev and Laya rarely predict it; PrismNLI predicts it more and uses it as an uncertainty sink (idx 62, 1404, 1104 "im definitely not feeling fearful" -> surprise).

## 3. High-confidence errors

*Measurement:* `frac_gold_prob_below_0.005_plain` = 0.151 Jev, 0.081 PrismNLI, 0.162 Laya. In the parquet, 302 Jev rows and 25 Laya rows put exactly 0 on gold (all errors by construction).

*Interpretation.* Jev's zeros are partly a format artefact (two-decimal probabilities, mass concentrated on one label), so 0.0 means "not considered" rather than a calibrated claim — but operationally they are confidently wrong: nothing would route them to review. Content is a mix of real semantic misses (idx 1183; idx 1589 "merely accepted what has been done", gold=joy -> sadness) and noise (idx 1142). Laya's 25 zeros are dominated by indefensible gold: "depth of sorrow" gold=surprise (1382), "sympathetic dread" gold=love (391), "calm ... sometimes i just get so sad" gold=joy (1711). PrismNLI's ~1.0 errors are mostly boundary cases (1537, 1475), but idx 1179 shows it can be confidently wrong in the direction of the dataset's lexical mapping.

## 4. Length effect

*Measurement:* accuracy by word-count tercile falls monotonically for all three: Jev 0.696 -> 0.580 -> 0.480; PrismNLI 0.812 -> 0.735 -> 0.624; Laya 0.671 -> 0.582 -> 0.505. Drop ~0.2 each; PrismNLI's margin holds at every length.

*Interpretation.* Long rows (23-61 words) carry several cues and a narrative turn (idx 62, 1588, 1627, 341); the hashtag encodes the author's final stance, often not the dominant surface sentiment. Length raises both model difficulty and label noise, and the two are not separable here.

## 5. Why PrismNLI leads — hypotheses, not findings

- **H1 Inherited exposure.** PrismNLI-0.4B descends from `deberta-v3-large-zeroshot-v2.0`, whose training mixture includes dair-ai/emotion train/validation. The test split is disjoint, but the label vocabulary and dataset-specific lexical mappings (lucky->joy, rude->anger, lovely->love) were seen. idx 1179 and 1376 are what this predicts; a truly zero-shot model should not give joy 0.97 to "still don't feel lucky".
- **H2 Template match.** The NLI hypothesis used here is close to the "This example tweet expresses the emotion: X" template used in that checkpoint's training on this dataset, making the run near-in-distribution prompting. Consistent with the `defined` variant (a different hypothesis) *lowering* PrismNLI accuracy (0.7245 -> 0.7015, McNemar p<0.001) while helping Jev and being neutral for Laya.
- **H3 Class-prior fit.** Descriptively, PrismNLI's prediction distribution tracks test support more closely, especially on anger and fear. Under H1 this is learned prior, not calibration skill.
- **H4 Better uncertainty behaviour regardless of exposure.** PrismNLI is flat on genuinely ambiguous rows (1979, 1404) where Laya saturates, which alone would improve selective accuracy.

The clean test of H1/H2 is a re-run on an emotion set with different provenance and label scheme (e.g. GoEmotions collapsed to six) to see whether the margin survives.
