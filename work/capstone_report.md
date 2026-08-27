# Capstone Report — Refresh Opportunity Scoring

- **Author:** Neha Rani
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/neha-raniii/flyrank-ml-internship
- **Date:** August 2026

---

## 0. Abstract

Content review teams cannot manually check every page on a site, so this project scores and ranks pages by refresh opportunity to answer one question: which pages should a reviewer look at first? Using the 30,000-page starter dataset and only signals observable before any decision was made, a hand-written baseline rule was tested, found to rely on a staleness assumption that did not hold on the data, and replaced with a rule grounded in a confirmed CTR-vs-position signal. A Random Forest classifier, trained and evaluated on an entirely held-out set of clients, then improved on that baseline — Precision@50 rose from 0.56 to 0.74 on the same split, and a naive random split (0.98) was shown to be a misleadingly optimistic alternative. The output is a ranked, reason-coded action queue of 20,018 pages, intended strictly as decision support for a content reviewer with limited time — not an automated or causal claim.

## 1. Problem framing

**Decision supported:** which content pages a review team should look at first when they have limited time each week.

**Unit of analysis:** one row = one content page (`content_id`), belonging to one client (`client_id`).

**Output:** a continuous priority score per page, sorted into a ranked queue with a reason code and a suggested action (`review_content`, `review_ctr_and_metadata`, `review_and_refresh`, `monitor`).

**Action a human takes:** a content strategist or SEO reviewer works down the ranked list and decides whether to refresh, expand, or leave a page as-is.

**Cost of a wrong call:** if the model flags a fine page as high-priority, the team wastes review time on it while a genuinely declining page waits longer to be noticed; if it misses a real decline, that page keeps losing visibility unnoticed. Neither failure is catastrophic, but both cost the team's limited attention — the resource this project exists to protect.

**Why ML helps here:** a fixed rule treats every signal as equally important with hard thresholds. Section 3's signal audit showed the most intuitive rule — staleness predicts decline — is actually *opposite* on this data (stale pages decline *less* often: 47.1% vs 54.2% for fresh pages, n=174 vs n=29,826). A hand-written rule cannot discover which combination of signals actually matters or how they interact; a learned model can, and it also surfaced `word_count` as a meaningful driver — a feature the hand-written baseline never used at all.

## 2. Data safety

**Data used:**
- Starter dataset — `data/raw/content_refresh_anonymized.csv`, 30,000 pseudonymized pages, used throughout framing, signal audit, baseline, and modeling.
- Warehouse release — `FlyRank/internship-warehouse` on Hugging Face (build `flyrank_pseudonymized_warehouse_release_v20260703`), ~79M daily fact rows, accessed via DuckDB for the Week 3 data-contract exploration (`month=2026-03`, mid-panel, with the final month held out as a sealed test window).

**Columns deliberately excluded, and why:**
- `trend_direction`, `trend_pct` — the label is built directly from `trend_direction`, and `trend_pct` is the exact percentage change that bucket comes from. A deliberate leak test (Section 4) confirmed including `trend_pct` pushes AUC from an honest 0.630 to a meaningless 1.000.
- Any product-computed flag (`health_score`, `priority_score`, `action_type`) — not present in this dataset by design, and never reconstructed and fed back in as a feature, since that would let a model copy an existing decision rather than discover real signal.
- `content_id`, `client_id` — identifiers only, used for grouping/joins, never as model inputs.
- `char_count` — near-duplicate of `word_count`; kept `word_count` only, to avoid redundant correlated features.

**Leakage risks considered:**
- Label-derived fields (`trend_direction`, `trend_pct`) were excluded from every feature set across the track and explicitly leak-tested in `work/notebooks/w03_feature_leakage_check.ipynb` — honest AUC 0.630 vs leaky AUC 1.000, a 0.370 gap that confirms the detection method.
- Split-design leakage was checked directly: a naive random split on the Week-5 model produced Precision@50 = 0.98, versus 0.74 under a client-holdout split (`work/notebooks/w06_validation_audit.ipynb`) — the random split let pages from the same client sit in both train and test, letting the model partly memorize client-specific patterns.

**Confirmation:** no client names, domains, URLs, titles, or raw queries appear anywhere in `work/`. All identifiers used are the pseudonymous `content_id` / `client_id` hashes, used only for grouping and joins.

## 3. Baseline

**Signal audit first:** before encoding any rule, two candidate signals were tested directly against the data (`work/notebooks/w04_signal_audit.ipynb`, `w04_baseline_score.ipynb`):

| Signal | Verdict | Finding |
|---|---|---|
| Staleness → decline (`days_since_last_update ≥ 180`) | **OPPOSITE** | Stale pages declined *less* often (47.1%, n=174) than fresh pages (54.2%, n=29,826) |
| CTR vs. position tier | **CONFIRMED** | Mean CTR falls cleanly by tier: top_3 (1.484) > page_1 (0.652) > striking (0.323) > page_3_5 (0.222) > deep (0.150) |
| Word count vs. decline | **OPPOSITE** (of common belief) | Decline rate *rises* with word count: <1K words decline 20.7% of the time vs 59.5% for 3K+ words — an association, not a causal claim |

**Baseline rule (encoded from the confirmed signal only):**

```
score  = (tier_avg_ctr − page_ctr) × impressions_90d
eligible = avg_position > 0 AND impressions_90d ≥ 500 AND ctr_gap > 0
reason_code = "ctr_below_position_norm"
action      = "review_ctr_and_metadata"
```

The staleness signal — the intuition behind FlyRank's own refresh-flag logic — was deliberately dropped rather than forced into the rule, since it did not hold up.

**Why this is a fair comparison:** the baseline was built, scored, and locked (13,698 flagged pages) *before* any model was trained, on the same dataset the model would later use, with the same evaluation metric (Precision@50).

**Baseline result (on the Week-5 client-holdout test split):** Precision@50 = **0.56** (28 of the top 50 flagged pages genuinely declining).

## 4. Model / analysis

**Method:** Random Forest Classifier (`work/notebooks/w05_model.ipynb`).

**Why it fits the lane:** Refresh/Content Opportunity Scoring is a scoring/ranking task with multiple weak, possibly-interacting signals. Section 3's signal audit already showed one intuitive signal (staleness) pointing the wrong way — meaning the real pattern is not obvious from any single feature. A random forest can combine multiple weak signals and their interactions in a way a single linear rule cannot.

**Feature list (7 features):**

| Feature | Available before decision point? |
|---|---|
| `impressions_90d` | Yes — observed past-window count |
| `avg_position` | Yes — measured over the same past window |
| `ctr` | Yes — derived purely from past impressions/clicks |
| `word_count` | Yes — filled with 0 where missing (known limitation, not a true zero) |
| `days_since_last_update` | Yes — simple date difference, always knowable |
| `search_volume` | Yes — external third-party estimate, not tied to this page's own outcome |
| `engagement_rate` | Yes — filled with 0 where missing (pages with no GA4 tracking, not necessarily zero engagement) |

**Deliberately left out:** `trend_direction`, `trend_pct` (label-derived — see Section 2), any product-computed flag, `content_id`/`client_id` (identifiers, used for grouping only).

**Target / proxy definition (one sentence):** `is_declining_label = (trend_direction == "down")` — a current-window trend bucket used as a beginner proxy, not a genuinely future-looking outcome; a stronger capstone iteration would define a prior-window → future-window label instead.

## 5. Evaluation

**Split design:** client-holdout (grouped) — entire clients withheld from training so the model is tested only on clients it never saw (21 clients train / 8 clients test, 0 client overlap confirmed). This matches the validation approach used in the starter pipeline.

**Why this split is honest:** a plain random row split lets pages from the same client appear in both train and test, letting the model partly memorize client-specific patterns rather than learn a signal that generalizes to a brand-new client. This was demonstrated directly: the same model scored **0.98** under a naive random split versus **0.74** under the honest client-holdout split — a 24-point gap purely attributable to split design, not model quality.

**Model vs. baseline, same split, same metric:**

| Method | Precision@50 | Correct in top 50 |
|---|---|---|
| Week-4 baseline rule (CTR vs. position) | 0.56 | 28 / 50 |
| Random Forest (client-holdout) | **0.74** | **37 / 50** |

**Base rate context:** the label is roughly balanced (54.2% declining vs. 45.8% not, on the full starter dataset), so this is not a case of Precision@50 riding a skewed majority class — the model's 0.74 reflects genuine ranking skill above both the baseline and the underlying base rate.

**Error analysis (beats a big metric table):** pages with very low `impressions_90d` likely carry less reliable scores, since the model has fewer comparable examples to learn from at low volume — the baseline rule sidestepped this by requiring `impressions_90d ≥ 500` outright, while the model has no equivalent explicit safeguard. This is flagged as an open follow-up rather than resolved here.

## 6. Interpretation

**Feature importances (Random Forest):**

| Feature | Importance |
|---|---|
| `impressions_90d` | 0.261 |
| `avg_position` | 0.220 |
| `word_count` | 0.212 |
| `ctr` | 0.113 |
| `engagement_rate` | 0.070 |
| `search_volume` | 0.069 |
| `days_since_last_update` | 0.055 |

**In plain words:** three features — impressions, position, and word count — account for roughly 70% of the model's decisions. `days_since_last_update` (staleness) is the *least* important feature, directly consistent with the Section 3 signal audit finding that staleness alone is unreliable on this data. The model also draws meaningfully on `word_count`, a feature the hand-written baseline never used — evidence the fixed rule was leaving real signal on the table.

**Surprises / negative results:**
- Staleness pointing the *opposite* direction from intuition (Section 3) was the most important negative result of the whole project, and it directly shaped the baseline rule (staleness was dropped) and is reflected in the model's own low reliance on it.
- The leak test (Section 2) deliberately produced a "too good to be true" result (AUC 1.000) specifically to demonstrate what leakage looks like before confirming it was absent from the real feature set.
- The random-vs-grouped split gap (0.98 vs 0.74) is itself a negative result about the *baseline's own* Week-5 reporting process: it shows how easily a validation design choice — not model quality — can produce a misleading headline number.

## 7. Recommendation

The trained model was applied across the eligible slice of the starter dataset, producing a ranked queue of 20,018 pages (`work/notebooks/w07_action_playbook.ipynb`, exported to `work/outputs/action_playbook_summary.json`):

| Reason code | Pages | Suggested action |
|---|---|---|
| `high_risk_decline` | 11,294 | `review_content` |
| `monitor` | 7,562 | `monitor` |
| `ctr_below_expected` | 1,115 | `review_ctr_and_metadata` |
| `high_risk_stale_visible` | 47 | `review_and_refresh` |

**How a FlyRank editor would use this tomorrow:** open the queue, start at the top, and for each flagged page check the reason code and the raw numbers (impressions, position, CTR) before acting — never edit, unpublish, or redirect a page from the score alone. High-CTR-gap pages at strong positions are the fastest, safest wins (title/metadata review); `high_risk_decline` pages need a human check for consolidation or seasonality before being treated as a real problem (Section 7 of the lane guide's decline-vs-consolidation distinction).

**Confidence and limits, stated explicitly:**
- Validated Precision@50 = 0.74 means roughly **1 in 4** top-ranked pages will still be a false positive even under honest, client-holdout validation — this is a prioritization tool, not a certainty score.
- The queue reflects a single snapshot in time and will go stale as real page performance changes; a 90-day re-scoring cadence is recommended (matching the dataset's own `impressions_90d` window).
- Built and validated on the 30,000-row starter dataset only — not yet re-validated at full 79M-row warehouse scale.
- A "declining" flag means "currently trending down" (a current-window proxy), not "guaranteed to keep declining" — the label is not a future outcome.
- No causal claim is made anywhere: a high score means "worth reviewing first," not "guaranteed to recover if refreshed." Proving that would require a controlled experiment this data cannot provide.

## 8. Reproducibility

**Commands to re-run everything from a fresh clone:**

```bash
git clone https://github.com/neha-raniii/flyrank-ml-internship
cd flyrank-ml-internship
pip install -r requirements.txt
```

Then open and run top-to-bottom, in order:

1. `work/notebooks/w01_research_question.ipynb`
2. `work/notebooks/w02_ml_task_framing.ipynb`
3. `work/notebooks/w03_data_contract.ipynb`
4. `work/notebooks/w03_feature_leakage_check.ipynb`
5. `work/notebooks/w04_signal_audit.ipynb`
6. `work/notebooks/w04_baseline_score.ipynb`
7. `work/notebooks/w05_model.ipynb`
8. `work/notebooks/w06_validation_audit.ipynb`
9. `work/notebooks/w07_action_playbook.ipynb`

**Random seeds:** `random_state=42` used consistently across every `train_test_split`, `GroupShuffleSplit`, and `RandomForestClassifier` call in the track, so re-runs on the same data reproduce the same split and the same reported numbers.

**Environment:** Google Colab default runtime (Python 3.12–3.13 depending on session), `scikit-learn`, `pandas`, `numpy`, `duckdb`, `huggingface_hub` — see `requirements.txt` at the repo root for the full pinned list.

**Committed evaluation artifacts:** the client-holdout evaluation in `w05_model.ipynb` and `w06_validation_audit.ipynb` is checkable directly from the repo — both the split-building cell (`GroupShuffleSplit` on `client_id`) and the resulting Precision@50 output are committed with visible outputs, not taken on faith. The final action queue's summary metrics are committed at `work/outputs/action_playbook_summary.json`.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — [flyrank.ai](https://flyrank.ai).

---

### Claims checklist (confirmed before submitting)

- [x] Base rate reported alongside Precision@50 (Section 5) — the label is close to balanced (~54% declining), so the 0.74 score is not an artifact of a skewed majority class.
- [x] Language throughout uses observed / measured / directional / decision-support framing — no causal claims without an experiment or causal design.
- [x] No claim of having predicted or reverse-engineered Google's algorithm anywhere in this report.
- [x] No client-identifying details anywhere — only pseudonymous `content_id` / `client_id` hashes, used for grouping only.
- [x] Numbers in this report match a fresh re-run of the committed notebooks (client-holdout Precision@50 = 0.74 baseline = 0.56, leak-test AUC gap = 0.370, random-vs-grouped split gap = 0.98 vs 0.74).
