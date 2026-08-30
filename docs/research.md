# Research synthesis — RSNA Knee Abnormality Detection 2026

*Produced 2026-08-28 by an 18-agent research workflow (8 researchers → 8 skeptics → synthesis → critic); critic fixes applied; numbers marked (notebook, not re-read) could not be verified.*

*Compiled 2026-08-28 from eight verified research passes (knee-MRI literature, prior RSNA winners, weak-label construction, backbones, fine-tuning recipe, MRI preprocessing, public competition artefacts, data-pipeline engineering). Every claim below is either tied to a citation or marked **[our hypothesis]**. Numbers quoted from Kaggle notebooks whose bodies could not be re-read by the skeptic pass are marked **(notebook, not re-read)**. The noise floor from CLAUDE.md governs throughout: a gold-58 macro-AUC difference under 0.05 is inconclusive (per-label Hanley–McNeil SE ≈ 0.09; a 3,000-rep study-level bootstrap of the teacher's gold macro-AUC gives SD 0.017), public-LB differences < 0.005 are noise, OOF-vs-teacher differences over 4,407 studies below ~0.01 are noise (asserted; P-02 step 1 measures it).*

---

## 1. Executive summary — the 10 decisions this research most strongly supports

| # | Decision | Strength of evidence | Why |
|---|---|---|---|
| 0 | **Targets must be probabilities, not rank percentiles (fixed today, v02).** `rank(pct=True)` with average-rank ties put every confident negative at 0.28–0.39 on labels where most sources say exactly 0 (MCL, Lateral OA, Baker's); no study on any label (0%) had a target < 0.1 before the gold override, while gold rows sat at hard 0/1. BCE fits the value, so the network was taught "definitely absent" = 0.3. Now the mean of source probabilities (2–72% of studies < 0.1 per label: Synovitis 2%, Baker's 26%, MCL 72%). | **Confirmed by measurement** — our own `src/build_targets.py` (target quantiles logged 2026-08-28); teacher gold macro-AUC 0.8948 (probability mean) vs 0.8934 (rank blend) — that delta is noise, the scale fix is the point. Critic item 1. | Rank space is right for *scoring* and for *ensembling predictions*; it is wrong for a *training target*. The rare labels the plan protects were exactly the ones being pulled toward 0.3. |
| 1 | **Build the uint8 preprocessing cache first — after a decode benchmark.** One CPU/GPU kernel, ~16 min for all 4,407 studies at 224 px, 7–11 GB at K=6–9 (built as K=16@224 ≈ 21 GB, two shards of ~10.6 GB), then ~100 s/epoch instead of 6–8 h/fold. Our "4.5 s/study" was an **extrapolation from 8 passes**; kernel v3 measured 2.08 s/study training at 2 slices/slot, `num_workers` 0, and 153 s per 100 test studies per fold. | **Very strong** — verified log timestamps ([hida1211 log](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4)), verified plan/measurements ([homeshwarnelakurthi PLATFORM.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)), every RSNA winner 2019–2024 trained from pre-extracted arrays. | Every other decision (epochs, slices, resolution, second backbone) is blocked by the per-epoch DICOM decode we measured at ~4.5 s/study. |
| 2 | **Backbone LR 2e-5 with per-block layer decay 0.75 and EMA 0.998 (implemented today as v02; v01 was 5e-5 uniform, no EMA).** 10% linear warmup + cosine, grad-clip 1.0 after `scaler.unscale_`, wd (now 0.02, none on bias/LayerNorm) and head dropout 0.1 were **already in the pipeline** — an earlier draft of this document wrongly listed them as missing. LR is stated for effective batch 4 studies (1 × accum 4, ~36 slot-slices each); with batch 8 from the cache the sweeps' 1e-5–2e-5 transfers directly ([How to train your ViT](https://arxiv.org/html/2106.10270) LR scaling; pilkwang 8e-6 @ batch 8). | **Strong** for the LR direction (three medical DINOv2 sweeps land at 1e-6–2e-5: [MedMNIST FM benchmark](https://arxiv.org/html/2501.14685v1), [DINOv2 radiology study](https://arxiv.org/html/2312.02366v3), [Medical Slice Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)); **medium** for LLRD (MAE convention, no medical ablation). | Uniform 5e-5 on a pretrained ViT-S with noisy soft targets is the catastrophic-forgetting regime described by every recipe source; the change is free. Still untested: a 2-seed repeat to measure run-to-run OOF noise before any A/B, then LR+EMA vs +LLRD as separate arms. |
| 3 | **Keep DINOv2 ViT-S/14 fine-tuned end to end with attention pooling over slices; do not freeze, do not go 3D, do not go ViT-B first.** | **Strong** — the only knee comparison that clears noise: MST-DINOv2 0.85 vs 3D ResNet50 0.69, p=0.001; frozen 0.79 ([MST](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)); ViT-S≈B in three independent sources; query attention > LSTM/Transformer on DINO features ([AnyMC3D Table 6](https://arxiv.org/pdf/2512.12887)). | Our architecture class is right; compute should go to data (slices, epochs, folds), not size. |
| 4 | **Implement laterality normalisation from DICOM geometry** (image-centre x sign with 20 mm dead zone; mirror coronal/axial, reverse sagittal order). | **Strong** — Laterality tag missing on 50.7% of series, geometry rule 97–98% agreement, verified in [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection) and independently implemented by three public pipelines. | 5 of 12 labels are side-specific; half the corpus currently enters mirrored relative to the other half. Anatomical correctness fix; AUC delta unmeasurable on gold. |
| 5 | **Add a site proxy (language + Manufacturer + Model) to fold grouping and report grouped vs random OOF.** | **Medium-strong** — knee external-validation drop 0.05±0.03 on a 23-condition slice transformer ([Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-12052-8)); secondary: MRI has the largest cross-scanner AUC drop of any modality (−0.097, but that figure is *prostate* Siemens→Other, [Guo et al.](https://arxiv.org/pdf/2409.04368)); one ResNet34 run on this corpus showed grouped 0.705 vs random 0.841 ([EXPERIMENTS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection), verified); language is a near-perfect vendor proxy. | Our report-text-only grouping likely inflates OOF and would inflate any capacity-increasing change. |
| 6 | **Judge experiments on OOF-vs-teacher over 4,407 studies for breakage/epoch, and on gold + LB for direction; never gate on gold deltas < 0.05 or LB < 0.005.** | **Strong** — arithmetic ([Andre et al.](https://arxiv.org/html/2601.17103), Hanley–McNeil), plus a public note that OOF-vs-LLM under-reads image-side gains (OOF +0.005 vs LB +0.017 for 224→336; **notebook, not re-read**, [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee)). | Without this discipline the fork-and-republish race repeats itself. |
| 7 | **Raise slices per slot (6 → 12–16) and cache with a 130 mm physical centre crop, per-series 1/99 percentile normalisation, MONOCHROME1/Rescale handled first.** | **Medium** — winners used 24–32+ slices per volume; 130 mm covers 99.57% of series (verified FINDINGS.md); normalisation is public consensus. No knee A/B of 6 vs 16 slices exists. | Slices are the cheap (linear) axis once cached; focal findings sit on a few slices. |
| 8 | **Replace concat-of-slots → linear with per-label masked attention over slot (and slice) tokens.** | **Medium** — plane/sequence specialisation per finding is documented ([MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), [CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)); Azcona found negative transfer with one shared vector; cross-slot attention is what the public leaders use. Untested with weak labels. | Cheap, handles missing slots correctly, tests as a config flag on the same cache. |
| 9 | **Fix the weak labels where silence dominates (synonym-aware, graded extraction if re-labelling with an open-weights model inside Kaggle). Do not add more LLM sources or fancier aggregators. The Synovitis←Effusion back-fill is NOT a confirmed win on our blend.** | **Strong** for "more sources don't help": our label audit (`artifacts/label_audit.md`, 2026-08-28) found hans_v4 and sol56 make identical decisions at the 0.5 cut (agreement 99.45% over all 4,407 studies; error-φ = **1.000 on gold for every label**; raw values differ, so this is consistent with — not proof of — the v4 blend including the sol56 table) and mean pairwise error-φ 0.88 (literature: frontier-LLM panels ≈ 0.39, ≈ 2.2 effective votes, [2605.29800](https://arxiv.org/html/2605.29800)); either way ~1.5 effective votes, and our 0.002 spread is 🔁 INCONCLUSIVE in experiments.md. **🔁 INCONCLUSIVE** for the Synovitis back-fill: the public card's +0.11 ([stevenleehans card](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)) was from a 0.678 baseline; on our blend gold 0.788 → 0.729, Δ −0.059 [95% CI −0.164, +0.042] — the CI spans zero. | The 58 labels are the whole supervised signal; the weakest teacher labels are the high-silence ones (Synovitis UNK 84%, Fracture 56%, Baker's 46%). Coverage is worst in Spanish (Fracture UNK 80%, Lateral Meniscus 40%). |
| 10 | **Default submission is 5 folds × fixed epoch × EMA of a single family (DINOv2-S), rank-mean over folds, no best-epoch selection. A second family (licence-clean timm ConvNeXt/EffNetV2, or RadImageNet R50 behind a flag) and slice-window TTA are candidates, not defaults.** | **Medium** — every RSNA winner ensembled across families by CV ([TheoViel](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma), [Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)); DINOv2 loses to ImageNet CNNs on some clinical MRI ([2402.07595](https://arxiv.org/abs/2402.07595)); geometric TTA hurts in medical settings ([2604.09697](https://arxiv.org/html/2604.09697v1)); RadImageNet weights carry **no stated weight licence** on GitHub (code MIT, article CC BY 4.0) — treat as restrictive until radimagenet.com T&C are read; brainstorm.md/CLAUDE.md state CC-BY-NC-SA-4.0 and need correcting once that is settled. Slice-window TTA is **[our hypothesis]** (no source shows it helps). | Diversity, not size; every blend gain will be inside our noise floor so include a member only if its own OOF is within ~0.02 of DINOv2-S. |

---

## 2. Per-question findings

### 2.1 Knee-MRI deep learning literature (architecture, pooling, planes, difficulty)

**What we learned**

- 2D-per-slice encoder + learned aggregator beats native 3D: MST-DINOv2 0.85±0.04 vs 3D ResNet50 0.69±0.05 on MRNet meniscus (p=0.001), the only knee comparison that clears noise; transformer vs mean pooling 0.85 vs 0.82 (p=0.31), frozen 0.79 (p=0.14), no slice positional embedding 0.85 vs 0.83 (p=0.41) — direction only. LR 1e-6 for DINOv2 at batch 2. [Medical Slice Transformer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)
- Plane preference per finding is model-dependent: MRNet found axial PD for meniscus/abnormality and coronal T1 for ACL; Azcona found axial best for both; ELNet used coronal for meniscus. Must be measured, not assumed. [MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), [Azcona](https://arxiv.org/abs/2010.01947), [ELNet](https://arxiv.org/abs/2005.02706)
- Pretrained ResNet18 + per-task tuned geometric augmentation (probabilities 5–90%) gave the best MRNet-val results (0.934 avg); from-scratch multi-plane 0.858 and multi-task 0.828 with "negative transfer". [Azcona](https://arxiv.org/abs/2010.01947)
- The closest 12-label analogue (CoPAS, 3D CNN, arthroscopy labels, ~800 train) reaches mean AUC 0.81: ACL 0.95, meniscus 0.76, effusion 0.79, MCL 0.78; contusion 0.82 → 0.70 without T1W/T2W non-fluid contrast. [CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)
- Bone-marrow oedema is the hardest classic finding even at 0.5 mm isotropic with ROI crops (AUC 0.83 vs 0.90–0.93). [Astuto](https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/)
- Lateral meniscus is ~0.1 AUC harder than medial even with 18,520 training studies (0.78 vs 0.88). [Fritz](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)
- ViT-T/S/B were 0.952/0.954/0.954 on MRNet ACL; in-domain SSL did not beat ImageNet AlexNet. [SB-SSL](https://arxiv.org/abs/2208.13923)
- 2D CNN not worse than 3D for ACL staging (p=0.27). [Namiri](https://arxiv.org/abs/2003.09089)
- 23-condition slice transformer on 3,121 studies: AUC ≥ 0.85 on 8/23, weakest PCL/LCL/tibial cartilage/meniscal degeneration, external drop 0.05±0.03. Dedicated synovitis model 0.83 internal / 0.76 external, radiologists 0.77. [Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-12052-8), [synovitis](https://pubmed.ncbi.nlm.nih.gov/37951778/)
- Zero-shot external drops of 0.05–0.10 are typical; retraining recovers most. [MRNet], [CoPAS], [Eur Radiol 2025]
- BatchNorm trained with tiny effective batch diverged after 10–15 epochs in ELNet; matters only if a CNN member is added. [ELNet](https://arxiv.org/abs/2005.02706)

**What does NOT work**: native 3D CNN as primary model; frozen backbone as speed shortcut; from-scratch multi-plane/multi-task trunks; horizontal flip (swaps medial/lateral); untuned or maximal augmentation; expecting architecture to lift Synovitis/BME far above ~0.8; ViT-B over ViT-S; in-domain SSL as a first priority; treating 120-exam MRNet-val deltas as real.

**Open questions**: which slot carries which label (needs OOF slot-masking ablation); resolution value for focal labels; per-label queries vs shared vector under weak labels; realistic OOF-to-private drop; whether laterality normalisation closes any of the lateral-meniscus gap.

### 2.2 Prior RSNA Kaggle winners (2019–2025)

**What we learned**

- All classification winners used a 2D backbone on 2.5D slices + learned sequence/attention fusion over 24–32+ slices ([Nischaydnk 2023 1st](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution): 96 slices → (32,3,384,384); [darraghdog 2022 3rd](https://github.com/darraghdog/RSNA22); [brendanartley 2024 2nd](https://github.com/brendanartley/RSNA-2024-Competition): middle 24 frames → LSTM → attention; [darraghdog 2019 2nd](https://github.com/darraghdog/rsna)).
- On DINO slice embeddings, learnable-query attention pooling (0.962) > Transformer (0.950) > mean (0.958 avg) > LSTM (0.903); DINOv2 vs DINOv3 differences 0.002–0.008 both ways; larger ViT +0.008–0.029 on CT. [AnyMC3D](https://arxiv.org/pdf/2512.12887)
- Winners trained 10–40 epochs (TheoViel: maxvit 40 ep at 4e-5; coatnet 20 ep at 2e-5; RNN 10 ep at 4e-5). [TheoViel](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma)
- Stage-1 embeddings + cheap stage-2 heads (2019 1st/2nd) make many heads/seeds affordable, but stage-1 backbones were fine-tuned first. [SeuTao](https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection)
- Heavy geometric + photometric augmentation was universal (Perspective, flips, rotate ±25, ShiftScaleRotate, blur, cutmix). Noise-only is weaker than every top solution.
- 384 px was the modal whole-slice size; crop-then-resize gave the resolution gain (yu4u: bw 0.833→0.881 with valid-region crop at 384). [yu4u deck](https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui)
- Views were dropped when they moved CV < 0.01–0.02 (brendanartley axial T2).
- Soft/pseudo labels and label denoising were load-bearing (yu4u: max(logit, GT) relabel twice, bw 0.881→0.917).
- Ensembles were folds × seeds × backbones, equal weights by CV; public ≈ private when CV was patient-grouped.
- 2025 aneurysm (segmentation + 3D on multi-A100) does not transfer. [k951286 deck](https://speakerdeck.com/k951286/kaggle-rsna-intracranial-aneurysm-detectionkonpe-fan-sheng-hui), [MIC-DKFZ](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)
- MRI normalisation in a 12k multi-vendor benchmark: per-volume z-score or 0.5–99.5 percentile clip. [AnyMC3D Table 3]

**What does NOT work**: LSTM/GRU heads on DINO features; frozen FM + linear head as the model (MedImageInsight 0.785 frozen vs 0.894 adapted); 3D/nnU-Net-first; H-flip in either variant — **with** medial↔lateral label swap (MCL has no lateral counterpart label; medial/lateral OA and meniscus would swap) or **without** swap (mirroring a right knee yields a left knee, so after laterality canonicalisation it re-introduces the chirality the normalisation removed). Both are dead-ended until a specific ablation is designed; focal loss + oversampling for a 16% positive rate; DINOv2→DINOv3 as an accuracy gain; public-LB-selected blends.

**Open questions**: T4 fit for 16 slices × 6 slots with checkpointing; the 2024 1st (Avengers) recipe; self-distillation on already-soft labels; hidden test size and 9 h limit.

### 2.3 Building targets from multilingual LLM report labels with 58 gold

**What we learned**

- Report labels systematically disagree with image labels; a labeller trained toward image truth gained 0.14 mean F1 and improved downstream image models. [VisualCheXbert](https://arxiv.org/abs/2102.11467), [Jain et al.](https://arxiv.org/abs/2104.00793)
- Open-weights LLMs match GPT-4o on extraction (Mistral-Large 92.6 vs GPT-4o 92.4 macro-F1; CheXpert rules 73.1), including German. [Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.240895), [JIIM 2025](https://pubmed.ncbi.nlm.nih.gov/40325326/). A 70B model at 4-bit does **not** fit 2×T4; 32B AWQ is the ceiling.
- Weak labels help until a gold-only classifier matches labeller accuracy; with 58 gold and a 0.89-AUC teacher we are orders of magnitude below that crossover. [2605.24771](https://arxiv.org/html/2605.24771)
- If gold is used for training, only as a linear/ridge stage (LP-FT); Platt/thresholds are a metric no-op. [BoxWRENCH](https://arxiv.org/html/2501.07727), [Kumar et al.](https://arxiv.org/abs/2202.10054). **Our resolution of gold's role:** gold is *validation* (held out per fold, reported not gated) and enters training only through `gold_weight` in the weak-label BCE — there is **no gold fine-tuning stage**. Whether 8× over-weighting (1.3% of studies, ~10% of loss mass) helps or memorises is untested: run a `gold_weight ∈ {1, 3, 8}` arm and note that gold weight also biases the fold-mean OOF-vs-teacher curve.
- **Target scale matters for BCE (measured 2026-08-28, critic item 1 confirmed).** Our v01 targets were `rank(pct=True)` percentiles with average-rank ties; on labels where most sources say exactly 0 every confident negative landed at 0.28–0.39 and no study on any label (0%) had a target < 0.1 before the gold override, while gold sat at hard 0/1. Replaced by the mean of source probabilities (2–72% of studies < 0.1 per label: Synovitis 2%, Baker's 26%, MCL 72%): teacher gold macro-AUC 0.8948 vs 0.8934 (Δ 0.0014, noise — the scale, not the AUC, is the fix). Rank space is for scoring and ensembling predictions, never for a training target. `src/build_targets.py` (`prob_blend` used; `rank_blend` kept as diagnostic).
- **Label audit (`artifacts/label_audit.md`):** hans_v4 and sol56 make identical decisions at the 0.5 cut (agreement 99.45% over all 4,407 studies; error-φ = 1.000 on gold for every label; raw values differ — consistent with, not proof of, the v4 blend including sol56); hans_v4~pilkwang 95.4%, pilkwang~sol56 95.2%; mean pairwise error-φ 0.88 vs ≈ 0.39 for independent LLM panels — ~1.5 effective votes. Silence: pilkwang is the only source that flags it (UNK); on UNK rows hans_v4 averages ~0.25 (many distinct values), the blend ~0.18, and the confidence weight 0.69 vs 0.80–0.89 on addressed rows — silence is barely down-weighted and looks like a confident negative. Source agreement by language (Spearman hans_v4~pilkwang): en 0.83, bg 0.67. Largest gold-vs-blend positive-rate gaps: Synovitis 47% vs 13%, Lateral Meniscus 40% vs 16%, Fracture 31% vs 7%, ACL 41% vs 21%. Spanish reports have the worst coverage (Fracture UNK 80%, Lateral Meniscus UNK 40%, Effusion UNK 29%).
- LLM panels have ~2.2 effective votes out of 9 (pairwise φ 0.39); Dawid–Skene < majority vote; best single judge ≈ panel. Matches our 3-source +0.0007 measurement. [2605.29800](https://arxiv.org/html/2605.29800)
- Aggregator choice is second-order (MV ≈ DS > Snorkel ≈ FlyingSquid). [BoxWRENCH], [WRENCH](https://arxiv.org/abs/2109.11377)
- Mapping uncertainty language to per-sample smoothing improved CXR mean AUC 79.6→84.1 on 340k images, biggest on rare classes. [Rep-GLS](https://arxiv.org/html/2508.02495)
- Under real-world imbalanced clinical noise, sample-selection methods (Co-teaching, DivideMix, DISC) collapse minority classes; only SCE/CDR marginally beat CE. [LNMBench](https://arxiv.org/html/2512.09315v1)
- Translate-then-extract costs precision (0.672→0.625) for a small F1 gain with multilingual models. [2403.10258](https://arxiv.org/html/2403.10258), [2602.21374](https://arxiv.org/html/2602.21374)
- Percentile bootstrap is the robust small-n AUC CI; classification needs hundreds–thousands of cases for reliable CIs. [Andre et al.](https://arxiv.org/html/2601.17103), [Feng et al.](https://pubmed.ncbi.nlm.nih.gov/26323286/)
- The clean set should be reserved for model selection, not LF engineering. [BoxWRENCH], [spinal MRI LLM labels](https://arxiv.org/html/2410.17235)
- Reports help only where the target is well represented in text; "not mentioned = 0" creates systematic false negatives for findings dictated under other names (bone-marrow oedema → Contusion; synovial thickening → Synovitis). [2510.24385](https://arxiv.org/abs/2510.24385)

**What does NOT work**: more label sources / DS / Snorkel / CARE; noisy-label sample selection; full FT on 58 gold; recalibration on gold; translate-first with a multilingual model; sub-3B extractors; 70B on 2×T4; hosted LLM APIs.

**Open questions**: gold severity protocol per finding (pilkwang notebook, unread); whether public scores are graded or binary; Qwen-32B-AWQ throughput on 2×T4; FN-vs-FP composition of teacher errors on the weak labels.

### 2.4 Backbone choice and licensing

**What we learned**

- Fine-tuned DINOv2 > ImageNet ResNet > 3D on knee MRI (0.85 / 0.78 / 0.69); frozen 0.79. [MST](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)
- End-to-end FT beats linear probing on all 12 MedMNIST sets; DINOv2 ViT-B beats best CNN; BiomedCLIP/MedSAM lag general models (71.9/72.3 vs 89.7%); encoder LR 1e-5 (ViT) vs 1e-4 (CNN). [2501.14685](https://arxiv.org/html/2501.14685v1)
- RadImageNet R50 > ImageNet R50 on small knee tasks (ACL 0.97 vs 0.91); unfreezing all layers best in 24/24 scenarios. [RadImageNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)
- LoRA/BitFit matched full FT within 0.002 AUROC on CXR with ViT-g; but [2510.07191](https://arxiv.org/abs/2510.07191) found full FT > PEFT at scale, and ConvNeXt-B > ViT-B/16; DINOv3 beats DINOv2 only at 512 px; 1024 px rarely helps.
- Model size and resolution scaling are non-monotonic for medical transfer; ViT-S ≈ ViT-B in three sources. [2509.06467](https://arxiv.org/html/2509.06467v3), [2402.07595](https://arxiv.org/abs/2402.07595), [2606.25989](https://arxiv.org/html/2606.25989)
- LLRD protects low-level features (uniform LR = "catastrophic forgetting" of SSL features); decay values 0.75/0.65 are BEiT/MAE convention, not medical evidence. [2606.25989], [MAE](https://arxiv.org/html/2111.06377)
- Medical-specific pretraining wins only below ~5 examples per class. [2408.08058](https://arxiv.org/abs/2408.08058)
- Licences: DINOv2 Apache-2.0; DINOv3 custom licence (commercial OK, must redistribute licence, gated); RAD-DINO MIT but research-only wording and CXR-only; **RadImageNet weights carry no stated licence** (code MIT, article CC BY 4.0, data "by request"). [dinov3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), [RadImageNet GitHub](https://github.com/BMEII-AI/RadImageNet)
- OrthoFoundation (DINOv3-L continued-pretrained on 1.2M knee images) ranks first on ten internal MRI tasks but weights are not released. [2601.18250](https://arxiv.org/abs/2601.18250)
- DINOv2 position embeddings are interpolated from 518 px pretraining at any other resolution (HF `Dinov2Model` `interpolate_pos_encoding`), so a 224 vs 336 comparison changes the embedding grid as well as the pixels; DINOv2-**with-registers** variants (`facebook/dinov2-with-registers-small`) remove the high-norm attention artefacts that can distort attention pooling. [Darcet et al.](https://arxiv.org/abs/2309.16588). Not evaluated by any knee source; a cheap swap to test.
- On this competition: DINOv2-S/14 224 baseline 0.809 public; EfficientNet-B0 320 mean-pool 0.664. [pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1), [JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)

**What does NOT work**: frozen features as the final model; BiomedCLIP/MedSAM/RAD-DINO; DINOv3 at 224; ViT-B/L at 4k studies; 1024 px; EfficientNet-B0; OrthoFoundation (not public); uniform high LR without LLRD/warmup.

**Open questions**: 448 px + 4 slices vs 224 + more slices; ConvNeXt-Small vs RadImageNet as diversity member; radimagenet.com T&C and the winner-licence clause; in-notebook DINO continued pretraining; whether the 0.809 baseline's 5,507 s runtime is real.

### 2.5 Fine-tuning recipe

**What we learned** (all verified unless noted)

- Backbone LR: medical sweeps land at 1e-5 (MedMNIST), ≤ 2e-5 (DINOv2-radiology), 1e-6 at batch 2 (MST); lr 1e-3 full FT collapsed to < 40% ([dinov2 #276](https://github.com/facebookresearch/dinov2/issues/276)). Head 1e-3 everywhere.
- MAE fine-tune recipe: AdamW, LLRD 0.75, wd 0.05, cosine, 5% warmup, drop-path 0.1 (B/L). [MAE](https://arxiv.org/html/2111.06377)
- Warmup + cosine + grad-clip 1.0 is universal ([How to train your ViT](https://arxiv.org/html/2106.10270); Nischaydnk) — and already in our pipeline (10% warmup, clip 1.0 after `scaler.unscale_`). LR scales with effective batch (same source): our sweeps' 1e-5–2e-5 were at batch 8–64; MST used 1e-6 at batch 2; we run 4 studies (≈36 slot-slices each) per step.
- Layer decay 0.75 per block: block 0 ≈ 0.75¹² ≈ 0.03× the top block for a 12-block ViT-S. [BEiT](https://arxiv.org/abs/2106.08254) Appendix; timm `param_groups_layer_decay`. EMA decay 0.998 with warm-up: timm `ModelEmaV3` docs; the "EMA is robust to label noise" claim rests on a single arXiv study — medium.
- Label-noise training has three phases; stop before memorisation; EMA is markedly more robust to label noise and helps early. [Label Wave](https://arxiv.org/html/2502.07551v1), [EMA study](https://arxiv.org/html/2411.18704), [ISIC note](https://zenn.dev/morim34/articles/bfa2465defee06) (EMA decay 0.995–0.9999, +0.0015 pAUC).
- Model soups / checkpoint averaging from the same init match or beat the best checkpoint at zero inference cost. [Model soups](https://arxiv.org/html/2203.05482)
- Heavy dropout/drop-path/AugReg helps large ViTs on long schedules and adds nothing when transferring. [How to train your ViT]
- Plain BCE is within ~0.01 mAUC of ASL/RAL/focal on CXR-LT (table values not re-read). [RAL](https://arxiv.org/abs/2308.05542)
- AUC-margin loss as a 2-epoch stage 2 won CheXpert (0.9305) and tolerates noise; needs hard labels and LibAUC offline. [Yuan et al.](https://arxiv.org/html/2012.03173), [LibAUC script](https://raw.githubusercontent.com/Optimization-AI/LibAUC/main/examples/scripts/07_optimizing_multi_label_auroc_loss_with_densenet121_on_chexpert.py)
- Multi-task gradient balancing does not beat a weighted loss sum. [Xin et al.](https://arxiv.org/abs/2209.11379)
- Geometric TTA degraded 11/12 MedMNIST pairs; flips underperformed on knee OA radiographs even after mirroring. [2604.09697](https://arxiv.org/html/2604.09697v1), [2311.06118](https://arxiv.org/html/2311.06118)
- Epoch/recipe selection must use the ~880-study OOF, never 11–12 gold per fold (our experiments.md).

**What does NOT work**: focal/ASL/RAL over BCE (inside noise); GradNorm/PCGrad; frozen backbone; LR 1e-3; heavy regularisation on a small ViT; geometric TTA; best-epoch on gold; pos_weight; label smoothing on soft targets; study-level mixup in v1; extending epochs before the cache exists.

**Open questions**: run-to-run OOF noise (2-seed repeat, same config — must be measured before any A/B, since the 0.01 OOF floor is asserted, not measured; [Picard 2021](https://arxiv.org/abs/2109.08203) for magnitude); LLRD 0.75 vs uniform 2e-5 as a separate arm (expect inconclusive); OOF-vs-teacher optimum vs gold optimum at 8–10 epochs; EMA decay 0.999 vs 0.998; AUCM with soft targets (needs hard labels — thresholding soft targets reintroduces the cut-point problem); fold soup vs rank-mean for the efficiency track **[our hypothesis]** — souping across folds is different-data/same-init and untested here; 3 folds + 2nd backbone vs 5 folds **[our hypothesis]**.

### 2.6 MRI preprocessing and data design

**What we learned**

- Laterality tag missing on ~half the corpus (vendor-wide, sometimes empty string); centre-x sign rule 97.4% (98.5% with 20 mm dead zone); IPP corner rule only ~59%; GE sub-sample unreliable. [pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) (pulled), [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection) (verified), [PMC6646614](https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/)
- Normalisation is plane-specific: flip W on coronal/axial; **reverse slice order** on sagittal (stacks are order-reversed, not mirrored). [pilkwang], [JunhaoLiXD]
- A 130 mm physical centre crop is inside the FOV of 99.57% of series; 160 mm is skipped on 60%; FOV median 160 mm, range 70–320. [pilkwang], [FINDINGS.md]
- 224→512 px on radiographs: median +1.05 pp AUROC, concentrated in small focal findings, but DINOv2-ViT was nearly flat (VinDr 89.2→89.1); 512→1024 gave 2/15 significant gains and ViT losses of −6.6/−7.9 pp. [2510.07191](https://arxiv.org/abs/2510.07191)
- Per-series 1/99 percentile → [0,1] → ImageNet mean/std is the consensus; N4/Nyul have only radiomics/segmentation evidence; z-score does not remove scanner shift. [pilkwang], [MRNet], [2307.03827](https://arxiv.org/abs/2307.03827), [Guo et al.](https://arxiv.org/pdf/2409.04368)
- Apply RescaleSlope/Intercept and MONOCHROME1 inversion before percentiles; MONAI Deploy shipped the un-inverted bug. [Innolitics](https://innolitics.com/articles/medical-imaging-best-practices/), [MONAI issue #282](https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/282)
- Fixed-count sampling over a central band of the spatially ordered stack + learned aggregator is standard; per-plane bands (sag ~0.08–0.92, ax 0.10–0.90, cor 0.20–0.80) because menisci/MCL sit near sagittal stack ends (**notebook, not re-read**: romanrozen).
- Triplet gap in mm rather than index is **[our hypothesis]**; no source ablated the gap. With 3–4 mm sagittal slice thickness `round(3 mm / spacing)` always gives g=1 (adjacent slices), a behaviour change from our current gap 2 — an ablation, not a default.
- Laterality canonicalisation needs a rule for the tag-vs-geometry *conflict* case (2–3% of tagged series at 97–98% agreement): trust geometry or drop laterality for that series, and count it. Sagittal "reverse order" only canonicalises stack direction; in-plane orientation (`ImageOrientationPatient` direction cosines, `PatientPosition` HFS/FFS) must be canonicalised for all planes too. [PMC6646614], pydicom orientation docs.
- Safe augmentation = small rigid + intensity jitter, zoom-in only; H-flip undoes laterality normalisation; V-flip is off-distribution; Gaussian noise alone did not help cross-scanner generalisation. [pilkwang], [Guo et al.]
- Fat-sat fluid slots carry oedema/effusion/synovitis; T1/non-FS carry meniscal morphology and bone outline; T1 slots (fill 50–62%) are the first to cut if needed. [Maarek 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12362699/), [pilkwang]
- MRI cross-manufacturer AUC drop −0.097 (**prostate** Siemens→Other 0.791→0.505) vs X-ray −0.067, CT −0.02 — a magnitude reference only; the knee-specific figure is the 0.05±0.03 external drop in [Eur Radiol 2025]. [Guo et al.](https://arxiv.org/pdf/2409.04368)
- Query attention over slices beats mean/median/LSTM/Transformer; more views help on anisotropic volumes. [AnyMC3D](https://arxiv.org/html/2512.12887), [TomoGraphView](https://arxiv.org/html/2511.09605)
- Do not resample anisotropic stacks to isotropic; detect 3D isotropic acquisitions and sample in mm. [AlignShift](https://arxiv.org/abs/2005.01969)

**What does NOT work**: trusting/defaulting the Laterality tag; IPP-corner side rule; pixel-flipping sagittal slots; H/V flip augmentation; zoom-out with padding; ≥160 mm crops; filename/InstanceNumber ordering; relaxing T1 predicates to PD/T2; N4/Nyul/VOI-LUT; > 512 px; every-slice processing without a cache; joint detectors.

**Open questions**: our own FOV histogram; Kaggle output cap → (slices, resolution) pair; 224 vs 336 for DINOv2-S on meniscus/fracture/contusion; fraction of studies with |centre_x| < 20 mm and no tag; whether InstitutionName/Manufacturer survive anonymisation; count of 3D isotropic series; per-label masked slot attention vs concat under weak labels; OA labels' dependence on T1 slots.

### 2.7 Public competition artefacts

**What we learned**

- Gold-58 → public LB offset ~+0.04 for fine-tuned DINOv2 systems (0.824→0.866, 0.857→0.903); frozen head 0.771→0.776; OOF-vs-LLM under-reads image gains (**notebook, not re-read**, [sadamtorres](https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee)).
- Random K-fold inflated by site memorisation: grouped 0.7049 vs random 0.8412 on a ResNet34 (verified [EXPERIMENTS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); Cyrillic 100% Philips, Dutch/German/Greek 100% Siemens.
- Public DINOv2 recipes: unfreeze last 6 blocks, LR 8e-6 / 1e-3, wd 0.02, 10 epochs, batch 8, OneCycle, rot 8°/scale 0.08/shift 0.05/intensity 0.1, no flips ([pilkwang], pulled). Fine-tuned vs frozen at 224: MedMen 0.679→0.850, MCL 0.708→0.825; 224→336 +0.017 LB at 2.25× FLOPs (**notebook, not re-read**).
- Cache: ~1–1.5 h CPU (4 procs) or ~16 min with 12 threads on a GPU kernel; 224 px × 16 slices = 15.9 GB (verified PLATFORM.md); public caches exist (alenic JPEG 224: 11.2 GB; barun2104 3D: 17.4 GB).
- The 0.94–0.95 LB is one lineage: 20 DINOv2-S ckpts + 5 DINOv3-S folds + RadImageNet R50 heads, per-finding rank weights, RadImageNet excluded from Baker's/Fracture, author-labelled overfit ([mattiaangeli](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled), [tonylica](https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble); **not re-read**). pilkwang's 20-ckpt ensemble alone 0.891.
- LLM labels: ~25% of cells exactly 0.5 (unaddressed); Synovitis unaddressed in ~84–88% of reports while 27/58 gold are positive; the card reports an Effusion back-fill lifting Synovitis gold AUC 0.678→0.790, **but on our blend it does not reproduce**: 0.788 → 0.729, Δ −0.059 [95% CI −0.164, +0.042] (`artifacts/label_audit.md`, 2026-08-28) — the public +0.11 was from a 0.678 baseline that our 3-source blend already exceeds; regex 0.8136 vs LLM v4 0.8927 (card figures match our experiments.md). [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
- The card's "rank-blend destroys 0.5 semantics" warning **did apply to us — to the targets, not the weights**. `confidence_weights()` reads raw source probabilities, but v01 wrote `rank_blend()` percentiles straight into `targets`, putting confident negatives at 0.28–0.39 (no study on any label < 0.1 before the gold override). Fixed 2026-08-28: targets are now the mean of source probabilities (teacher gold 0.8948 vs 0.8934, Δ inside the noise floor); `rank_blend` is diagnostic only. (Critic item 1, confirmed against `src/build_targets.py`.)
- Only 3,991/4,407 studies have fat-sat fluid series in all three planes; series have 11–320 slices, median 30. [JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)
- Inference: decode once per study, all folds on cached tensors; 5-fold ViT-B 224 ~3 h, 336 ~4 h on ~1,300 hidden studies (**not re-read**); TIME_BUDGET ~8 h, prevalence fallback, checkpoint fingerprints.
- Community stability remedies: EMA (0.997), fixed epochs, gold weight 3–8× (we use 8.0, untested against {1, 3}), rank-mean folds. Our v02 now validates and saves EMA(0.998) weights and logs per-label + OOF rows per epoch; `MODE=infer` is the submitted configuration.
- **Submission #1 (smoke) scored exactly 0.500**: the notebook produced constant predictions at rerun — the hidden test image root / `.dcm` extension were assumed rather than probed, and the fallback silently filled 0.5. Fix: resolve the image root by a shallow glob at runtime, log the fill-rate, write **no placeholder** (loud failure), and refuse to write a submission when < 90% of test studies are imaged and have ≥ 1 slot or when columns are constant.
- Efficiency LB exists (top 0.948/0.944/0.938, non-monotone in score) — formula unread. [ryanholbrook LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)
- One participant reads Rule 4.b as forbidding hosted LLM APIs; label tables derive from GPT-5.6/Gemini/Claude anyway; Qwen3-14B-AWQ reached 0.881 gold locally (**not re-read**).
- Transfer syntax: one census says 100% Explicit VR Little Endian; one notebook claims JPEG Lossless/2000 present. Mount decoder wheels regardless.

**What does NOT work**: forking the 0.95 ensemble; frozen head; resolution before adaptation; flips; corner-based or tag-only laterality; filename/InstanceNumber order; regex labels; calibrated priors with 50% floor (collapsed to base rate, JunhaoLiXD V02); random K-fold as comparison metric; best epoch on ~12 gold; SWA (no gain, card-only).

**Open questions**: Efficiency formula; Rules text; discussion threads (inaccessible); grouped-vs-random gap for a fine-tuned DINOv2; compressed syntaxes in test; hidden test size/site mix; stability of the +0.04 offset on private.

#### 2.7.1 Anatomy of a 0.936 public notebook, read in full (2026-08-30)

`crazy_good_rsna.ipynb` (prvsiyan's Apache-2.0 port of Roman Tamrazov's "DINOsaur V10"; public
**0.936**, score verified by Tian) is the first top-lineage notebook we have read cell by cell
rather than from its card. It **trains nothing**: ten cells that mount four families of public
checkpoints (tonylica `rsna-knee-bend-dinov3-0917-repro-assets`, dreaddevelopment
`raptor-knee-maxspan` / `-widedense` / `-finespacing`, metaresearch DINOv2-S), run each on the
hidden test, and rank-fuse them. Its own intermediate filenames give the decomposition:

| Cumulative stage | Public LB (stated in the notebook) |
|---|---|
| DINOv2-S/14 branch alone — 5 folds × several members, 336 px, sliding-window + affine-jitter TTA | **≈ 0.899** (`submission_public_0899.csv`) |
| + 16-slices-as-channels ViT (w 0.45) + RadImageNet R50 frozen-feature heads (w 0.5 on 10 labels) + 88-feature stacking calibrator (w 0.4) | **≈ 0.920** (`submission_transformer_0920.csv`) |
| + CoAtNet-2 rMLP @384 "Raptor" (w ≈ 0.5; **0.924 alone**) | **0.935** (`_base_0935`) |
| + per-label weights tuned on gold-58 + "clinical residual" (`clinical_moderate` profile) | **0.936** |

**Our 0.896 two-head blend is at parity with its entire DINOv2 branch.** The 0.040 gap is
families, not recipe. *Identical* to our pipeline, so not the gap: the six recovered slots by
name, 130 mm crop, geometry laterality with a 20 mm dead zone, IPP·normal slice ordering,
per-series 1–99 percentile normalisation, LLM probabilistic targets over 4,349 studies, rank-mean
fusion, 5 folds — and our per-label slot-query attention head (`SlotHead`, there with a
hand-written label→slot prior table). Same lineage as pilkwang's.

The techniques it has that we do not, with an estimate of each one's share of the gap
(**[stated]** = a number written in the notebook; **[est]** = our judgement from those numbers plus
experiments.md):

| Technique | What it is | Ours | Contribution |
|---|---|---|---|
| **Cross-family rank fusion** | DINOv2-S + 16-ch ViT + RadImageNet R50 + CoAtNet, rank-fused with per-label weights and correlation guards | one family, two heads | **≈ +0.036 of the +0.040 [stated: 0.899 → 0.935]**. Same lesson as our #6/#7 (folds +0.000 on top of heads): diversity is the whole game |
| **CoAtNet-2 rMLP @384, 64 slices/study, per-label attention over 62 triplet windows, SWA** (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; 140 mm crop; 5 slots SAG-FS 18 / SAG 14 / COR-FS 12 / COR 8 / AX 12; band 2–98 %; trained on 4,349 LLM-labelled studies, 58 gold held out) | strongest single model: **0.924 LB, no TTA, no ensemble [stated]**; gold-58 0.905–0.917 | best single 0.877 | **+0.015 in the fusion [stated: 0.920 → 0.935]**; +0.047 as a single model. Sub-attribution [est]: backbone (75 M, IN-12k supervised) +0.01–0.02 — their 45-gold panel: coatnet-384 0.9025, swin-B-384 0.8825, convnext-B-384 0.8754, convnext-L-384 0.8752, effv2-L-480 0.8716, maxvit-384 0.8438; 384 px +0.005–0.01; 64 slices at 2–98 % vs our 36 triplets at 8–92 % +0.005–0.01 ("cutting the outer slices was measurably costing accuracy on the collaterals and the lateral meniscus"); attention over all windows instead of slot-then-label ≈ +0.005; SWA ≈ +0.005 |
| **16-slices-as-channels ViT** (`in_chans=16` patch embed or a gated `DepthCompress` stem → 3 ch; slot identity injected as an extra ViT token; RoPE → DINOv3-class weights), 336 px, 5 folds, readouts `xattn`/`xres`/`clsadd` | the whole slice stack in one forward pass — a different input representation | — | **+0.005–0.01 [est]**, part of the +0.021 transformer-stack step; low ρ with triplet models by construction |
| **RadImageNet ResNet-50 frozen GAP features + trained attention heads** (2048-d tokens, 3 FS slots × 8 slices @224, 5 folds × 2 slot configs) | encoder frozen, only heads train; excluded from Baker's/Fracture | P-10 flag, unbuilt | **+0.005–0.01 [est]**; ~0.3 session. Licence unresolved (CLAUDE.md) |
| **Sliding-window + affine-jitter TTA with per-label window pooling** — max for Fracture/Contusion/Menisci/Baker's, top-2 for ACL/MCL, mean for OA/Effusion/Synovitis | every window position over the cached slices, ×2 with a jittered view (rot 8°, scale 0.08, shift 0.05, gain 0.1) | none (`predict()` is one pass) | **+0.003–0.008 on the DINO branch, ≈ +0.002–0.004 after fusion [est]**. This is P-12 plus a focal/diffuse split |
| **Stacking calibrator** — 88-feature linear model over the branch rank blocks + protocol metadata (n_series, per-plane counts, FS/fluid counts), gated per label, w 0.4 | fitted offline; coefficients embedded base64 | — | **+0.002–0.005 [est]**; fragile to a protocol-mix shift on private |
| Same-arch complement checkpoints (legacy v4 w 0.10–0.16 on 7 labels; v9 "FineSpacing" specialist) with correlation guards | | — | **+0.001–0.002 [est]**; their own note: CoAtNet + Swin-B + EffV2-L on the same input blended to **+0.001** over CoAtNet alone |
| **Gold-58-tuned per-label weights + "clinical residual"** (e.g. ACL −0.10 × mean rank of Contusion and Lateral Meniscus) | the cell calls itself "an aggressive leaderboard experiment, not an unbiased estimate of private-test performance" | — | **+0.001 [stated: 0.935 → 0.936]**. Do not copy |
| Corpus 3,155 → 4,349 LLM-labelled studies | | already done | 0 for us [stated: +0.013 gold, +0.010 LB for them] |

Sum of the estimates ≈ 0.036 + 0.003 + 0.001 ≈ 0.040, consistent with the observed gap.

**CORRECTED / EXTENDED 2026-08-30 (afternoon) — a second, cell-level re-read** (every line of cells 02–08,
calibrator payload decoded) fixes and sharpens the table above:

- **No DINOv3 code exists.** `find_dinov2(variant)` ignores its argument and always loads the mounted
  `metaresearch/dinov2/pytorch/small/1`; every "protocol model" is ViT-S/14. The DINO branch's img / slices /
  band / crop come from a `manifest.json` that is not in the notebook (code defaults 336 px, band 0.2–0.8,
  crop 130). Members are fingerprint-checked against a seeded input (`FINGERPRINT_TOL 0.002`).
- **The submitted DINO output is the NO-jitter view** (`submission_public_0899.csv`, "exact no-jitter
  public-frontier rank mean"): the affine-jitter TTA is computed when time allows and then discarded. The
  per-label window pooling that *is* submitted: max for Fracture / Contusion / both Menisci / Baker's, top-2
  mean for ACL / MCL, mean otherwise, over **overlapping** windows (every start 0..slices−3).
- **The CoAtNet "Raptor" branch has no laterality normalisation at all**, no TTA, 5 ordered slots
  (Sag fluid 18 / Sag non-fluid 14 / Cor fluid 12 / Cor non-fluid 8 / Axial any 12 = 64 slices) chosen
  from `test_series.csv` preferring the requested `Fluid_Sensitive`, MaxSpan v5 span 0.02–0.98 (`low =
  int(n·0.02)`, `high = int(n·0.98) − 1`), WideDense v4 / FineSpacing v9 span 0.06–0.94, `pos = IPP · (row ×
  col)` ordering, `apply_modality_lut` + MONOCHROME1 inversion, **per-slot-series [2, 98] percentile before
  crop**, 140 mm crop → 336 (`INTER_AREA`) → 384 bilinear per window; `K_EVAL = 62` windows over the
  concatenated volume (windows can straddle slot boundaries; **no slot embedding**); head = `LayerNorm →
  Linear(F,256) → Tanh → Dropout(0.2) → Linear(256,12)`, softmax over windows **per label**, per-label
  classifier. "swa" appears only in the v5 filename. The v4 complement enters 7 labels at w 0.10–0.16 with
  correlation guards (`corr > 0.992 → ×0.5`, `< 0.65 → ×0.4`); v4/v9 run only with **2 GPUs**.
- **16-channel branch:** timm ViT at **336 px**, `SLICE_BAND (0.12, 0.88)`, 16 evenly spaced slices ordered
  by **InstanceNumber only**, **per-slice** [1, 99] on a 4× subsample, laterality = `IPP[0] < 0 → flip cor/ax`
  (no dead zone), stems `in_chans=16` | `DepthCompress` (gated 1×1 depth blocks → 3 ch) | `SlotDepthMixer`
  (5-tap learned depth smoothing per plane), readouts `mean_max` / `attn` / `xattn` (12 queries over all patch
  tokens) / `xres` / `clsadd`; fused at **w 0.45**. This is why our linear 16-ch patch embed (`v07s`) is not
  a test of their member (P-15 retry spec).
- **RadImageNet branch:** frozen torchvision R50 GAP (2048-d), input `/127.5 − 1`, 224 px, two 5-fold head
  bundles (3 FS slots no crop; 4 slots 130 mm) + a slot-permuted "pass 2"; α 0.5 on 10 labels (**excluded
  from Baker's and Fracture**), pass-2 0.15; head = per-label queries + `MultiheadAttention(512, 8)` over
  slot × 8 position tokens.
- **Calibrator (V18):** 88 features = 6 rank blocks × 12 + 4 anatomical group means + 12 protocol counts;
  linear per label, mixed at **0.40 into 7 labels** (ACL, Medial OA, Lateral OA, PF OA, Effusion, Baker's,
  Contusion). Decoded coefficients are dominated by the label's own rank (Effusion `baseline:Effusion +0.157`);
  protocol columns ≤ 0.003 — it is a per-label reweighting, not a protocol model.
- **Fusion weights:** transformer vs CoAtNet 0.50 everywhere, 0.53–0.61 CoAtNet on ACL / MM / LM / Lateral
  OA / Fracture after the calibrator; `clinical_moderate` residual (ACL −0.10 × mean rank(Contusion, LM),
  Lateral OA +0.10, Synovitis +0.10, Fracture +0.05, …) is the porter's addition with **no stated LB**.
- Stated numbers not in the table above: live LB **0.914** for CoAtNet alone with blends 0.914–0.915
  ("ensembling is worth ~+0.001 there"); 45-gold panel coatnet384 0.9025 / swin-B 0.8825 / effv2-L 0.8716 /
  cnn336 0.8833 / convnext-B 0.8754 / maxvit 0.8438; corpus 3,155 → 4,349 = gold 0.8923 → 0.9054.

What this changed in our plan (proposals P-25, P-26, P-23 #2, P-15): the per-label window attention and
the 2–98 % band are the parts copied; the 140 mm crop / [2, 98] normalisation / no-laterality / slot-crossing
windows are deliberately **not** (a shared test-time builder and our measured +0.015 laterality gain matter
more); the calibrator, residual and tuned weights are in "Rejected without testing".

Three things it changes for us: (1) **P-10's direction is confirmed** by the strongest evidence
available, but their own panel puts ConvNeXt among the *weakest* single families and a
high-resolution, many-slice hybrid at the top — a ConvNeXt-T member is a diversity bet, not a
strength bet, and must be judged on ρ; (2) their strongest single model is 384 px with 64
slices, i.e. P-11 + P-08's K sweep, which this repo had downgraded; (3) every checkpoint they use
is a public Kaggle dataset, so mounting them as `INFER_MEMBERS`-style members is legal and the
fastest route to ~0.93 — at the cost of becoming one more fork of the ensemble whose author
expects a private shakeup. → **P-23** in proposals.md (raised to the top of the backlog by Tian,
2026-08-30).

### 2.8 Data-pipeline engineering

**What we learned**

- uint8 cache: 4,407 × 6 × 9 × 224² = 11.1 GB held in RAM by hida1211, built in ~16 min with 12 pydicom threads; then 103 s/epoch over ~3,300 studies at batch 8 with 6 blocks unfrozen (verified [hida1211 log](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4)) vs our ~4.5 s/study (extrapolated from 8 passes). **Our default is `np.load(mmap_mode='r')` on `.npy` shards, not in-RAM**: K=12@224 = 4,407×6×12×224² ≈ 15.9 GB plus DataLoader worker copies does not reliably fit ~29.8 GB, and a single kernel output is capped at ~20 GB (Kaggle docs, to cite once read), so two shards; caching at 336 (K=12 ≈ 36 GB) requires sharding across kernels/datasets. With mmap, `num_workers=0` is fine.
- Header passes are mount-latency-bound: 32 ordering threads, 16 header threads, 12 pixel threads ([pilkwang] source).
- pydicom needs pylibjpeg(+libjpeg,+openjpeg) or GDCM for JPEG Lossless/2000; Kaggle's requirements list neither; reruns are internet-off. [pydicom plugin table](https://pydicom.github.io/pydicom/stable/guides/plugin_table.html), [kaggle_requirements.txt](https://raw.githubusercontent.com/Kaggle/docker-python/main/kaggle_requirements.txt), [jirkaborovec] (pulled)
- dicomsdl ~10× faster than pydicom on a laptop micro-benchmark; used by RSNA 2023 1st. [dicomsdl](https://github.com/tsangel/dicomsdl/blob/master/README.md)
- GPU decode pays only for large images (~1× at 64², 2.5× at 512² CT, 33× at 1900 px J2K). [nvImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/samples/DICOM-pydicom.html)
- ViT-S inference is cheap relative to decode (4,617 img/s on a 3090 fp16; extrapolated ~1,000 img/s on T4). [timm benchmark CSV](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv)
- Community plans for ≥ 1,322 hidden test studies (30% of train); ~29.8 GB RAM available ([pilkwang] log).
- uint8 after percentile normalisation; np.load(mmap_mode='r') works only for .npy, silently ignored for .npz. [NumPy #5976](https://github.com/numpy/numpy/issues/5976)
- fp16 autocast + GradScaler on T4; channels_last only for cuDNN conv nets (8–35%). [PyTorch memory_format](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
- torch.compile runs on CC 7.5 but compile time and shape recompiles outweigh gains for 1–2 min of GPU inference. The cheap T4 speedup is SDPA attention (`attn_implementation="sdpa"` in HF `Dinov2Model`); memory fallback order for K=12–16 × 6 slots full FT on 15 GB: (a) `torch.utils.checkpoint` per block, (b) freeze patch-embed + first 4 blocks, (c) batch 1 with accumulation.
- Preprocess-once-to-8-bit fixed-size volumes is the winner pattern ([Nischaydnk], [gunesevitan](https://github.com/gunesevitan/rsna-2023-abdominal-trauma-detection): all train DICOMs to PNG in ~20 min).

**What does NOT work**: decoding in the DataLoader; pip install at scoring time; GPU decode for ≤ 512 px MRI; float32 caches; .npz + mmap; bf16 on Turing; channels_last for ViT; torch.compile by default; large ensembles for the efficiency track.

**Open questions**: efficiency formula and Base; hidden test size; TransferSyntaxUID census; CPU- vs I/O-bound decode on the mount; measured T4 ViT-S throughput; per-epoch time for a fully unfrozen ViT-S; dual-T4 use at scoring; mounting the train cache in the scoring notebook.

### 2.9 Gaps the critic found (not covered by the eight research passes)

- **Hidden-test vs train distribution (site/vendor/language).** Nothing estimates whether the test set is drawn from the same 16–19 sites; if it is, site-grouped CV is *pessimistic* and P-02 could steer away from variants that score. Log the header-derived site mix of the test studies at rerun and choose grouping strictness from it.
- **Missing-slot census at inference.** 3,991/4,407 train studies have all three fat-sat planes (JunhaoLiXD; reconcile with our 24-study 95.8–100% fill — different slot definitions); nothing checks test slot-fill matches train. Assert and log per-slot fill-rate at inference; fall back to the relaxed matching tier.
- **Identical normalisation train/test under the cache.** `read_triplet` clips per-triplet today; a per-series routine in the cache changes both train and val, so P-01's ±0.01 OOF sanity check must be run with the *new* normalisation on both, from one fingerprinted function.
- **Seed-noise baseline.** No card measures run-to-run OOF variance; a 2-seed fold-0 repeat is the first experiment after the cache and calibrates every later threshold ([Picard 2021](https://arxiv.org/abs/2109.08203)).
- **Weak-vs-gold prevalence per label** — now measured (`artifacts/label_audit.md` §2): Synovitis 13% vs 47%, Lateral Meniscus 16% vs 40%, Fracture 7% vs 31%, ACL 21% vs 41%.
- **Label correlation.** Gold φ (n=58, SE of φ ≈ 0.13): Effusion~Synovitis 0.40, Medial OA~Medial Meniscus 0.42, Contusion~Fracture 0.33, Medial~Lateral OA 0.32 (weak: 0.28 / 0.36 / 0.28 / 0.49). (An earlier draft swapped gold/weak for Medial OA~Medial Meniscus.) A per-label attention head (P-09) can lose the shared-vector benefit — report whether P-09 hurts correlated pairs.
- **Memory fallback order** for K=12–16 × 6 slots full FT on a 15 GB T4: checkpoint per block → freeze patch-embed + first 4 blocks → batch 1 + accumulation (added to P-08).
- **DINOv2 pos-embed interpolation and register tokens** (Section 2.4).
- **Decode failures** — `read_triplet` returns zeros and the study still trains; the cache pass must record failures per series and exclude or mask them (P-19).
- **Submission-time robustness** — time budget per study derived from N_test measured at start, deterministic cuDNN flags, assert `submission.csv` row order equals `test.csv`, probe the image root by glob, no placeholder file (loud failure), refuse constant predictions (Submission #1 scored 0.500).
- **Efficiency Prize may score CPU-only runtime** (RSNA-style precedent); if so ViT-S × 6 slots × K on CPU is the binding constraint, not GPU — check the `ryanholbrook` notebook body.

---

## 3. Recommended default training recipe

*"v02" is what `src/kaggle_pipeline.py` implements as of 2026-08-28; "v01" is the previous (smoke-only) configuration, kept so the history is visible.*

| Parameter | v01 value | Our current value (v02) | Recommended | Why | Source |
|---|---|---|---|---|---|
| Data source | DICOM decoded every epoch | same | uint8 cache `[study, 6, K, P, P]` + presence mask, built once | 100× faster epochs; enables everything below | [hida1211 log], [PLATFORM.md], all RSNA winners |
| Slices per slot (K) | 6 | 6 | 12–16 cached, random strided subset in training, all at inference | Winners used 24–32+; focal findings sit on few slices; linear cost | [Nischaydnk], [darraghdog/RSNA22], [brendanartley] |
| Slice band | fixed | fixed | per-plane: sag 0.08–0.92, ax 0.10–0.90, cor 0.20–0.80 | menisci/MCL near sagittal ends | romanrozen (not re-read), [pilkwang] |
| Triplet gap | index 2 | index 2 | keep index 2; ablate g = clip(round(3 mm / spacing), 1, 2) later (with 3–4 mm slices it always gives g=1) | fixed index = variable physical gap, but untested | **[our hypothesis]**, spacing varies (JunhaoLiXD) |
| Crop | none (full FOV → 224) | same | 130 mm physical centre crop, then resize | 99.57% coverage; normalises mm/px ~3× | [FINDINGS.md], [pilkwang] |
| Resolution | 224 | 224 | 224; cache at 224 (K=12 ≈ 15.9 GB, two shards under the ~20 GB output cap); a 336 cache (≈36 GB) needs sharding across kernels and is a later member test | DINOv2-ViT nearly flat on resolution; +0.017 LB for 2.25× FLOPs | [2510.07191], sadamtorres (not re-read) |
| Normalisation | per-triplet 1/99 clip | same | per-series 0.5–99.5 or 1/99 over the sampled stack → uint8; Rescale + MONOCHROME1 first; ImageNet mean/std on GPU | consensus; consistent scale across slices in one pool | [pilkwang], [AnyMC3D Table 3], [Innolitics], [MONAI #282] |
| Laterality | none | none | side from tag else sign(median centre_x) with 20 mm dead zone; flip W on COR/AX, reverse SAG order; never flip when unresolved | tag missing on 50.7%; 5/12 labels side-specific | [FINDINGS.md], [pilkwang] |
| Slot set | 6 recovered slots | same | keep all 6; log leave-one-slot-out OOF; cut SAG_T1 then COR_T1 only if no label drops > 0.02 OOF | contusion needs T1/T2; OA may lean on T1 | [CoPAS], [Maarek 2025] |
| Backbone | DINOv2 ViT-S/14, full FT | same | same | 0.85 vs 0.78 ResNet vs 0.69 3D; S≈B | [MST], [2509.06467], [2402.07595] |
| Backbone LR | 5e-5 uniform | **2e-5 top block** (effective batch 4 studies) | keep 2e-5 at batch 4–8; scale with effective batch; sweep {1e-5, 2e-5, 5e-5} on fold 0 only after the seed-noise baseline | every medical sweep lands ≤ 2e-5 at batch 8–64 | [2501.14685], [2312.02366], [MST] |
| Layer-wise LR decay | none | **0.75 per block** (patch/pos embed one more step) | same; test as its own arm vs uniform 2e-5 | protects SSL low-level features; free | [MAE Table 9], [2606.25989] |
| Head LR | 1e-3 | 1e-3 | 1e-3 | unanimous | all recipe sources |
| Head warmup | none | none | optional: freeze backbone ~0.5 epoch, then unfreeze — separate arm, blog-grade evidence | random head would slam backbone at step 1 | [HF pneumonia blog] (blog-grade) |
| Weight decay | 0.01 | **0.02, none on bias/LayerNorm** | 0.02–0.05 | MAE 0.05; pilkwang 0.02 | [MAE], [pilkwang] |
| Schedule | 10% linear warmup → cosine | same | same (already present; 5–10% warmup both fine) | universal | [MAE], [How to train your ViT], [Nischaydnk] |
| Grad clip | global norm 1.0 after `scaler.unscale_` | same | same (already present) | universal stabiliser | [MAE], [Nischaydnk] |
| Effective batch | 1 study × accum 4 | same | 8 studies (batch 8 from cache, or 2 × 4) | matches public recipes; pairs with lower LR; ViT has no BN | [pilkwang], [hida1211], [MST] |
| Epochs | 4 | 4 | 8 (after cache); one fixed count from fold-mean OOF curve | winners 10–40; label-noise memorisation phase | [TheoViel], [Label Wave] |
| Checkpoint | best.pt by per-fold gold AUC, raw weights | **`best.pt` = EMA 0.998 weights after the last completed epoch (fixed-epoch selection; per-epoch score logged only)** | same; optionally average last 2–3 epochs | per-fold gold selection is a coin flip; EMA robust to noise | [2411.18704], [Model soups], [hida1211] |
| Drop-path / dropout | head dropout 0.1 | same | drop_path 0.0, head dropout 0.1 (already present) | small ViT, short schedule | [How to train your ViT] |
| Targets | **rank-percentile blend of 3 sources** (confident negatives at 0.28–0.39) | **mean of source probabilities** (gold 0.8948) | same; rank space only for scoring/ensembling | BCE fits the value, not the order | our `build_targets.py`, critic item 1 |
| Loss | weighted soft-target BCE, no pos_weight | same | same; optional SCE arm later | BCE within 0.01 of ASL/RAL; noise-robust losses fail on imbalanced clinical noise | [RAL], [LNMBench] |
| Multi-task balancing | mean over 12 | same | same; static per-label weight only if a label stalls | GradNorm/PCGrad no gain | [Xin et al.] |
| Gold weight | 8.0 | 8.0 | 8.0 until a `gold_weight ∈ {1, 3, 8}` arm is run; gold is validation, no gold fine-tuning stage | community 3–8× (🔁 untested here; 1.3% of studies, ~10% of loss mass) | [JunhaoLiXD], hida1211 |
| Augmentation | Gaussian noise | same | rot ±8°, scale 1.00–1.08 (zoom-in only), shift 5%, gain ±10%, gamma 0.8–1.25, random cached slice offset; same affine per triplet; **no flips** | every strong result used rigid+intensity; flips break laterality | [pilkwang], [Azcona], [Guo et al.] |
| Head | concat 6 slots + mask → linear | same | 12 learned queries, masked attention over slot (or all slice) tokens + slot-type embedding → per-label logit; slot dropout 0.15 | plane/sequence specialisation; missing slots renormalised | [MRNet], [CoPAS], [AnyMC3D], mattiaangeli (not re-read) |
| Folds | 5, grouped by report text | same | 5, grouped by (report hash ∪ site proxy = language + Manufacturer + Model) | grouped vs random gap up to +0.136 | [EXPERIMENTS.md], [Guo et al.] |
| Validation logging | fold macro AUC | **per-label AUC + OOF csv written every epoch (`_ep{e}_oof.csv`, plus `_oof.csv` for the checkpointed epoch)** | add per-label pred std, gold macro with 2,000-rep percentile bootstrap, grouped vs random OOF, site-level OOF, 2-seed noise baseline | decision protocol | [Andre et al.], experiments.md |
| TTA | none | none | none by default; 2–3 slice-window offsets **[our hypothesis]** as a later test | only label-safe TTA; geometric TTA hurts; no source shows slice-window TTA helps | [2604.09697], [2311.06118] |
| Ensemble | rank-mean folds | rank-mean folds | **5-fold single family (DINOv2-S) is the default** until P-10/P-13 report; then rank-mean × {DINOv2-S, one CNN} equal weights unless grouped OOF says otherwise; never tuned on LB | winners' practice | [TheoViel], [Nischaydnk], [SeuTao] |
| AMP | AMP | AMP (fp16 + GradScaler) | fp16 autocast + GradScaler; channels_last only for CNN member | Turing has no bf16 tensor cores | [PyTorch memory_format] |
| Attention kernel | HF default | HF default | `attn_implementation="sdpa"`; memory fallback: checkpoint per block → freeze patch-embed + 4 blocks → batch 1 | cheap T4 speedup; K=12–16 memory budget | critic item 27 / 19 |
| Inference | decode per fold; image root assumed | **`MODE=infer` from mounted checkpoints** (the submitted configuration); shallow image-root glob; no placeholder — loud failure; model inputs (slices/slot, gap, img_size) read from the checkpoint config; `MODE=auto` → infer only if every fold has a mounted best.pt | decode once per study → all checkpoints; probe image root by glob; **no placeholder; refuse to submit constant columns or < 90% imaged-with-slot coverage** (Submission #1 = 0.500); fingerprint each checkpoint; log s/100 studies; slot-fill census | decode dominates; silent failures | pilkwang/tonylica (not re-read), [PLATFORM.md] |
| Decoders | pydicom | pydicom | pydicom + mounted pylibjpeg trio wheels (`pip install --no-index`); count decode failures | compressed syntaxes may exist; internet off | [pydicom plugin table], [kaggle_requirements.txt] |
| Threads | num_workers 2 | 2 | ORDER 32 / HDR 16 / PIX 12 threads in cache kernel; num_workers 0 with mmap `.npy` shards | mount latency-bound | [pilkwang] source |

---

## 4. Proposal cards (ranked by expected value / cost)

### P-00 Targets in probability space, not rank percentiles (DONE 2026-08-28, v02)
Hypothesis: `rank(pct=True)` percentiles with average-rank ties teach BCE that a confident negative is 0.3 on rare labels; the mean of source probabilities keeps the LLMs' 0/1 semantics on the same scale as gold.
Origin: critic item 1, confirmed against `src/build_targets.py`
Evidence: confident negatives at 0.28–0.39 on MCL / Lateral OA / Baker's (MCL p25 = 0.312); no study on any label (0%) < 0.1 before the gold override; after the fix 2–72% per label are < 0.1 (Synovitis 2%, Baker's 26%, MCL 72%) and the teacher gold macro-AUC is 0.8948 vs 0.8934 (Δ 0.0014, noise — the scale is the fix); stevenleehans card warning.
Measure: per-label target quantiles (logged by `build_targets.py`); downstream, per-label OOF-vs-teacher and pred std when the first real run exists.
Noise floor: n/a for the scale fix; 0.01 OOF for any downstream effect.
Cost: done (~40 lines; `rank_blend` kept as diagnostic).
If it works: rare labels no longer pulled toward 0.3; gold and weak rows on one scale.
If it fails: n/a — correctness fix; a per-source monotone calibration to gold is the fallback if source scales disagree.
Depends on: nothing. Note: `confidence_weights()` was always computed from raw probabilities and is unaffected.

### P-01 Preprocessing cache kernel (uint8, ordered, cropped, laterality-normalised) — ⏳ running (shards A/B on Kaggle; built at K=16 @ 224, 2 shards ≈ 21 GB total / ~10.6 GB each)
Hypothesis: A one-off cache of all 4,407 studies removes the DICOM decode bottleneck, cutting epoch time from hours to minutes and unblocking every other experiment.
Origin: public consensus / competition write-up
Evidence: 11.1 GB cache (K=9@224) built in ~16 min then 103 s/epoch (verified [hida1211 log](https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4)); K=12@224 uint8 = 4,407×6×12×224² ≈ 15.9 GB → two shards under the ~20 GB per-kernel output cap, ~1–1.5 h on 4 CPU procs ([PLATFORM.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); all RSNA winners trained from pre-extracted arrays ([Nischaydnk](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution), [gunesevitan](https://github.com/gunesevitan/rsna-2023-abdominal-trauma-detection)); our own ~4.5 s/study figure is an **extrapolation from 8 passes** (experiments.md) — benchmark first, as handoff.md's step 1 says.
Measure: decode benchmark first (s/study on the mount), then wall-clock per epoch and per fold; OOF macro-AUC vs teacher must be unchanged vs the decode-every-epoch run **with the same (new) normalisation on both train and val** — otherwise the tolerance test is confounded (sanity, not a gain). Loader default: `np.load(mmap_mode='r')` on `.npy` shards, `num_workers=0`.
Noise floor: n/a (engineering); sanity check tolerance ±0.01 OOF.
Cost: 1 session (CPU kernel) + ~300 lines (header pass, decode pass, manifest, loader); cache versioned by (P, K, crop, laterality rule).
If it works: epochs 4 → 8+, slices 6 → 12–16, second backbone, 336 px member and TTA all become affordable; inference reuses the same function.
If it fails: (output cap or time) fall back to K=6 at 224 (7.4 GB, one shard); caching at 336 (K=12 ≈ 36 GB) is not attempted without sharding across kernels; nothing else in this plan proceeds without it.
Depends on: nothing; blocks P-02 through P-15.

### P-02 Site-grouped folds and grouped-vs-random OOF logging
Hypothesis: Report-text-only grouping lets the model memorise scanner/site signatures, inflating OOF and biasing every comparison toward higher-capacity variants.
Origin: competition write-up / peer-reviewed
Evidence: grouped 0.7049 vs random 0.8412 on a ResNet34 run of this corpus, gap growing every epoch (verified [EXPERIMENTS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); language = vendor (Cyrillic 100% Philips, Dutch/German/Greek 100% Siemens); knee external-validation drop 0.05±0.03 ([Eur Radiol 2025](https://link.springer.com/article/10.1007/s00330-025-12052-8)) and 0.05–0.10 across knee literature ([CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)); secondary: MRI cross-manufacturer drop −0.097 is a prostate figure ([Guo et al.](https://arxiv.org/pdf/2409.04368)). Caveat: if the hidden test shares the train sites, grouped CV is pessimistic — log the test site mix at rerun (Section 2.9).
Measure: grouped OOF vs random-split OOF macro-AUC over 4,407, plus per-site OOF; gap size is the deliverable.
Noise floor: ~0.01 OOF; a gap > 0.02 confirms inflation.
Cost: 0.2 session, ~40 lines in `build_targets.py` (site_key from header manifest; language guess fallback).
If it works: honest model comparison; predicts private-LB behaviour; also tells us whether laterality/aug changes reduce per-site spread.
If it fails: (headers anonymised, no Manufacturer) fall back to language-only grouping and note it.
Depends on: header scan with Manufacturer/ManufacturerModelName/MagneticFieldStrength (part of P-01).

### P-03 LR / layer-wise decay / EMA (implemented as v02 on 2026-08-28; not yet measured)
Hypothesis: The v01 uniform 5e-5 backbone LR, no EMA and best-epoch-on-gold selection over-train and mis-select; a standard SSL-ViT fine-tuning recipe is at least as good and more stable. Warmup (10%) + cosine, grad-clip 1.0 and wd were already present in v01 and are **not** part of this change. v02 = lr_backbone 2e-5, LLRD 0.75, wd 0.02 (none on bias/LayerNorm or on cls/mask/pos-embedding tokens), EMA 0.998 validated and saved; **checkpoint = EMA weights after the last completed epoch (fixed-epoch selection, no best-epoch pick on gold; per-epoch score logged only); OOF written every epoch**.
Origin: peer-reviewed / public consensus
Evidence: medical ViT sweeps land at 1e-5 ([2501.14685](https://arxiv.org/html/2501.14685v1)), ≤ 2e-5 ([2312.02366](https://arxiv.org/html/2312.02366v3)), 1e-6 ([MST](https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/)); lr 1e-3 collapse ([dinov2 #276](https://github.com/facebookresearch/dinov2/issues/276)); MAE recipe LLRD 0.75, wd 0.05, 5% warmup ([MAE](https://arxiv.org/html/2111.06377)); uniform LR = catastrophic forgetting ([2606.25989](https://arxiv.org/html/2606.25989)); EMA robust to label noise ([2411.18704](https://arxiv.org/html/2411.18704)); public recipes 8e-6/1e-3 with 6 blocks unfrozen ([pilkwang](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)).
Measure: first a 2-seed repeat of one config on fold 0 (seed-noise baseline); then OOF macro-AUC vs teacher over 4,407 (fold 0 first, then all); per-epoch OOF curve smoothness; gold reported not gated. Split by evidence strength into arms: (A) LR 2e-5 + EMA (strong / medium evidence) vs (B) A + LLRD 0.75 (no medical ablation) vs optional (C) B + 0.5-epoch head-freeze (blog-grade). State LR relative to effective batch: 2e-5 at 4 studies × ~36 slot-slices; batch 8 from the cache is the precondition for quoting the sweeps' 1e-5–2e-5 directly.
Noise floor: 0.01 OOF (to be replaced by the measured seed noise); single components inside that floor are reported as inconclusive, not adopted or rejected.
Cost: done for the code (~60 lines: param groups with layer decay, EMA class); 0.3 session for the arms once the cache exists.
If it works: more stable folds, the fixed-epoch selection already shipped is confirmed safe, foundation for 8-epoch runs.
If it fails: (OOF drops > seed noise) revert LLRD first, then LR to 5e-5, keeping EMA; sweep {1e-5, 2e-5, 5e-5} on fold 0.
Depends on: P-01 for the 8-epoch part; the LR/EMA change can ship before the cache.

### P-04 Multi-epoch schedule with fixed epoch count from the fold-mean OOF curve
Hypothesis: 8 epochs (vs 4) with cosine, EMA and a single fixed epoch count improves OOF and gold without memorising teacher errors. (Fixed-epoch *selection* is already v02 behaviour — `best.pt` is the last-epoch EMA weights; this card is about the *count*, read off the per-epoch OOF csvs v02 writes.)
Origin: competition write-up / peer-reviewed
Evidence: winners 10–40 epochs ([TheoViel](https://github.com/TheoViel/kaggle_rsna_abdominal_trauma), [Nischaydnk]); public knee recipes 10–12 epochs ([pilkwang], [hida1211]); label-noise memorisation phase ([Label Wave](https://arxiv.org/html/2502.07551v1)); MST early stop on val AUC.
Measure: per-epoch OOF-vs-teacher macro and per-label AUC over 4,407, fold-averaged; gold curve logged alongside to detect divergence.
Noise floor: 0.01 OOF; gold 0.05.
Cost: 1 session (5 folds × 8 epochs from cache ≈ 5 × 15–25 min for full-unfrozen ViT-S — to be measured).
If it works: single fixed epoch count for all folds; enables per-label failure analysis on a converged model.
If it fails: (OOF peaks at 3–4) keep 4 epochs and spend budget on slices/members.
Depends on: P-01, P-03.

### P-05 Laterality normalisation from DICOM geometry
Hypothesis: Canonicalising knee side (mirror COR/AX, reverse SAG order) halves the chirality the model must learn and lifts the four medial/lateral labels and MCL.
Origin: public consensus / competition write-up
Evidence: Laterality missing on 12,367/24,371 series; centre-x rule 97.4% (98.5% with 20 mm dead zone); corner rule 58.8% (verified [FINDINGS.md](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)); plane-specific operation and 20 mm dead zone ([pilkwang] source, pulled); three independent implementations ([JunhaoLiXD](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection), hida1211, sadamtorres); Fritz's persistent 0.1 lateral gap ([Fritz](https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/)).
Measure: OOF AUC over 4,407 for Medial/Lateral Meniscus, Medial/Lateral OA, MCL before/after; tag-vs-geometry agreement on tagged series (assert ≥ 0.95); fraction unresolved; **count of tag-vs-geometry conflicts and the rule applied** (trust geometry, or leave that series un-normalised); in-plane orientation (`ImageOrientationPatient`, `PatientPosition`) canonicalised for all planes, not only laterality.
Noise floor: 0.01–0.02 per label OOF; gold cannot resolve it.
Cost: 0.3 session, ~80 lines in header scan + cache builder; visual spot-check of ~20 knees.
If it works: correctness fix baked into the cache. (H-flip remains a dead end in both variants — see Section 5; it does not become legal.)
If it fails: (no OOF change) keep it anyway as anatomically correct; investigate GE sub-sample.
Depends on: P-01 (cache version string encodes the rule).

### P-06 Per-label failure analysis and monitoring
Hypothesis: One label at chance costs ~0.029 macro-AUC; systematic per-label diagnostics (OOF AUC, pred std, teacher AUC, FN/FP composition on gold) will find the cheapest lever per weak label.
Origin: our hypothesis / peer-reviewed for the difficulty ordering
Evidence: literature difficulty order ACL < medial meniscus < OA/effusion < lateral meniscus ≈ BME < synovitis matches our teacher ([Eur Radiol 2024 review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12021734/), [Astuto], [Fritz], [synovitis paper](https://pubmed.ncbi.nlm.nih.gov/37951778/)); label collapse to base rate observed publicly ([JunhaoLiXD V02]); Synovitis unaddressed in ~84–88% of reports ([stevenleehans]).
Measure: per-epoch per-label OOF AUC and prediction std over 4,407 (v02 already writes per-label AUC and an OOF csv every epoch, `_ep{e}_oof.csv`, plus `_oof.csv` for the checkpointed epoch); on gold, FN-vs-FP counts for Synovitis/Contusion/Fracture/Lateral OA; macro over the 6 weakest labels reported next to the 12; a `gold_weight ∈ {1, 3, 8}` arm (gold is 1.3% of studies but ~10% of loss mass at 8×; it also biases the fold-mean OOF-vs-teacher curve).
Noise floor: per-label OOF 0.015–0.02; gold per-label 0.09 (diagnostic only).
Cost: 0.1 session, ~50 lines of logging; CSV-only for the gold audit.
If it works: directs effort (label synonym fix vs severity cut-point vs slot/resolution) per label.
If it fails: n/a — diagnostics.
Depends on: nothing (gold audit runs now on `artifacts/targets.csv`).

### P-07 Synovitis back-fill and label coverage audit (no new sources) — audit DONE 2026-08-28; back-fill 🔁 INCONCLUSIVE
Hypothesis: Replacing "unaddressed" Synovitis targets with the Effusion soft label, and auditing per-language coverage, improves the weakest teacher label more than any model change.
Origin: competition write-up / peer-reviewed
Evidence: the public card's Synovitis gold AUC 0.678→0.790 after Effusion back-fill ([stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)) **was from a 0.678 baseline; on our blend it does not reproduce**: 0.788 → 0.729, paired-bootstrap Δ −0.059 [95% CI −0.164, +0.042] (`artifacts/label_audit.md` §5; 41 of 58 gold are unaddressed for Synovitis, 14 of those positive). Audit also found hans_v4 and sol56 deciding identically at the 0.5 cut (agreement 99.45% over 4,407 studies; error-φ 1.000 on gold for every label; raw values differ — consistent with, not proof of, the v4 blend including sol56) and Spanish as the worst-covered language (Fracture UNK 80%). Version AUCs match our own measurements (experiments.md); reports help only where the finding is represented in text ([2510.24385](https://arxiv.org/abs/2510.24385)); more LLM sources add ~nothing (n_eff 2.2, [2605.29800](https://arxiv.org/html/2605.29800); our 0.002 spread); radiologist ceiling ~0.77 on synovitis.
Measure: per the repo's own rule (experiments.md: gold deltas < 0.05 are inconclusive) — coverage per language and student OOF-vs-teacher for Synovitis over 4,349 lead; gold AUC (HM SE 0.061 at this prevalence) is reported, not gated. **Weights should be recomputed from the back-filled values**: without recomputation the back-filled rows average weight 0.69; recomputing from Effusion gives 0.81 (measured).
Noise floor: gold per-label 0.09 (HM SE 0.061 for Synovitis) — the measured −0.059 is inside it; decide on coverage + student OOF.
Cost: 0.1 session, ~30 lines in `build_targets.py` (audit script exists: `src/label_audit.py`).
If it works: template for Contusion synonym handling (cannot be audited from the tables — no source exposes matched terms; open for P-16).
If it fails: logged 🔁 INCONCLUSIVE on gold; keep the un-back-filled blend unless student OOF says otherwise; do not try a Fracture←Contusion back-fill without validation.
Depends on: nothing.

### P-08 Slice sampling: 6 → 12–16 cached slices per slot with per-plane bands and random offsets
Hypothesis: More slices per slot with attention pooling raises recall for focal findings (meniscus, Baker's, MCL) at linear cost.
Origin: competition write-up / peer-reviewed
Evidence: winners 24–32+ slices per volume ([Nischaydnk], [darraghdog/RSNA22], [brendanartley]); more views help on anisotropic volumes ([TomoGraphView](https://arxiv.org/html/2511.09605)); query attention benefits from more tokens ([AnyMC3D](https://arxiv.org/pdf/2512.12887)); 2024 2nd's extra view moved CV 0.01–0.02.
Measure: OOF macro and per-label AUC over 4,407 at K=6 vs K=12/16 on the same cache and folds; seconds/study on T4.
Noise floor: 0.01 OOF.
Cost: 0.5 session (timing + one paired run); cache size drives the choice: uint8 bytes = 4,407 × 6 × K × P², so K=12@224 ≈ 15.9 GB (two shards under the ~20 GB per-kernel output cap), K=16@224 ≈ 21 GB, K=12@256 ≈ 21 GB, K=10@288 ≈ 22 GB, K=12@336 ≈ 36 GB (needs sharding across kernels). Memory fallback on the 15 GB T4 for K=12–16 × 6 slots full FT: (a) `torch.utils.checkpoint` per block, (b) freeze patch-embed + first 4 blocks, (c) batch 1 + accumulation; SDPA attention first. Triplet gap in mm is a separate ablation, not part of this change.
If it works: default K raised; training subset sampling doubles as augmentation.
If it fails: keep K=6 and spend the budget on resolution or a second member.
Depends on: P-01.

### P-09 Per-label masked attention head over slots (and optionally all slice tokens)
Hypothesis: 12 learned queries attending over present slot/slice tokens beat concat → linear by letting each finding read its own sequences and by handling missing slots correctly.
Origin: peer-reviewed / competition write-up
Evidence: per-finding plane dependence ([MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699), [CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)); negative transfer with shared vectors ([Azcona](https://arxiv.org/abs/2010.01947)); query attention > alternatives ([AnyMC3D]); public leaders use all-series cross-attention with slot-type embedding (mattiaangeli, **not re-read**); pilkwang `SlotHead`.
Measure: OOF macro and per-label AUC over 4,407 vs the concat head, same cache/folds.
Noise floor: 0.01 OOF.
Cost: 0.3 session, ~80 lines; config flag.
If it works: default head; enables leave-one-slot-out ablations via attention masks.
If it fails: keep concat; record in experiments.md.
Depends on: P-01 (fast epochs to run the A/B).

### P-10 Second architecture family for the rank blend (timm ConvNeXt/EffNetV2 first, RadImageNet R50 behind a flag)
Hypothesis: A CNN member adds error diversity that an all-DINOv2 blend lacks.
Origin: competition write-up / peer-reviewed
Evidence: every RSNA winner blended families ([TheoViel], [Nischaydnk], [darraghdog/RSNA22]); ConvNeXt-B > ViT-B/16 on CXR ([2510.07191](https://arxiv.org/abs/2510.07191)); DINOv2 lost to ImageNet CNNs on clinical brain MRI ([2402.07595](https://arxiv.org/abs/2402.07595)); RadImageNet > ImageNet R50 on knee tasks ([RadImageNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/)) but the weights have **no stated weight licence** (code MIT, article CC BY 4.0) — treat as restrictive until radimagenet.com T&C are read, and correct brainstorm.md/CLAUDE.md (which say CC-BY-NC-SA-4.0) once settled; public +0.003 from a RadImageNet head blend is inside noise (prvsiyan, **not re-read**).
Measure: member's own OOF over 4,407 (include only if within ~0.02 of DINOv2-S); blend OOF and gold; LB.
Noise floor: 0.01 OOF, 0.005 LB — blend gain will likely be unmeasurable; justify by CV robustness.
Cost: 1 session for the CNN member (LR 1e-4 backbone, BN frozen or GroupNorm, channels_last); RadImageNet frozen-feature head ≈ 0.3 session.
If it works: 2-family rank-mean submission.
If it fails: single-family 5-fold submission; RadImageNet dropped at final if the licence clause is unclear.
Depends on: P-01, P-04.

### P-11 Resolution: 224 vs 336 after the 130 mm crop
Hypothesis: Higher effective mm/px helps small focal labels (Lateral Meniscus, PF OA, Fracture, Contusion).
Origin: peer-reviewed / competition write-up
Evidence: 224→512 median +1.05 pp on CXR, concentrated in focal findings, but DINOv2-ViT nearly flat ([2510.07191](https://arxiv.org/abs/2510.07191)); 224→336 +0.017 LB at 2.25× FLOPs, preds correlate 0.90 (sadamtorres, **not re-read**); yu4u crop+384 gain confounded with cropping ([yu4u deck](https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui)).
Measure: per-label OOF over 4,407 for the four focal labels at 224 vs 336, K fixed; adopt only if their mean moves > 0.02.
Noise floor: 0.015–0.02 per label OOF.
Cost: 1 session (2.25× tokens); a ≥ 336 cache (K=12 ≈ 36 GB) exceeds the ~20 GB per-kernel output cap and must be sharded across kernels/datasets, or built for a subset of slots; DINOv2 pos-embeds are interpolated at either size (`interpolate_pos_encoding`).
If it works: 336 member (init from the 224 checkpoint) added to the blend.
If it fails: stay at 224; spend budget on slices/folds.
Depends on: P-01 (cache at ≥ 336), P-08 (K decided first).

### P-12 Slice-window TTA (label-safe only) **[our hypothesis]**
Hypothesis: Averaging logits over 2–3 slice-index offsets per slot reduces variance at small inference cost; geometric TTA is skipped.
Origin: competition write-up / public consensus
Evidence: no source shows slice-window TTA helps — winners used (geometric) TTA ([brendanartley], [SeuTao]); geometric TTA degraded 11/12 medical pairs ([2604.09697](https://arxiv.org/html/2604.09697v1)); flips hurt knee OA even after mirroring ([2311.06118](https://arxiv.org/html/2311.06118)); H-flip unsafe for 5 labels.
Measure: OOF macro-AUC over 4,407 with vs without TTA; inference seconds/100 studies.
Noise floor: 0.005 OOF — expected gain is below it; justified by cost only.
Cost: 0.1 session, ~30 lines.
If it works: default for the accuracy track; off for the efficiency variant.
If it fails: drop.
Depends on: P-01 (decode-once inference).

### P-13 3 vs 5 folds under a fixed session budget **[our hypothesis]**
Hypothesis: 3 folds + a second backbone beats 5 folds of DINOv2-S for the same T4 hours. Until this reports, the default stays 5 folds of a single family.
Origin: our hypothesis
Evidence: no source quantifies seed vs fold vs architecture gains; winners ran 4–5 folds × several models ([darraghdog/RSNA22] 5 × 3 seeds; [Nischaydnk] 4 folds); fold soups match best single at zero cost ([Model soups](https://arxiv.org/html/2203.05482)).
Measure: OOF (on the common held-out set) and gold for {5-fold ViT-S} vs {3-fold ViT-S + 3-fold CNN}; LB as secondary.
Noise floor: 0.01 OOF; likely inconclusive.
Cost: 1–2 sessions; needs P-10.
If it works: budget rule for the final submission.
If it fails: default to 5 folds of the best single family.
Depends on: P-01, P-04, P-10.

### P-14 DINOv2-S vs DINOv2-B (LoRA if memory-bound)
Hypothesis: ViT-B adds little over ViT-S at 4k studies; test only as a late diversity member.
Origin: peer-reviewed
Evidence: ViT-T/S/B 0.952/0.954/0.954 on MRNet ACL ([SB-SSL](https://arxiv.org/abs/2208.13923)); size non-monotonic ([2509.06467](https://arxiv.org/html/2509.06467v3)); S ≈ B ([2402.07595], [2606.25989]); +0.008–0.029 on CT with LoRA ([AnyMC3D]); LoRA parity on clean CXR ([2312.02366]) but full FT > PEFT at scale ([2510.07191]).
Measure: OOF macro over 4,407 vs ViT-S at matched wall-clock.
Noise floor: 0.01 OOF.
Cost: 1 session (~3× ViT-S; gradient checkpointing or LoRA r=8).
If it works: extra blend member.
If it fails: confirmed dead end; frees budget.
Depends on: P-01, P-04, after P-10.

### P-15 DINOv3-S/16 as a diversity member (not a replacement)
Hypothesis: DINOv3 at 224 is not better than DINOv2 but is decorrelated enough to help a rank blend.
Origin: peer-reviewed / competition write-up
Evidence: DINOv2 vs DINOv3 differences 0.002–0.008 both directions ([AnyMC3D Table 8]); DINOv3 wins only at 512 px ([2510.07191]); public leaders blend 5 DINOv3-S folds (mattiaangeli, **not re-read**); custom licence, gated download, must mirror LICENSE.md ([dinov3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)).
Measure: member OOF and blend OOF over 4,407; gold.
Noise floor: 0.01 OOF.
Cost: 1 session + Kaggle Model mirror with licence file.
If it works: third member.
If it fails: dropped; CNN member preferred.
Depends on: P-01, P-10.

### P-16 Re-labelling with an open-weights LLM inside Kaggle (graded, synonym-aware, native language)
Hypothesis: A 14–32B open model prompted for 4-level grades plus explicit negation and synonym lists produces better soft targets than binary frontier-model tables, especially for Synovitis/Contusion/Fracture.
Origin: peer-reviewed / competition write-up
Evidence: open models ≈ GPT-4o on extraction ([Radiology 2025](https://pubs.rsna.org/doi/10.1148/radiol.240895), [JIIM 2025](https://pubmed.ncbi.nlm.nih.gov/40325326/)); uncertainty-graded smoothing +4.5 mean AUC on CXR ([Rep-GLS](https://arxiv.org/html/2508.02495)); image-aligned labels beat report labels ([VisualCheXbert](https://arxiv.org/abs/2102.11467)); native > translate for multilingual models ([2602.21374](https://arxiv.org/html/2602.21374)); Qwen3-14B-AWQ reached 0.881 gold (sadamtorres, **not re-read**); 70B does not fit 2×T4.
Measure: gold macro-AUC of new targets (paired bootstrap vs current blend), OOF-vs-new-teacher of the student, per-language coverage, sample of flipped reports read by hand.
Noise floor: gold 0.05 macro (CLAUDE.md rule; 3,000-rep study-level bootstrap of the teacher's gold macro-AUC: SD 0.017) / per-label HM SE ≈ 0.09 — most improvements will be unresolvable on gold; decide on coverage + flipped-report audit + student OOF.
Cost: 1–2 sessions (vLLM on 2×T4, weights as Kaggle Model) + ~150 lines; rules risk resolved by staying on-platform.
If it works: replaces blend targets via a grade → soft-value rubric in `build_targets.py`.
If it fails: keep the 3-source blend + P-07 back-fill.
Depends on: P-06 gold audit (FN vs FP per weak label) to choose synonym vs cut-point focus.

### P-17 Noise-robust loss / soft-target variants (SCE, self-distillation, AUC-margin stage 2)
Hypothesis: Symmetric CE or round-2 targets (0.5 LLM + 0.5 rank-normalised round-1 OOF) reduce teacher-error memorisation; an AUC-margin stage-2 matches the metric.
Origin: peer-reviewed / competition write-up
Evidence: only SCE/CDR beat CE under real clinical noise ([LNMBench](https://arxiv.org/html/2512.09315v1)); pseudo-label blends load-bearing for winners ([brendanartley], yu4u); AUCM 2-epoch stage won CheXpert ([Yuan et al.](https://arxiv.org/html/2012.03173)); BCE within 0.01 of alternatives ([RAL]).
Measure: OOF-vs-teacher over 4,407 (and vs round-1 OOF for distillation); gold sanity only.
Noise floor: 0.01 OOF; self-distillation judged on gold is unresolvable.
Cost: 0.3 session each; AUCM needs LibAUC offline or ~40-line reimplementation **and hard labels** — thresholding our soft targets reintroduces the cut-point problem this document warns about, so AUCM is lowest priority or dropped.
If it works: a ≥ 0.01 OOF gain adopted.
If it fails: BCE stays; bottom of backlog.
Depends on: P-01, P-04, P-06.

### P-18 Efficiency-track variant and decode-once inference — 🔧 infer mode + loud-failure submission shipped (kernel v4); variant untested
Hypothesis: A single DINOv2-S at 224 with decode-once inference and no TTA is competitive on the Efficiency LB at no accuracy cost we can measure. Shipped: `MODE=infer` from mounted checkpoints, no placeholder file (a crash → missing submission → visible error), coverage gate on test studies imaged **and** with ≥ 1 slot, constant-column gate, shallow image-root glob, model inputs read from the checkpoint config. Measured (kernel v3, smoke): 153 s per 100 test studies per fold at 2 slices/slot.
Origin: competition write-up
Evidence: Efficiency LB top 0.948 vs accuracy 0.952 and non-monotone ranking (verified CSV, [ryanholbrook LB](https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb)); ViT-S inference ≪ decode ([timm CSV](https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv)); fold soup as 5× inference saver is **[our hypothesis]** — [Model soups] averaged same-data fine-tunes, souping across folds (different data, same init) is untested here. Formula unread; if the Efficiency Prize scores CPU-only runtime (RSNA-style precedent), CPU inference of ViT-S × 6 slots × K is the binding constraint.
Measure: wall-clock per 100 test studies; public LB of the lean variant.
Noise floor: 0.005 LB.
Cost: 0.2 session (EFFICIENCY_MODE flag, timing logs); must read the formula in a browser first.
If it works: second prize track with the same model.
If it fails: (formula penalises differently) adjust after reading it.
Depends on: P-01.

### P-19 Decoder wheels and TransferSyntax census
Hypothesis: Compressed DICOMs may exist in train or test; without mounted pylibjpeg/GDCM wheels they vanish silently at rerun.
Origin: public consensus
Evidence: pydicom needs plugins for JPEG Lossless/2000 ([plugin table](https://pydicom.github.io/pydicom/stable/guides/plugin_table.html)); Kaggle requirements lack them ([kaggle_requirements.txt](https://raw.githubusercontent.com/Kaggle/docker-python/main/kaggle_requirements.txt)); one notebook reports compressed series (jirkaborovec, pulled) vs a census claiming 100% uncompressed ([FINDINGS.md]).
Measure: TransferSyntaxUID value counts over 24,371 series; decode-failure count at test (assert < 0.1%); decode failures per series recorded in the cache manifest and excluded/masked rather than trained on as zeros (`read_triplet` returns zeros today).
Noise floor: n/a.
Cost: 0.1 session, ~20 lines + a wheels dataset.
If it works: silent-failure trap closed; add to traps.md.
If it fails: n/a — insurance.
Depends on: P-01 header pass.

### P-20 Leave-one-slot-out ablation and T1 slot retirement
Hypothesis: SAG_T1 (50% fill) and COR_T1 (62.5%) can be retired if no label drops > 0.02 OOF, freeing budget for fat-sat slices; Contusion/Fracture/OA are the labels at risk.
Origin: peer-reviewed / our hypothesis
Evidence: contusion 0.82 → 0.70 without T1W/T2W ([CoPAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/)); fat-sat fluid carries oedema ([Maarek 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12362699/)); 2024 2nd dropped a view moving CV < 0.02 ([brendanartley]); plane preference is model-specific ([MRNet], [Azcona], [ELNet]).
Measure: per-label OOF over 4,407 with each slot's presence mask forced to 0 at inference (no retraining).
Noise floor: 0.015–0.02 per label OOF.
Cost: 0.1 session (inference-time flag).
If it works: slot budget reallocated to K in fat-sat slots.
If it fails: keep all 6 slots.
Depends on: P-01 only. With the current concat head, zeroing a slot's vector and setting its mask to 0 is equally trivial, so this can run on the first real model; P-09 is not required.

---

## 5. Things to avoid (dead ends)

| Dead end | Reason | Source |
|---|---|---|
| Native 3D CNN / nnU-Net / segmentation-first | 0.69 vs 0.85 (p=0.001) on MRNet; CoPAS 0.81 mean with clean labels; multi-A100 budgets | [MST], [CoPAS], [MIC-DKFZ 2025] |
| Frozen DINOv2 + head as the final model | 0.79 vs 0.85 knee; gold 0.771 / LB 0.776 vs 0.866 fine-tuned | [MST], sadamtorres (not re-read) |
| Backbone LR 5e-5+ uniform, no layer decay (v01) | every medical recipe ≤ 2e-5 at batch 8–64; lr 1e-3 collapse | [2501.14685], [dinov2 #276] |
| Rank-percentile blend as the *training target* | average-rank ties put confident negatives at 0.28–0.39; BCE fits the value | our `build_targets.py`, critic item 1 (fixed v02) |
| Assuming the hidden test image root / `.dcm` extension; silently filling 0.5 (including a 0.5 placeholder file) | Submission #1 scored exactly 0.500 — constant predictions at rerun | our submission log; fix: probe root by glob, no placeholder, refuse to submit constants |
| ViT-B/L before a second family | S ≈ B in three sources; 3× cost | [SB-SSL], [2509.06467], [2402.07595] |
| DINOv2 → DINOv3 swap at 224 as an accuracy gain | differences 0.002–0.008 both ways; wins only at 512 | [AnyMC3D], [2510.07191] |
| BiomedCLIP / MedSAM / RAD-DINO / OrthoFoundation | far below general ViTs; CXR-only; weights not public | [2501.14685], [2505.10823], [2601.18250] |
| EfficientNet-B0 mean-pool | 0.664 vs 0.809 public on this task | [JunhaoLiXD] |
| Horizontal flip (with or without label swap), vertical flip | without swap: re-introduces the chirality laterality normalisation removed; with swap: MCL has no lateral counterpart; V-flip off-distribution — both variants dead-ended until a specific ablation is designed | [pilkwang], traps.md |
| Geometric TTA | degraded 11/12 medical pairs; flips hurt knee OA | [2604.09697], [2311.06118] |
| Laterality tag alone / default L / IPP-corner rule | tag missing 50.7%; corner 58.8% | [FINDINGS.md] |
| Pixel-flipping sagittal slots | stacks are order-reversed, not mirrored | [pilkwang] |
| Filename / InstanceNumber ordering | ρ ≈ −0.01; interleaved series | CLAUDE.md, [JunhaoLiXD] |
| Crops ≥ 160 mm; zoom-out padding; resolution > 512 | skipped on 60% of series; fabricated tissue; ViT losses −6.6/−7.9 pp | [pilkwang], [2510.07191] |
| N4 / Nyul / VOI-LUT before normalisation | segmentation/radiomics evidence only; infeasible at 24k series | [2307.03827], [2406.01736] |
| Decoding DICOM in the DataLoader each epoch; float32 caches; .npz + mmap; in-RAM cache with forked workers; GPU decode | 100× slower; 29.6 GB; mmap ignored for .npz; worker copies of a 16 GB array; ~1–2.5× for ≤ 512 px | [hida1211], [NumPy #5976], [nvImageCodec] |
| pip install at scoring time | internet off; compressed series vanish | [pydicom plugin table] |
| bf16 on T4; channels_last for ViT; torch.compile by default | no bf16 tensor cores; cuDNN-only benefit; compile > gain | [PyTorch memory_format] |
| More LLM label sources, Dawid–Skene, Snorkel, CARE, learned source weights | n_eff ≈ 2.2; DS < MV; our 0.002 spread | [2605.29800], [BoxWRENCH] |
| Co-teaching / DivideMix / DISC / focal / ASL / GradNorm / PCGrad | minority collapse; ≤ 0.01 over BCE; no gain over weighted sum | [LNMBench], [RAL], [Xin et al.] |
| pos_weight, Platt scaling, thresholds, label smoothing on soft targets | no effect on rank order | metric arithmetic |
| Full FT or best-epoch selection on 58 gold (~12 per fold) | SE 0.09; coin flip; our NaN-fold bug | experiments.md, [Andre et al.] |
| Calibrated priors with a 50% floor for zero-support states | OOF collapsed to base rate | [JunhaoLiXD V02] |
| Translate-then-extract with a multilingual model; sub-3B extractors; 70B on 2×T4 | precision loss; F1 0.74; ~40 GB weights | [2602.21374], VRAM arithmetic |
| Hosted LLM APIs for report text | plausible Rule 4.b violation; open-weights parity | homeshwarnelakurthi README, [Radiology 2025] |
| Forking the public 0.95 ensemble; tuning on public LB | author-labelled overfit; 0.001–0.003 movements | mattiaangeli (not re-read), CLAUDE.md |
| Random or report-only K-fold as the comparison metric | grouped vs random gap +0.136 | [EXPERIMENTS.md] |
| Extending epochs before the cache exists | 6–8 h/fold (extrapolated from 8 passes) does not fit 9 h | experiments.md |
| Adopting the Synovitis←Effusion back-fill on the public card's +0.11 alone | on our blend gold 0.788 → 0.729, CI includes 0 — 🔁 INCONCLUSIVE, needs the student-OOF arm | `artifacts/label_audit.md` |
| Any A/B before a 2-seed noise baseline | the 0.01 OOF floor is asserted, not measured | [Picard 2021] |
| In-domain SSL continued pretraining as a first priority | SB-SSL in-domain ViT still below ImageNet AlexNet; costs a full session | [SB-SSL] |
| Treating MRNet-val or gold deltas < 0.05, LB < 0.005 as real | 120-exam / 58-study noise floors | [Andre et al.], CLAUDE.md |
| P100 accelerator | no Pascal kernels | CLAUDE.md |

---

## 6. Source list

**Knee-MRI literature (peer-reviewed)**
- Medical Slice Transformer — https://pmc.ncbi.nlm.nih.gov/articles/PMC12227771/
- MRNet (Bien et al. 2018) — https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699
- Azcona et al. 2020 — https://arxiv.org/abs/2010.01947
- ELNet (MIDL 2020) — https://arxiv.org/abs/2005.02706
- CoPAS (Nat Commun 2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/
- Astuto et al. (Radiology: AI 2021) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8166108/
- Fritz et al. (Skeletal Radiol 2020) — https://pmc.ncbi.nlm.nih.gov/articles/PMC7299917/
- SB-SSL (Atito et al. 2022) — https://arxiv.org/abs/2208.13923
- Namiri et al. 2020 — https://arxiv.org/abs/2003.09089
- Eur Radiol 2025 23-condition slice transformer — https://link.springer.com/article/10.1007/s00330-025-12052-8
- Synovitis DL (Academic Radiology 2024) — https://pubmed.ncbi.nlm.nih.gov/37951778/
- Systematic review (Eur Radiol 2024) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12021734/
- MPFuseNet — https://arxiv.org/abs/2108.08136
- KneeXNet (Frontiers 2025) — https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1590962/full
- Stanford MRNet leaderboard — https://stanfordmlgroup.github.io/competitions/mrnet/
- Maarek et al. 2025 BML detection — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12362699/
- OrthoFoundation — https://arxiv.org/abs/2601.18250

**Prior RSNA competition write-ups**
- Nischaydnk RSNA 2023 1st — https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- TheoViel RSNA 2023 2nd — https://github.com/TheoViel/kaggle_rsna_abdominal_trauma
- gunesevitan RSNA 2023 14th — https://github.com/gunesevitan/rsna-2023-abdominal-trauma-detection
- yu4u RSNA 2023 10th deck — https://speakerdeck.com/yu4u/rsna-2023-abdominal-trauma-detection-fan-sheng-hui
- darraghdog RSNA 2022 3rd — https://github.com/darraghdog/RSNA22
- pascal-pfeiffer RSNA 2022 5th — https://github.com/pascal-pfeiffer/kaggle-rsna-2022-5th-place
- brendanartley RSNA 2024 2nd — https://github.com/brendanartley/RSNA-2024-Competition
- SeuTao RSNA 2019 1st — https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
- darraghdog RSNA 2019 2nd — https://github.com/darraghdog/rsna
- k951286 RSNA 2025 deck — https://speakerdeck.com/k951286/kaggle-rsna-intracranial-aneurysm-detectionkonpe-fan-sheng-hui
- MIC-DKFZ RSNA 2025 — https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution
- AnyMC3D / Revisiting 2D FMs for 3D classification — https://arxiv.org/pdf/2512.12887

**Weak labels, label noise, evaluation**
- VisualCheXbert — https://arxiv.org/abs/2102.11467 ; Jain et al. — https://arxiv.org/abs/2104.00793
- Radiology 2025 LLM labelling — https://pubs.rsna.org/doi/10.1148/radiol.240895
- JIIM 2025 LLM labels for CXR — https://pubmed.ncbi.nlm.nih.gov/40325326/
- Weak-label crossover study — https://arxiv.org/html/2605.24771
- BoxWRENCH — https://arxiv.org/html/2501.07727 ; WRENCH — https://arxiv.org/abs/2109.11377
- LP-FT (Kumar et al.) — https://arxiv.org/abs/2202.10054
- LLM judge panels — https://arxiv.org/html/2605.29800 ; CARE — https://arxiv.org/abs/2603.00039
- Rep-GLS — https://arxiv.org/html/2508.02495 ; Pham et al. — https://arxiv.org/abs/1911.06475
- LNMBench — https://arxiv.org/html/2512.09315v1
- Multilingual extraction — https://arxiv.org/html/2403.10258 ; https://arxiv.org/html/2602.21374 ; MOSAIC — https://arxiv.org/abs/2509.04471
- AUC CIs — https://arxiv.org/html/2601.17103 ; Feng et al. — https://pubmed.ncbi.nlm.nih.gov/26323286/
- Spinal MRI LLM labels — https://arxiv.org/html/2410.17235
- When are radiology reports useful — https://arxiv.org/abs/2510.24385
- Label Wave — https://arxiv.org/html/2502.07551v1

**Backbones, fine-tuning, licensing**
- MedMNIST FM benchmark — https://arxiv.org/html/2501.14685v1
- DINOv2 radiology study — https://arxiv.org/html/2312.02366v3
- RadImageNet paper — https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/ ; GitHub — https://github.com/BMEII-AI/RadImageNet ; site — https://www.radimagenet.com/
- DINOv3 resolution scaling — https://arxiv.org/abs/2510.07191
- Does DINOv3 set a new medical standard — https://arxiv.org/html/2509.06467v3
- ImageNet vs DINOv2 on clinical MRI — https://arxiv.org/abs/2402.07595
- LLRD marine paper — https://arxiv.org/html/2606.25989
- HF DINOv2-S pneumonia blog — https://huggingface.co/blog/t22000t/dino-model-rsna
- Embeddings to accuracy — https://arxiv.org/html/2505.10823 ; RAD-DINO card — https://huggingface.co/microsoft/rad-dino
- Data scarcity FMs — https://arxiv.org/abs/2408.08058
- DINOv3 licence — https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md ; HF card — https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- MAE — https://arxiv.org/html/2111.06377 ; How to train your ViT — https://arxiv.org/html/2106.10270 ; BEiT (layer decay) — https://arxiv.org/abs/2106.08254
- Vision Transformers Need Registers (Darcet et al.) — https://arxiv.org/abs/2309.16588 ; HF facebook/dinov2-with-registers-small
- Picard 2021, seed variance — https://arxiv.org/abs/2109.08203
- dinov2 issue #276 — https://github.com/facebookresearch/dinov2/issues/276
- EMA study — https://arxiv.org/html/2411.18704 ; ISIC EMA note — https://zenn.dev/morim34/articles/bfa2465defee06
- Model soups — https://arxiv.org/html/2203.05482
- RAL / CXR-LT — https://arxiv.org/abs/2308.05542
- Deep AUC maximisation — https://arxiv.org/html/2012.03173 ; LibAUC script — https://raw.githubusercontent.com/Optimization-AI/LibAUC/main/examples/scripts/07_optimizing_multi_label_auroc_loss_with_densenet121_on_chexpert.py
- MTO does not help — https://arxiv.org/abs/2209.11379 ; DeepChest — https://arxiv.org/abs/2505.23595
- TTA is not better — https://arxiv.org/html/2604.09697v1 ; knee OA augmentation — https://arxiv.org/html/2311.06118

**MRI preprocessing**
- Laterality metadata errors — https://pmc.ncbi.nlm.nih.gov/articles/PMC6646614/
- FLAIR normalisation — https://arxiv.org/abs/2307.03827 ; breast radiomics normalisation — https://arxiv.org/abs/2406.01736
- Guo et al. domain shift — https://arxiv.org/pdf/2409.04368
- Innolitics DICOM best practices — https://innolitics.com/articles/medical-imaging-best-practices/
- MONAI Deploy MONOCHROME1 issue — https://github.com/Project-MONAI/monai-deploy-app-sdk/issues/282
- TomoGraphView — https://arxiv.org/html/2511.09605 ; AlignShift — https://arxiv.org/abs/2005.01969 ; 2.5D review — https://arxiv.org/abs/2010.06163

**This competition (Kaggle; bodies partly re-read via CLI pull, otherwise existence-verified only)**
- pilkwang/rsna-knee-baseline-v1 (pulled) — https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1
- hida1211/rsna-knee-public-4-fold-dinov2-v4 (log pulled) — https://www.kaggle.com/code/hida1211/rsna-knee-public-4-fold-dinov2-v4
- jirkaborovec/kneeabdet-12-label-mri-screening-with-ptl-timm (pulled)
- sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee (not re-read) — https://www.kaggle.com/code/sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee
- mattiaangeli/bend-the-knee-to-dinov3-ensembled (not re-read) — https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-dinov3-ensembled
- tonylica/rsna-knee-dino-radimagenet-rank-ensemble (not re-read) — https://www.kaggle.com/code/tonylica/rsna-knee-dino-radimagenet-rank-ensemble
- prvsiyan/rsna-knee-read-the-report-then-the-knee (not re-read) — https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee
- ryanholbrook/rsna-knee-abnormalities-efficiency-lb (CSV pulled) — https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb
- stevenleehans/rsna-knee-llm-report-labels — https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels
- yunusgmsoy Gemini labels — https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels ; laymond Qwen3-8B labels — https://www.kaggle.com/datasets/laymond/rsna-knee-abnormality-qwen3-8b-weak-labels
- homeshwarnelakurthi repo (README, FINDINGS.md, EXPERIMENTS.md, PLATFORM.md verified) — https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection
- JunhaoLiXD repo (README verified) — https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection

**Pipeline engineering**
- pydicom plugin table — https://pydicom.github.io/pydicom/stable/guides/plugin_table.html
- Kaggle requirements — https://raw.githubusercontent.com/Kaggle/docker-python/main/kaggle_requirements.txt
- dicomsdl — https://github.com/tsangel/dicomsdl/blob/master/README.md ; timing notebook — https://raw.githubusercontent.com/tsangel/dicomsdl/master/tutorials/timeit_test.ipynb
- nvImageCodec DICOM sample — https://docs.nvidia.com/cuda/nvimagecodec/samples/DICOM-pydicom.html ; NVIDIA blog — https://developer.nvidia.com/blog/advancing-medical-image-decoding-with-gpu-accelerated-nvimagecodec
- timm inference benchmark — https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/benchmark-infer-amp-nhwc-pt240-cu124-rtx3090.csv
- NumPy mmap/.npz issue — https://github.com/numpy/numpy/issues/5976 ; mmap vs zarr — https://pythonspeed.com/articles/mmap-vs-zarr-hdf5/
- PyTorch channels_last tutorial — https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
- torch.compile / Triton capability — https://discuss.pytorch.org/t/torch-compile-triton-cuda-capability/182068 ; TorchBench — https://arxiv.org/abs/2304.14226

**Local**
- `CLAUDE.md`, `docs/experiments.md`, `docs/brainstorm.md`, `docs/traps.md`, `docs/handoff.md`, `src/build_targets.py`, `src/kaggle_pipeline.py`, `src/label_audit.py`, `artifacts/label_audit.md` (C:\Users\Tian\Desktop\RSNA_Knee)

**Not accessible in this research (must be read in a browser):** competition Rules text (Rule 4.b, winner licence, 9 h limit), Efficiency Prize evaluation page (and whether it scores CPU-only), discussion threads, pilkwang notebook's gold-protocol cells, radimagenet.com Terms & Conditions, Kaggle docs on the per-kernel output / dataset size caps.
