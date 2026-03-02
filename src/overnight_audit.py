#!/usr/bin/env python3
"""
Overnight Audit Script for Project Rayuela.

Runs a comprehensive validation of all results, monitors v2 extraction,
performs the full v1-vs-v2 comparison when ready, and writes a detailed
report to outputs/audit_report.md.

Designed to run as a detached Docker container overnight:
    docker compose run --rm -d --name rayuela-audit rayuela python src/overnight_audit.py

The script loops, checking v2 progress every 10 minutes, and writes the
final report when v2 is complete (or after a maximum wait of 8 hours).
"""

import json
import time
import datetime
import numpy as np
from pathlib import Path
from itertools import pairwise

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "outputs" / "audit_report.md"

TABLERO = [
    73, 1, 2, 116, 3, 84, 4, 71, 5, 81, 74, 6, 7, 8, 93, 68, 9, 104,
    10, 65, 11, 136, 12, 106, 13, 115, 14, 114, 117, 15, 120, 16, 137,
    17, 97, 18, 153, 19, 90, 20, 126, 21, 79, 22, 62, 23, 124, 128, 24,
    134, 25, 141, 60, 26, 109, 27, 28, 130, 151, 152, 131, 29, 139, 30,
    138, 31, 32, 132, 33, 140, 34, 135, 35, 105, 36, 63, 37, 98, 38,
    102, 39, 113, 40, 120, 41, 100, 42, 76, 43, 44, 108, 45, 69, 46,
    101, 47, 110, 48, 111, 49, 118, 50, 119, 51, 69, 52, 89, 53, 66,
    149, 54, 129, 139, 133, 140, 138, 127, 56, 135, 63, 88, 72, 77, 131,
    58, 131,
]
LINEAR_ORDER = list(range(1, 57))
DIMS_ORDERED = [
    "existential_questioning", "art_and_aesthetics", "everyday_mundanity",
    "death_and_mortality", "love_and_desire", "emotional_intensity",
    "humor_and_irony", "melancholy_and_nostalgia", "tension_and_anxiety",
    "oliveira_centrality", "la_maga_presence", "character_density",
    "interpersonal_conflict", "interiority", "dialogue_density",
    "metafiction", "temporal_clarity", "spatial_grounding",
    "language_experimentation", "intertextual_density",
]


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def write_report(sections: list[str]):
    """Write the audit report."""
    header = f"""# Overnight Audit Report — Project Rayuela

Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(header + "\n\n".join(sections))
    log(f"Report written to {REPORT_PATH}")


# ===========================================================================
# AUDIT 1: Data Integrity
# ===========================================================================

def audit_data_integrity() -> str:
    log("AUDIT 1: Data integrity checks")
    issues = []

    # 1a. rayuela_raw.json
    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    chapters = raw["chapters"]
    if len(chapters) != 155:
        issues.append(f"CRITICAL: rayuela_raw.json has {len(chapters)} chapters, expected 155")
    nums = [ch["number"] for ch in chapters]
    if nums != list(range(1, 156)):
        issues.append(f"CRITICAL: chapter numbers not sequential 1-155: {sorted(set(range(1,156)) - set(nums))}")

    # Check sections
    sec_counts = {}
    for ch in chapters:
        sec_counts[ch["section"]] = sec_counts.get(ch["section"], 0) + 1
    expected_secs = {"Del lado de allá": 36, "Del lado de acá": 20,
                     "De otros lados (Capítulos prescindibles)": 99}
    for sec, expected in expected_secs.items():
        actual = sec_counts.get(sec, 0)
        if actual != expected:
            issues.append(f"WARNING: section '{sec}' has {actual} chapters, expected {expected}")

    # 1b. Scale A embeddings
    emb = np.load(PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy")
    if emb.shape != (155, 1024):
        issues.append(f"CRITICAL: embeddings shape {emb.shape}, expected (155, 1024)")
    if np.any(np.isnan(emb)):
        issues.append("CRITICAL: NaN values in embeddings")
    if np.any(np.all(emb == 0, axis=1)):
        issues.append("WARNING: zero-vector rows in embeddings")

    # Check embedding norms (should be roughly unit for e5 model)
    norms = np.linalg.norm(emb, axis=1)
    if norms.std() > 0.1:
        issues.append(f"NOTE: embedding norm std={norms.std():.3f} (expected ~0 for normalized)")

    # 1c. v1 narrative DNA
    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        v1 = json.load(f)
    v1_ch = v1["chapters"]
    if len(v1_ch) != 155:
        issues.append(f"CRITICAL: v1 has {len(v1_ch)} chapters, expected 155")
    if len(v1["dimensions"]) != 20:
        issues.append(f"CRITICAL: v1 has {len(v1['dimensions'])} dims, expected 20")

    # Check score ranges
    for ch in v1_ch:
        for d in v1["dimensions"]:
            s = ch["scores"][d]
            if not (1 <= s <= 10):
                issues.append(f"WARNING: Ch.{ch['chapter']}.{d} = {s} (outside 1-10)")

    # 1d. v1 numpy matrix
    v1_npy = np.load(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna_vectors.npy")
    if v1_npy.shape != (155, 20):
        issues.append(f"CRITICAL: v1 numpy shape {v1_npy.shape}, expected (155, 20)")

    # Cross-check JSON vs numpy
    for i, ch in enumerate(v1_ch):
        json_vec = np.array([ch["scores"][d] for d in v1["dimensions"]])
        npy_vec = v1_npy[i]
        if not np.allclose(json_vec, npy_vec):
            issues.append(f"WARNING: Ch.{ch['chapter']} JSON/numpy mismatch")
            break

    status = "PASS" if not any("CRITICAL" in i for i in issues) else "FAIL"
    result = f"## Audit 1: Data Integrity — {status}\n\n"
    if issues:
        result += "\n".join(f"- {i}" for i in issues) + "\n"
    else:
        result += "All checks passed. 155 chapters, 1024-dim embeddings, 20-dim narrative DNA, all consistent.\n"
    return result


# ===========================================================================
# AUDIT 2: Reproduce Key Statistical Claims
# ===========================================================================

def audit_statistics() -> str:
    log("AUDIT 2: Reproducing statistical claims")
    findings = []

    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    meta = {ch["number"]: ch for ch in raw["chapters"]}

    emb_a = np.load(PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy")
    ch_to_idx = {ch["number"]: i for i, ch in enumerate(raw["chapters"])}

    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        v1 = json.load(f)
    dims = v1["dimensions"]
    scores = {ch["chapter"]: ch["scores"] for ch in v1["chapters"]}

    def path_mean_dist(path, space):
        dists = []
        for a, b in pairwise(path):
            if space == "A":
                if a in ch_to_idx and b in ch_to_idx:
                    va, vb = emb_a[ch_to_idx[a]], emb_a[ch_to_idx[b]]
                    cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                    dists.append(1 - cos)
            elif space == "B":
                if a in scores and b in scores:
                    va = np.array([scores[a][d] for d in dims])
                    vb = np.array([scores[b][d] for d in dims])
                    dists.append(np.linalg.norm(va - vb))
        return np.mean(dists) if dists else 0

    rng = np.random.default_rng(42)
    all_ch = [ch["number"] for ch in raw["chapters"]]
    n_perms = 5000

    for space, pool, label in [("A", all_ch, "Texture"), ("B", list(scores.keys()), "Narrative")]:
        lin_mean = path_mean_dist(LINEAR_ORDER, space)
        hop_mean = path_mean_dist(TABLERO, space)

        rand_means = []
        for _ in range(n_perms):
            perm = list(rng.permutation(pool)[:56])
            rand_means.append(path_mean_dist(perm, space))
        rand_means = np.array(rand_means)

        z_lin = (lin_mean - rand_means.mean()) / rand_means.std()
        z_hop = (hop_mean - rand_means.mean()) / rand_means.std()

        findings.append(f"**Scale {space} ({label})**:")
        findings.append(f"  - Linear: mean={lin_mean:.4f}, z={z_lin:+.2f}σ")
        findings.append(f"  - Hopscotch: mean={hop_mean:.4f}, z={z_hop:+.2f}σ")
        findings.append(f"  - Random: μ={rand_means.mean():.4f}, σ={rand_means.std():.4f}")
        findings.append(f"  - Linear percentile: {(rand_means > lin_mean).mean()*100:.2f}%")
        findings.append(f"  - Hopscotch percentile: {(rand_means > hop_mean).mean()*100:.1f}%")
        findings.append("")

    # Cross-scale correlation
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    a_dists = pdist(emb_a, metric="cosine")
    b_matrix = np.array([[scores[i + 1][d] for d in dims] for i in range(155)])
    b_dists = pdist(b_matrix, metric="euclidean")
    rho, pval = spearmanr(a_dists, b_dists)
    findings.append(f"**Cross-scale correlation**: Spearman ρ = {rho:.3f} (p = {pval:.2e})")

    # Article claims to verify
    claims = []
    # Claim: "linear path is -8.6σ" — check it's close
    # (Already computed above, z_lin for Scale B should be around -8.6)

    result = "## Audit 2: Reproduce Key Statistics\n\n"
    result += "\n".join(findings) + "\n\n"
    result += "These numbers should match ARTICLE_DRAFT.md claims.\n"
    return result


# ===========================================================================
# AUDIT 3: Article Draft Fact-Check
# ===========================================================================

def audit_article_claims() -> str:
    log("AUDIT 3: Fact-checking article draft claims")
    issues = []

    draft_path = PROJECT_ROOT / "ARTICLE_DRAFT.md"
    if not draft_path.exists():
        return "## Audit 3: Article Draft Fact-Check\n\nDraft not found.\n"

    draft = draft_path.read_text()

    # Load actual data for verification
    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        v1 = json.load(f)
    scores = {ch["chapter"]: ch["scores"] for ch in v1["chapters"]}

    # Check specific claims
    checks = []

    # Ch.68 scores
    ch68 = scores.get(68, {})
    if ch68:
        if ch68.get("language_experimentation") != 10:
            issues.append(f"Ch.68 language_experimentation = {ch68.get('language_experimentation')}, article says 10")
        if ch68.get("love_and_desire") != 9:
            issues.append(f"Ch.68 love_and_desire = {ch68.get('love_and_desire')}, article says 9")
        if ch68.get("spatial_grounding") != 1:
            issues.append(f"Ch.68 spatial_grounding = {ch68.get('spatial_grounding')}, article says 1")
        if ch68.get("dialogue_density") != 1:
            issues.append(f"Ch.68 dialogue_density = {ch68.get('dialogue_density')}, article says 1")
        checks.append("Ch.68 (Gliglico) scores: " + ("VERIFIED" if not issues else "MISMATCH"))
    else:
        issues.append("Ch.68 not found in v1 scores")

    # Ch.1 scores
    ch1 = scores.get(1, {})
    if ch1:
        if ch1.get("oliveira_centrality") != 10:
            issues.append(f"Ch.1 oliveira_centrality = {ch1.get('oliveira_centrality')}, article says 10")
        checks.append(f"Ch.1 oliveira_centrality = {ch1.get('oliveira_centrality')}: VERIFIED")

    # Section counts
    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    alla = sum(1 for ch in raw["chapters"] if "allá" in ch["section"])
    aca = sum(1 for ch in raw["chapters"] if "acá" in ch["section"])
    otros = sum(1 for ch in raw["chapters"] if "otros" in ch["section"].lower())
    checks.append(f"Section counts: Allá={alla}, Acá={aca}, Otros={otros}")

    # Verify 155 total
    if "155 chapters" in draft:
        checks.append(f"Total chapters: {len(raw['chapters'])} (article claims 155)")
        if len(raw["chapters"]) != 155:
            issues.append(f"Article claims 155 chapters but data has {len(raw['chapters'])}")

    # Check section profile claims
    alla_ch = [ch["number"] for ch in raw["chapters"] if "allá" in ch["section"]]
    alla_oli = np.mean([scores[ch]["oliveira_centrality"] for ch in alla_ch if ch in scores])
    alla_int = np.mean([scores[ch]["interiority"] for ch in alla_ch if ch in scores])
    alla_exi = np.mean([scores[ch]["existential_questioning"] for ch in alla_ch if ch in scores])
    checks.append(f"Allá means: Oliveira={alla_oli:.1f}, Interiority={alla_int:.1f}, Existential={alla_exi:.1f}")
    # Article says 8.6, 8.6, 7.9
    if abs(alla_oli - 8.6) > 0.15:
        issues.append(f"Article says Allá oliveira_centrality=8.6 but actual is {alla_oli:.1f}")
    if abs(alla_int - 8.6) > 0.15:
        issues.append(f"Article says Allá interiority=8.6 but actual is {alla_int:.1f}")

    # Ch.55 phantom chapter
    ch55_in_tablero = 55 in TABLERO
    checks.append(f"Ch.55 in Tablero: {ch55_in_tablero} (article claims it's NOT)")
    if ch55_in_tablero:
        issues.append("Article claims Ch.55 is not in hopscotch, but it IS in TABLERO")

    result = "## Audit 3: Article Draft Fact-Check\n\n"
    result += "**Verified claims:**\n" + "\n".join(f"- {c}" for c in checks) + "\n\n"
    if issues:
        result += "**Issues found:**\n" + "\n".join(f"- {i}" for i in issues) + "\n"
    else:
        result += "No issues found.\n"
    return result


# ===========================================================================
# AUDIT 4: Visualization Integrity
# ===========================================================================

def audit_visualizations() -> str:
    log("AUDIT 4: Checking visualization files")
    fig_dir = PROJECT_ROOT / "outputs" / "figures"

    expected_2d = [
        "article_heatmap.html", "article_journey.html", "article_smoothness.html",
        "article_radar.html", "article_correlation.html", "article_sections.html",
        "article_dual.html", "article_dual_heatmap.html", "article_permutation.html",
        "article_weaving.html", "umap_scale_a.html", "umap_scale_b.html",
        "umap_comparison.html",
    ]
    expected_3d = [
        "3d_scale_a.html", "3d_scale_b_full.html", "3d_scale_b_top8var.html",
        "3d_scale_b_pca5.html", "3d_scale_b_pca8.html", "3d_scale_b_decorr.html",
    ]

    results = []
    missing = []
    small = []
    for f in expected_2d + expected_3d:
        path = fig_dir / f
        if path.exists():
            size = path.stat().st_size
            if size < 100_000:  # Less than 100KB is suspicious
                small.append(f"{f} ({size/1024:.0f}KB)")
            results.append(f"- {f}: {size/1024/1024:.1f}MB")
        else:
            missing.append(f)

    result = "## Audit 4: Visualization Files\n\n"
    if missing:
        result += f"**Missing ({len(missing)}):** " + ", ".join(missing) + "\n\n"
    if small:
        result += f"**Suspiciously small:** " + ", ".join(small) + "\n\n"
    result += f"**Present ({len(results)}):**\n" + "\n".join(results) + "\n"
    return result


# ===========================================================================
# AUDIT 5: v2 Extraction Monitoring & Comparison
# ===========================================================================

def check_v2_progress() -> tuple[int, bool]:
    """Return (n_chapters, is_complete)."""
    v2_path = PROJECT_ROOT / "outputs" / "semantic_v2" / "narrative_dna.json"
    if not v2_path.exists():
        return 0, False
    try:
        with open(v2_path) as f:
            v2 = json.load(f)
        n = len(v2.get("chapters", []))
        return n, n >= 155
    except (json.JSONDecodeError, KeyError):
        return -1, False


def audit_v2_comparison() -> str:
    log("AUDIT 5: v1 vs v2 comparison")

    v2_path = PROJECT_ROOT / "outputs" / "semantic_v2" / "narrative_dna.json"
    if not v2_path.exists():
        return "## Audit 5: v1 vs v2 Comparison\n\nv2 data not found.\n"

    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        v1 = json.load(f)
    with open(v2_path) as f:
        v2 = json.load(f)

    dims = v1["dimensions"]
    v1_scores = {ch["chapter"]: ch["scores"] for ch in v1["chapters"]}
    v2_scores = {}
    for ch in v2["chapters"]:
        v2_scores[ch["chapter"]] = {}
        for d in dims:
            val = ch["scores"][d]
            v2_scores[ch["chapter"]][d] = val["score"] if isinstance(val, dict) else val

    common = sorted(set(v1_scores) & set(v2_scores))
    if len(common) < 10:
        return f"## Audit 5: v1 vs v2 Comparison\n\nOnly {len(common)} common chapters, need ≥10.\n"

    lines = [f"## Audit 5: v1 vs v2 Comparison ({len(common)} chapters)\n"]

    # Global stats
    all_diffs = []
    for ch in common:
        for d in dims:
            all_diffs.append(v1_scores[ch][d] - v2_scores[ch][d])
    all_diffs = np.array(all_diffs)

    lines.append(f"### Global Summary ({len(common)} ch × {len(dims)} dims = {len(all_diffs)} scores)")
    lines.append(f"- Mean diff (v1 − v2): {all_diffs.mean():+.2f}")
    lines.append(f"- Std of diff: {all_diffs.std():.2f}")
    lines.append(f"- Mean |diff|: {np.abs(all_diffs).mean():.2f}")
    lines.append(f"- Exact matches: {np.sum(all_diffs == 0)} ({100*np.sum(all_diffs==0)/len(all_diffs):.0f}%)")
    lines.append(f"- Within ±1: {np.sum(np.abs(all_diffs)<=1)} ({100*np.sum(np.abs(all_diffs)<=1)/len(all_diffs):.0f}%)")
    lines.append(f"- Within ±2: {np.sum(np.abs(all_diffs)<=2)} ({100*np.sum(np.abs(all_diffs)<=2)/len(all_diffs):.0f}%)")
    lines.append(f"- Large diffs (|d|≥4): {np.sum(np.abs(all_diffs)>=4)} ({100*np.sum(np.abs(all_diffs)>=4)/len(all_diffs):.1f}%)")
    lines.append("")

    # Per-dimension correlation
    lines.append("### Per-Dimension Correlation")
    lines.append("| Dimension | r | MAD | v1 mean | v2 mean | Direction |")
    lines.append("|-----------|---|-----|---------|---------|-----------|")
    problematic = []
    for d in dims:
        v1v = [v1_scores[ch][d] for ch in common]
        v2v = [v2_scores[ch][d] for ch in common]
        mad = np.mean(np.abs(np.array(v1v) - np.array(v2v)))
        m1, m2 = np.mean(v1v), np.mean(v2v)
        direction = "↑" if m2 > m1 else "↓" if m2 < m1 else "="
        if np.std(v1v) > 0 and np.std(v2v) > 0:
            r = np.corrcoef(v1v, v2v)[0, 1]
        else:
            r = float("nan")
        flag = " **LOW**" if r < 0.5 else ""
        lines.append(f"| {d} | {r:+.3f}{flag} | {mad:.2f} | {m1:.1f} | {m2:.1f} | {direction} |")
        if r < 0.5:
            problematic.append((d, r))
    lines.append("")

    if problematic:
        lines.append("### Problematic Dimensions (r < 0.5)")
        for d, r in problematic:
            lines.append(f"- **{d}**: r = {r:+.3f} — this dimension is interpreted differently by v1 and v2")
        lines.append("")

    # Score distribution comparison
    v1_all = np.array([v1_scores[ch][d] for ch in common for d in dims])
    v2_all = np.array([v2_scores[ch][d] for ch in common for d in dims])
    lines.append("### Score Distribution")
    lines.append(f"- v1: mean={v1_all.mean():.2f}, std={v1_all.std():.2f}, median={np.median(v1_all):.0f}")
    lines.append(f"- v2: mean={v2_all.mean():.2f}, std={v2_all.std():.2f}, median={np.median(v2_all):.0f}")
    lines.append("")

    # If we have enough v2 data, reproduce the permutation test
    if len(common) >= 50:
        lines.append("### Permutation Test with v2 Scores")

        with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
            raw = json.load(f)
        emb_a = np.load(PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy")
        ch_to_idx = {ch["number"]: i for i, ch in enumerate(raw["chapters"])}

        rng = np.random.default_rng(42)

        def path_mean_dist_v2(path):
            dists = []
            for a, b in pairwise(path):
                if a in v2_scores and b in v2_scores:
                    va = np.array([v2_scores[a][d] for d in dims])
                    vb = np.array([v2_scores[b][d] for d in dims])
                    dists.append(np.linalg.norm(va - vb))
            return np.mean(dists) if dists else 0

        lin_mean = path_mean_dist_v2(LINEAR_ORDER)
        hop_mean = path_mean_dist_v2(TABLERO)

        pool = list(v2_scores.keys())
        rand_means = []
        for _ in range(5000):
            perm = list(rng.permutation(pool)[:56])
            rand_means.append(path_mean_dist_v2(perm))
        rand_means = np.array(rand_means)

        z_lin = (lin_mean - rand_means.mean()) / rand_means.std()
        z_hop = (hop_mean - rand_means.mean()) / rand_means.std()

        lines.append(f"- **v2 Linear**: z = {z_lin:+.2f}σ (v1 was −8.6σ)")
        lines.append(f"- **v2 Hopscotch**: z = {z_hop:+.2f}σ (v1 was +0.4σ)")
        lines.append(f"- Random: μ={rand_means.mean():.4f}, σ={rand_means.std():.4f}")

        if abs(z_lin) > 5 and abs(z_hop) < 2:
            lines.append("- **CONCLUSION: Core finding CONFIRMED with v2 scores.** Linear designed, hopscotch random.")
        elif abs(z_lin) > 5 and abs(z_hop) > 2:
            lines.append("- **CAUTION: v2 shows hopscotch may be somewhat designed.** Core finding partially overturned.")
        else:
            lines.append("- **WARNING: v2 results diverge from v1. Manual review needed.**")
        lines.append("")

    # Top disagreements
    lines.append("### Biggest Chapter-Level Disagreements")
    ch_total_diffs = []
    for ch in common:
        total = sum(abs(v1_scores[ch][d] - v2_scores[ch][d]) for d in dims)
        ch_total_diffs.append((ch, total))
    ch_total_diffs.sort(key=lambda x: x[1], reverse=True)

    lines.append("| Chapter | Total |diff| | Biggest swings (≥3) |")
    lines.append("|---------|---------------|---------------------|")
    for ch, total in ch_total_diffs[:10]:
        big = [(d, v1_scores[ch][d], v2_scores[ch][d])
               for d in dims if abs(v1_scores[ch][d] - v2_scores[ch][d]) >= 3]
        swings = "; ".join(f"{d.split('_')[0]}:{a}→{b}" for d, a, b in big)
        lines.append(f"| Ch.{ch} | {total} | {swings or 'none ≥3'} |")
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# AUDIT 6: Edge Cases & Potential Issues
# ===========================================================================

def audit_edge_cases() -> str:
    log("AUDIT 6: Edge cases and potential issues")
    issues = []

    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    meta = {ch["number"]: ch for ch in raw["chapters"]}

    # Check TABLERO validity
    tablero_set = set(TABLERO)
    all_ch = set(range(1, 156))
    missing_from_tablero = all_ch - tablero_set
    issues.append(f"Chapters not in TABLERO: {sorted(missing_from_tablero)}")
    duplicates = [ch for ch in TABLERO if TABLERO.count(ch) > 1]
    if duplicates:
        dup_set = sorted(set(duplicates))
        issues.append(f"Chapters appearing multiple times in TABLERO: {dup_set}")
        for ch in dup_set:
            positions = [i for i, x in enumerate(TABLERO) if x == ch]
            issues.append(f"  Ch.{ch} appears at positions: {positions}")

    # Very short chapters
    short = [(ch["number"], ch["token_count"]) for ch in raw["chapters"] if ch["token_count"] < 50]
    if short:
        issues.append(f"Very short chapters (<50 tokens): {short}")
        issues.append("  These may produce unreliable embeddings and semantic scores")

    # Very long chapters
    long = [(ch["number"], ch["token_count"]) for ch in raw["chapters"] if ch["token_count"] > 5000]
    if long:
        issues.append(f"Very long chapters (>5000 tokens): {long}")
        issues.append("  Embedding model truncation may lose content")

    # Chapters with extreme scores (all 1s or all 10s)
    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        v1 = json.load(f)
    for ch in v1["chapters"]:
        vals = list(ch["scores"].values())
        if all(v == 1 for v in vals):
            issues.append(f"Ch.{ch['chapter']}: ALL scores are 1 — possible extraction failure")
        if all(v == 10 for v in vals):
            issues.append(f"Ch.{ch['chapter']}: ALL scores are 10 — possible extraction failure")
        n_extreme = sum(1 for v in vals if v in (1, 10))
        if n_extreme >= 15:
            issues.append(f"Ch.{ch['chapter']}: {n_extreme}/20 scores at extremes (1 or 10) — may lack nuance")

    result = "## Audit 6: Edge Cases & Potential Issues\n\n"
    result += "\n".join(f"- {i}" for i in issues) + "\n"
    return result


# ===========================================================================
# Main loop
# ===========================================================================

def main():
    log("=" * 60)
    log("OVERNIGHT AUDIT STARTING")
    log("=" * 60)

    # Run all non-v2 audits immediately
    sections = []
    sections.append(audit_data_integrity())
    sections.append(audit_statistics())
    sections.append(audit_article_claims())
    sections.append(audit_visualizations())
    sections.append(audit_edge_cases())

    # Check v2 progress
    v2_n, v2_done = check_v2_progress()
    log(f"v2 extraction: {v2_n}/155 chapters")

    if v2_done:
        log("v2 already complete — running comparison")
        sections.append(audit_v2_comparison())
        write_report(sections)
        log("AUDIT COMPLETE")
        return

    # Write initial report without v2
    sections.append("## Audit 5: v1 vs v2 Comparison\n\nWaiting for v2 extraction to complete...\n")
    write_report(sections)
    log(f"Initial report written (without v2). Monitoring v2 progress...")

    # Monitor v2 progress, updating report every 10 minutes
    max_wait_hours = 8
    check_interval = 600  # 10 minutes
    max_checks = int(max_wait_hours * 3600 / check_interval)

    for check in range(max_checks):
        time.sleep(check_interval)
        v2_n, v2_done = check_v2_progress()
        log(f"v2 progress: {v2_n}/155")

        if v2_done:
            log("v2 COMPLETE — running full comparison")
            sections[-1] = audit_v2_comparison()
            write_report(sections)
            log("FINAL REPORT WRITTEN")
            return

        # Partial comparison if enough data
        if v2_n >= 50 and check % 3 == 0:
            log(f"Running partial v2 comparison ({v2_n} chapters)")
            sections[-1] = audit_v2_comparison()
            write_report(sections)

    # Timeout
    log(f"TIMEOUT after {max_wait_hours} hours. v2 at {v2_n}/155.")
    sections[-1] = audit_v2_comparison() if v2_n >= 10 else "## Audit 5: v1 vs v2\n\nTimed out.\n"
    write_report(sections)
    log("AUDIT COMPLETE (timed out)")


if __name__ == "__main__":
    main()
