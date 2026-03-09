#!/usr/bin/env python3
"""
Scale A': Classical stylometric features — content-free by construction.

Unlike Scale A (E5 embeddings, which capture both meaning and form), these
features measure HOW the prose is written, not WHAT it says. Sentence length,
punctuation patterns, function word frequencies, and vocabulary richness are
stylistic choices independent of thematic content.

If the trajectory findings (linear=smooth, hopscotch=random) hold under Scale A',
the argument becomes airtight: the ordering effects are not artifacts of any
single representation.

Feature groups:
  1. Sentence structure (5 features)
  2. Vocabulary richness (3 features)
  3. Function words (4 features)
  4. Punctuation profile (7 features)
  5. Syntactic complexity (2 features)
  6. Code-switching (2 features)
  7. Readability (3 features)

Total: 26 features per chapter.

Usage (inside Docker container):
    python src/stylometrics.py

Input:  data/rayuela_raw.json
Output: outputs/embeddings/chapter_stylometrics.npy      (155 × 26)
        outputs/embeddings/chapter_stylometrics_metadata.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import spacy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings"

# Spanish function word lists — these carry grammar, not content.
# Grouped by part of speech for interpretability.
ARTICLES = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
PREPOSITIONS = {
    "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre",
    "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras",
}
CONJUNCTIONS = {
    "y", "e", "o", "u", "ni", "pero", "sino", "mas", "aunque", "que",
    "porque", "como", "si", "cuando", "donde", "mientras", "pues",
}
PRONOUNS = {
    "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "me", "te", "se", "nos", "os", "le",
    "les", "lo", "la", "los", "las", "mí", "ti", "sí", "conmigo",
    "contigo", "consigo", "esto", "eso", "aquello", "este", "ese",
    "aquel", "esta", "esa", "aquella",
}

# Code-switching detection: common French and English words that are NOT
# also Spanish words. We exclude cognates (e.g., "radio", "hotel") to
# avoid false positives.
FRENCH_MARKERS = {
    "le", "les", "des", "une", "du", "je", "tu", "il", "elle", "nous",
    "vous", "ils", "elles", "est", "sont", "dans", "avec", "pour", "sur",
    "pas", "mais", "qui", "que", "ce", "cette", "mon", "ton", "son",
    "notre", "votre", "leur", "même", "très", "bien", "tout", "tous",
    "rien", "fait", "être", "avoir", "faire", "dire", "comme", "plus",
    "aussi", "encore", "alors", "donc", "où", "quand", "comment",
    "pourquoi", "ici", "là", "oui", "non", "peut", "autre",
    "peu", "trop", "jamais", "toujours", "chez", "entre", "sous",
    "après", "avant", "sans",
}
# Remove words that overlap with Spanish
FRENCH_MARKERS -= {"le", "les", "une", "tu", "nous", "est", "son", "entre",
                    "sur", "que", "como", "plus", "sans", "sous", "pour"}

ENGLISH_MARKERS = {
    "the", "and", "but", "not", "with", "from", "this", "that", "have",
    "has", "had", "was", "were", "been", "being", "are", "will", "would",
    "could", "should", "may", "might", "shall", "can", "must", "do",
    "does", "did", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "there", "here", "then", "than", "very",
    "just", "only", "also", "too", "so", "if", "or", "yet", "because",
    "although", "though", "while", "after", "before", "until", "since",
    "about", "into", "through", "between", "against", "above", "below",
    "each", "every", "all", "both", "few", "more", "most", "other",
    "some", "such", "any", "many", "much", "own", "same", "her", "his",
    "its", "our", "their", "my", "your",
}


# ---------------------------------------------------------------------------
# Feature descriptions (for metadata output)
# ---------------------------------------------------------------------------

FEATURE_SPEC = [
    # Group 1: Sentence structure
    ("sent_len_mean",    "Mean sentence length in words"),
    ("sent_len_median",  "Median sentence length in words"),
    ("sent_len_std",     "Standard deviation of sentence length"),
    ("sent_len_max",     "Maximum sentence length in words"),
    ("sent_len_cv",      "Coefficient of variation of sentence length (std/mean)"),
    # Group 2: Vocabulary richness
    ("mattr",            "Moving-average type-token ratio (window=50 words, length-independent)"),
    ("hapax_ratio",      "Hapax legomena ratio (words appearing once / total unique)"),
    ("vocab_density",    "Vocabulary density (unique words / sentence count)"),
    # Group 3: Function words (per 1000 words)
    ("articles_per_k",       "Article frequency per 1000 words"),
    ("prepositions_per_k",   "Preposition frequency per 1000 words"),
    ("conjunctions_per_k",   "Conjunction frequency per 1000 words"),
    ("pronouns_per_k",       "Pronoun frequency per 1000 words"),
    # Group 4: Punctuation profile (per 1000 words)
    ("semicolons_per_k",     "Semicolons per 1000 words"),
    ("colons_per_k",         "Colons per 1000 words"),
    ("em_dashes_per_k",      "Em-dashes per 1000 words"),
    ("ellipses_per_k",       "Ellipses per 1000 words"),
    ("exclamations_per_k",   "Exclamation marks per 1000 words"),
    ("questions_per_k",      "Question marks per 1000 words"),
    ("parens_per_k",         "Parentheses (open+close) per 1000 words"),
    # Group 5: Syntactic complexity (spaCy)
    ("parse_depth_mean",     "Mean parse tree depth across sentences"),
    ("subordinate_ratio",    "Ratio of subordinate clauses to total clauses"),
    # Group 6: Code-switching
    ("french_per_k",         "French word frequency per 1000 words"),
    ("english_per_k",        "English word frequency per 1000 words"),
    # Group 7: Readability
    ("word_len_mean",        "Mean word length in characters"),
    ("syllable_mean",        "Mean estimated syllable count per word"),
    ("para_len_mean",        "Mean paragraph length in words"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapters(data_path: Path) -> list[dict]:
    """Load chapter data from a JSON file with a 'chapters' array.

    Handles both Rayuela schema (token_count) and corpus schema (word_count)
    by normalizing to 'token_count' for compatibility.
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    chapters = data["chapters"]
    # Normalize field names: corpus uses word_count, Rayuela uses token_count
    for ch in chapters:
        if "token_count" not in ch and "word_count" in ch:
            ch["token_count"] = ch["word_count"]
    return chapters


# ---------------------------------------------------------------------------
# Helper: Spanish syllable counter
# ---------------------------------------------------------------------------

def count_syllables_es(word: str) -> int:
    """
    Estimate syllable count for a Spanish word.

    Spanish syllabification is more regular than English: every syllable
    has exactly one vowel nucleus. We count vowel groups (handling
    diphthongs and hiatuses approximately).
    """
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouáéíóúü"
    count = 0
    prev_vowel = False
    for char in word:
        if char in vowels:
            if not prev_vowel:
                count += 1
            prev_vowel = True
        else:
            prev_vowel = False

    return max(count, 1)


# ---------------------------------------------------------------------------
# Feature extraction — no spaCy required
# ---------------------------------------------------------------------------

def extract_basic_features(text: str) -> dict:
    """
    Extract features that don't require spaCy (groups 1-4, 6-7).

    This function handles sentence splitting, tokenization, function words,
    punctuation, code-switching, and readability — everything except
    syntactic parse trees.
    """
    # --- Sentence splitting (regex — good enough for counting) ---
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 0]
    if not sentences:
        sentences = [text]

    # --- Tokenize (lowercase words only, no punctuation) ---
    words = re.findall(r'[a-záéíóúüñàâçèêëîïôùûœæ]+', text.lower())
    n_words = len(words) if words else 1  # avoid division by zero
    word_counts = Counter(words)
    n_unique = len(word_counts)

    # --- Paragraph splitting ---
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    para_word_counts = [len(re.findall(r'\S+', p)) for p in paragraphs]

    # --- Group 1: Sentence structure ---
    sent_lengths = [len(re.findall(r'\S+', s)) for s in sentences]
    sent_arr = np.array(sent_lengths, dtype=float)

    features = {}
    features["sent_len_mean"] = float(np.mean(sent_arr))
    features["sent_len_median"] = float(np.median(sent_arr))
    features["sent_len_std"] = float(np.std(sent_arr)) if len(sent_arr) > 1 else 0.0
    features["sent_len_max"] = float(np.max(sent_arr))
    features["sent_len_cv"] = (
        features["sent_len_std"] / features["sent_len_mean"]
        if features["sent_len_mean"] > 0 else 0.0
    )

    # --- Group 2: Vocabulary richness ---
    # MATTR: Moving-Average TTR — length-independent vocabulary richness.
    # Window=50 to cover Rayuela's ultra-short expendable chapters (some < 20 words).
    # Only 10/155 chapters (6.5%) fall back to raw TTR with this window.
    # (Window=500 would force 72/155 chapters (46%) to fall back — unacceptable.)
    mattr_window = 50
    if len(words) <= mattr_window:
        features["mattr"] = n_unique / n_words  # fallback for ultra-short chapters
    else:
        window_ttrs = []
        for start in range(len(words) - mattr_window + 1):
            window = words[start:start + mattr_window]
            window_ttrs.append(len(set(window)) / mattr_window)
        features["mattr"] = float(np.mean(window_ttrs))
    hapax = sum(1 for count in word_counts.values() if count == 1)
    features["hapax_ratio"] = hapax / n_unique if n_unique > 0 else 0.0
    features["vocab_density"] = n_unique / len(sentences)

    # --- Group 3: Function words (per 1000 words) ---
    scale = 1000.0 / n_words
    features["articles_per_k"] = sum(word_counts.get(w, 0) for w in ARTICLES) * scale
    features["prepositions_per_k"] = sum(word_counts.get(w, 0) for w in PREPOSITIONS) * scale
    features["conjunctions_per_k"] = sum(word_counts.get(w, 0) for w in CONJUNCTIONS) * scale
    features["pronouns_per_k"] = sum(word_counts.get(w, 0) for w in PRONOUNS) * scale

    # --- Group 4: Punctuation profile (per 1000 words) ---
    features["semicolons_per_k"] = text.count(';') * scale
    features["colons_per_k"] = text.count(':') * scale
    # Em-dash: both Unicode em-dash and double-hyphen convention
    em_dashes = text.count('—') + text.count('–')
    features["em_dashes_per_k"] = em_dashes * scale
    # Ellipses: Unicode ellipsis OR three dots
    ellipses = text.count('…') + len(re.findall(r'\.{3}', text))
    features["ellipses_per_k"] = ellipses * scale
    features["exclamations_per_k"] = text.count('!') * scale
    # Count ¡ too — Spanish uses inverted marks
    features["exclamations_per_k"] += text.count('¡') * scale
    features["questions_per_k"] = (text.count('?') + text.count('¿')) * scale
    features["parens_per_k"] = (text.count('(') + text.count(')')) * scale

    # --- Group 6: Code-switching ---
    features["french_per_k"] = sum(
        word_counts.get(w, 0) for w in FRENCH_MARKERS
    ) * scale
    features["english_per_k"] = sum(
        word_counts.get(w, 0) for w in ENGLISH_MARKERS
    ) * scale

    # --- Group 7: Readability ---
    word_lengths = [len(w) for w in words]
    features["word_len_mean"] = float(np.mean(word_lengths)) if word_lengths else 0.0
    syllable_counts = [count_syllables_es(w) for w in words]
    features["syllable_mean"] = float(np.mean(syllable_counts)) if syllable_counts else 0.0
    features["para_len_mean"] = float(np.mean(para_word_counts)) if para_word_counts else 0.0

    return features


# ---------------------------------------------------------------------------
# Feature extraction — spaCy syntactic features
# ---------------------------------------------------------------------------

def tree_depth(token) -> int:
    """Recursively compute the depth of a token in the dependency tree."""
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)


def extract_syntactic_features(doc) -> dict:
    """
    Extract parse-tree-based features using a processed spaCy Doc.

    Group 5: Syntactic complexity
      - parse_depth_mean: average depth of dependency trees across sentences
      - subordinate_ratio: subordinate clauses / total clauses

    spaCy's dependency labels for subordinate clauses in Spanish include
    'advcl' (adverbial clause), 'acl' (adjectival clause), 'ccomp'
    (clausal complement), 'xcomp' (open clausal complement).
    """
    depths = []
    n_clauses = 0
    n_subordinate = 0

    subordinate_deps = {"advcl", "acl", "ccomp", "xcomp", "relcl"}

    for sent in doc.sents:
        # Tree depth: from the root of each sentence
        root = sent.root
        depths.append(tree_depth(root))

        # Count clauses: any verb head is a clause
        for token in sent:
            if token.pos_ == "VERB":
                n_clauses += 1
                if token.dep_ in subordinate_deps:
                    n_subordinate += 1

    features = {}
    features["parse_depth_mean"] = float(np.mean(depths)) if depths else 0.0
    features["subordinate_ratio"] = (
        n_subordinate / n_clauses if n_clauses > 0 else 0.0
    )

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scale A': Classical stylometric features (content-free)"
    )
    parser.add_argument(
        "--input", default=str(DATA_PATH),
        help=f"Input JSON path (default: {DATA_PATH.name})"
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    print("=" * 65)
    print("Scale A' — Classical Stylometric Features")
    print("Content-free by construction")
    print("=" * 65)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()

    # Load chapters
    chapters = load_chapters(input_path)
    print(f"Loaded {len(chapters)} chapters")
    print()

    # Load spaCy model
    print("Loading spaCy model (es_core_news_lg)...")
    nlp = spacy.load("es_core_news_lg")
    # Increase max length for long chapters (some exceed default 1M chars)
    nlp.max_length = 2_000_000
    print(f"  Pipeline: {nlp.pipe_names}")
    print()

    # Extract features for each chapter
    feature_names = [name for name, _ in FEATURE_SPEC]
    n_features = len(feature_names)
    all_features = np.zeros((len(chapters), n_features), dtype=np.float64)

    print(f"Extracting {n_features} features per chapter...")
    print("  Feature groups: sentence structure (5), vocabulary (3),")
    print("  function words (4), punctuation (7), syntax (2),")
    print("  code-switching (2), readability (3)")
    print()

    for i, ch in enumerate(chapters):
        text = ch["text"]

        # Basic features (fast — regex-based)
        basic = extract_basic_features(text)

        # Syntactic features (slow — requires full spaCy parse)
        doc = nlp(text)
        syntactic = extract_syntactic_features(doc)

        # Merge into feature vector
        combined = {**basic, **syntactic}
        for j, name in enumerate(feature_names):
            all_features[i, j] = combined[name]

        # Progress every 10 chapters
        if (i + 1) % 10 == 0 or i == 0 or i == len(chapters) - 1:
            print(f"  Ch.{ch['number']:>3d} ({i+1:>3d}/{len(chapters)}) — "
                  f"words={ch.get('token_count', 0):>5d}, "
                  f"MATTR={combined['mattr']:.3f}, "
                  f"sent_len_mean={combined['sent_len_mean']:.1f}, "
                  f"parse_depth={combined['parse_depth_mean']:.1f}")

    print()

    # --- Save outputs ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Feature matrix
    npy_path = output_dir / "chapter_stylometrics.npy"
    np.save(npy_path, all_features)
    print(f"Feature matrix saved: {npy_path}")
    print(f"  Shape: {all_features.shape}")

    # Metadata (feature descriptions + basic stats)
    metadata = {
        "feature_names": feature_names,
        "feature_descriptions": {name: desc for name, desc in FEATURE_SPEC},
        "n_chapters": len(chapters),
        "n_features": n_features,
        "feature_stats": {},
    }
    for j, name in enumerate(feature_names):
        col = all_features[:, j]
        metadata["feature_stats"][name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "median": float(np.median(col)),
        }

    meta_path = output_dir / "chapter_stylometrics_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Metadata saved:      {meta_path}")
    print()

    # --- Summary table ---
    print(f"Feature summary (all {len(chapters)} chapters):")
    print(f"  {'Feature':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─' * 22} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for j, name in enumerate(feature_names):
        col = all_features[:, j]
        print(f"  {name:<22} {np.mean(col):>8.2f} {np.std(col):>8.2f} "
              f"{np.min(col):>8.2f} {np.max(col):>8.2f}")

    print()
    print("Done. Run the trajectory permutation test next:")
    print("  python src/trajectory_stylometrics.py")


if __name__ == "__main__":
    main()
