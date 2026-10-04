#!/usr/bin/env python3
"""
Entity Resolution Pipeline — Amazon ML Challenge 2026

Two modes:
  baseline  — blocking + similarity threshold (fast, no training needed)
  ml        — blocking + LightGBM classifier (better, needs training phase)

Usage:
  # Quick baseline on sample (dev/debug):
  python pipeline.py baseline --data-dir ../../dataset --sample 5000

  # Full baseline (generates submission):
  python pipeline.py baseline --data-dir ../../dataset

  # Train ML model:
  python pipeline.py train --data-dir ../../dataset --sample 50000

  # Full ML pipeline (train + predict):
  python pipeline.py ml --data-dir ../../dataset

  # Evaluate on training data:
  python pipeline.py evaluate --data-dir ../../dataset --sample 10000
"""

import argparse
import os
import sys
import re
import time
import json
import pickle
import logging
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:
    from unidecode import unidecode
    HAS_UNIDECODE = True
except ImportError:
    HAS_UNIDECODE = False
    def unidecode(s): return s

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('pipeline')

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'top_k': 200,              # max candidates per S1 entity from blocking
    'max_df_ratio': 0.005,    # skip tokens in >0.5% of records
    'min_token_len': 2,       # min token length for blocking keys
    'baseline_name_weight': 0.45,
    'baseline_addr_weight': 0.35,
    'baseline_token_weight': 0.20,
    'baseline_threshold': 0.62,
    'neg_ratio': 5,           # negative:positive ratio for ML training
    'val_fraction': 0.15,     # validation split for threshold tuning
    'chunk_size': 10000,      # process S1 entities in chunks of this size
}

# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

LEGAL_SUFFIXES = {
    'pvt': 'private', 'priv': 'private',
    'ltd': 'limited', 'ltda': 'limited',
    'corp': 'corporation',
    'inc': 'incorporated',
    'co': 'company',
    'assoc': 'associates',
    'intl': 'international', 'intrl': 'international',
    'natl': 'national',
    'mfg': 'manufacturing',
    'svcs': 'services', 'svc': 'service',
    'grp': 'group',
    'hldg': 'holding', 'hldgs': 'holdings',
    'tech': 'technology', 'techs': 'technologies',
    'sol': 'solutions',
    'ent': 'enterprise', 'entr': 'enterprise',
    'mgmt': 'management',
    'dev': 'development',
    'sys': 'systems',
    'mkt': 'market', 'mktg': 'marketing',
    'comm': 'communication', 'comms': 'communications',
    'pharm': 'pharmaceutical',
    'fin': 'financial',
    'edu': 'education',
    'consult': 'consulting',
    'infra': 'infrastructure',
    'hosp': 'hospital',
    'med': 'medical',
    'dist': 'distribution',
    'prop': 'properties', 'props': 'properties',
    'sarl': 'sarl',   # French: keep as-is (société à responsabilité limitée)
    'sas': 'sas',     # French
    'eurl': 'eurl',   # French
    'sa': 'sa',        # French
}

ADDR_ABBREVS = {
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'pkwy': 'parkway', 'cir': 'circle', 'apt': 'apartment',
    'ste': 'suite', 'fl': 'floor', 'bldg': 'building',
    'hwy': 'highway', 'trl': 'trail', 'sq': 'square',
    'expy': 'expressway', 'rte': 'route', 'ctr': 'center',
}

# Words to IGNORE during blocking (only truly generic legal/structural terms)
BLOCK_STOPWORDS = frozenset({
    'private', 'limited', 'corporation', 'incorporated', 'company',
    'llc', 'llp', 'the', 'of', 'and', 'in', 'at', 'for', 'to',
    'a', 'an', 'is', 'it', 'on', 'by', 'dba',
    'pvt', 'ltd', 'inc', 'corp',
    'sarl', 'sas', 'sa', 'eurl', 'gmbh', 'ag',
    'limited', 'liability', 'partnership',
    'www', 'com', 'http', 'https', 'org', 'net',
})


def normalize_name(text):
    """Normalize a business name for comparison."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ''
    text = str(text)
    # Transliterate non-Latin scripts (Hindi, Kannada, etc.) to ASCII
    if HAS_UNIDECODE:
        text = unidecode(text)
    text = text.lower().strip()
    # Remove URLs and email-like patterns
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+\.\S+\.(com|org|net|in|co)\b', ' ', text)
    # Replace & with 'and'
    text = text.replace('&', ' and ')
    # Remove pipe and special delimiters
    text = text.replace('|', ' ')
    # Remove non-alphanumeric (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Expand legal suffix abbreviations
    tokens = text.split()
    expanded = []
    for t in tokens:
        t_clean = t.rstrip('.')
        if t_clean in LEGAL_SUFFIXES:
            expanded.append(LEGAL_SUFFIXES[t_clean])
        else:
            expanded.append(t)
    text = ' '.join(expanded)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(text):
    """Normalize a business address for comparison."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return ''
    text = str(text)
    if HAS_UNIDECODE:
        text = unidecode(text)
    text = text.lower().strip()
    text = text.replace('&', ' and ')
    text = text.replace('|', ' ')
    # Remove "near X" landmark references
    text = re.sub(r'\bnear\s+\S+(\s+\S+)?', ' ', text)
    # Remove non-alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    # Expand address abbreviations
    tokens = text.split()
    expanded = []
    for t in tokens:
        t_clean = t.rstrip('.')
        if t_clean in ADDR_ABBREVS:
            expanded.append(ADDR_ABBREVS[t_clean])
        else:
            expanded.append(t)
    text = ' '.join(expanded)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_country(text):
    """Normalize country string."""
    if not text or (isinstance(text, float) and np.isnan(text)):
        return 'unknown'
    return str(text).strip()


def get_blocking_tokens(name_norm):
    """Extract significant tokens from normalized name for blocking."""
    tokens = name_norm.split()
    significant = [t for t in tokens if t not in BLOCK_STOPWORDS and len(t) >= 2]
    return significant if significant else tokens[:3]


# Address words too common to be useful as blocking keys
ADDR_STOPWORDS = frozenset({
    'road', 'street', 'avenue', 'drive', 'lane', 'court', 'place',
    'boulevard', 'highway', 'circle', 'trail', 'square', 'parkway',
    'apartment', 'suite', 'floor', 'building', 'unit', 'block',
    'sector', 'phase', 'plot', 'no', 'near', 'opposite', 'behind',
    'main', 'new', 'old', 'east', 'west', 'north', 'south',
    'cross', 'st', 'rd', 'ave', 'dr', 'ln', 'ct', 'pl',
    'po', 'box', 'route', 'rte', 'town', 'city', 'village',
    'nagar', 'colony', 'vihar', 'puram', 'ganj', 'bagh', 'marg',
    'rue', 'boulevard', 'allée', 'chemin', 'place', 'impasse',  # French
})


def get_address_blocking_tokens(addr_norm):
    """Extract significant tokens from address for blocking (city names, etc.)."""
    tokens = addr_norm.split()
    significant = [t for t in tokens
                   if t not in ADDR_STOPWORDS and len(t) >= 3 and not t.isdigit()]
    return significant


def extract_numbers(text):
    """Extract numeric tokens (street/plot numbers, PINs) from text."""
    return set(re.findall(r'\b\d{2,}\b', text))  # 2+ digit numbers only


def get_prefix_keys(tokens, prefix_len=3):
    """Generate prefix-based blocking keys for typo tolerance."""
    return [t[:prefix_len] for t in tokens if len(t) >= prefix_len]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_source(path, sample=None):
    """Load a source TSV file."""
    log.info(f"Loading {os.path.basename(path)}...")
    t0 = time.time()
    df = pd.read_csv(path, sep='\t', dtype=str, engine='c',
                     keep_default_na=False, na_filter=False)
    if sample and len(df) > sample:
        df = df.head(sample)
    log.info(f"  → {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def load_ground_truth(path, s1_ids=None):
    """Load ground truth, return dict: s1_id -> set(matched_ids)."""
    log.info(f"Loading ground truth from {os.path.basename(path)}...")
    df = pd.read_csv(path, sep='\t', dtype=str, engine='c',
                     keep_default_na=False, na_filter=False)
    gt = {}
    for row in df.itertuples(index=False):
        s1_id = row.source1_entity_id
        if s1_ids and s1_id not in s1_ids:
            continue
        matched = row.matched_entity_ids
        gt[s1_id] = set(matched.split(',')) if matched else set()
    log.info(f"  → {len(gt):,} entries")
    return gt


# ============================================================================
# PREPROCESSING — add normalized columns
# ============================================================================

def preprocess_df(df):
    """Add normalized name/address columns to dataframe."""
    log.info(f"  Normalizing {len(df):,} records...")
    t0 = time.time()
    df['name_norm'] = df['business_name'].apply(normalize_name)
    df['addr_norm'] = df['business_address'].apply(normalize_address)
    df['country_norm'] = df['country'].apply(normalize_country)
    log.info(f"  Done in {time.time()-t0:.1f}s")
    return df


# ============================================================================
# BLOCKING — CANDIDATE GENERATION
# ============================================================================

def build_blocking_index(s23_df, config):
    """
    Build inverted indices on S2/S3 for multiple blocking strategies.
    Returns 4 indices: token, prefix, number, address-word.
    """
    log.info("Building blocking index...")
    t0 = time.time()

    max_df_ratio = config.get('max_df_ratio', 0.005)

    # Count tokens per country for DF filtering
    country_sizes = Counter()
    token_country_counts = defaultdict(lambda: defaultdict(int))
    addr_token_country_counts = defaultdict(lambda: defaultdict(int))

    for row in s23_df.itertuples(index=False):
        country = row.country_norm
        country_sizes[country] += 1
        for tok in set(get_blocking_tokens(row.name_norm)):
            token_country_counts[country][tok] += 1
        for tok in set(get_address_blocking_tokens(row.addr_norm)):
            addr_token_country_counts[country][tok] += 1

    # Build indices, skipping high-DF tokens
    token_index = defaultdict(lambda: defaultdict(set))
    prefix_index = defaultdict(lambda: defaultdict(set))
    number_index = defaultdict(lambda: defaultdict(set))
    addr_word_index = defaultdict(lambda: defaultdict(set))

    for row in s23_df.itertuples(index=False):
        eid = row.entity_id
        country = row.country_norm
        name = row.name_norm
        addr = row.addr_norm
        max_df = max(int(country_sizes[country] * max_df_ratio), 50)

        # Name token index
        tokens = get_blocking_tokens(name)
        for tok in set(tokens):
            if token_country_counts[country][tok] <= max_df:
                token_index[country][tok].add(eid)

        # Prefix index (3-char prefixes of name tokens)
        for prefix in set(get_prefix_keys(tokens)):
            prefix_index[country][prefix].add(eid)

        # Address number index
        for num in extract_numbers(addr):
            number_index[country][num].add(eid)

        # Address word index (city names, distinctive location words)
        addr_tokens = get_address_blocking_tokens(addr)
        addr_max_df = max(int(country_sizes[country] * max_df_ratio * 2), 100)
        for tok in set(addr_tokens):
            if addr_token_country_counts[country][tok] <= addr_max_df:
                addr_word_index[country][tok].add(eid)

    elapsed = time.time() - t0
    log.info(f"  Index built in {elapsed:.1f}s")
    for c in sorted(token_index.keys()):
        log.info(f"  {c}: {len(token_index[c]):,} name-tokens, "
                 f"{len(prefix_index[c]):,} prefixes, "
                 f"{len(number_index[c]):,} numbers, "
                 f"{len(addr_word_index[c]):,} addr-words")

    return token_index, prefix_index, number_index, addr_word_index


def generate_candidates(s1_df, token_index, prefix_index, number_index,
                        addr_word_index, config):
    """
    For each S1 entity, find candidates via INDEPENDENT strategy pools.
    Each strategy contributes its own top-K, then we union them.
    This ensures matches findable only by address or prefix aren't drowned out.
    """
    top_k = config.get('top_k', 200)
    log.info(f"Generating candidates (max {top_k} per entity)...")
    t0 = time.time()

    candidates = {}
    n_total = len(s1_df)
    total_candidates = 0

    for i, row in enumerate(s1_df.itertuples(index=False)):
        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            log.info(f"  {i+1:,}/{n_total:,} ({100*(i+1)/n_total:.1f}%) "
                     f"[{rate:.0f} ent/s, ETA {eta/60:.1f}min]")

        s1_id = row.entity_id
        name = row.name_norm
        addr = row.addr_norm
        country = row.country_norm

        # Independent strategy pools
        token_scores = Counter()
        prefix_scores = Counter()
        number_scores = Counter()
        addr_word_scores = Counter()

        name_tokens = get_blocking_tokens(name)

        # Strategy 1: Exact name-token overlap
        if country in token_index:
            for tok in name_tokens:
                if tok in token_index[country]:
                    for cid in token_index[country][tok]:
                        token_scores[cid] += 1

        # Strategy 2: Name-prefix overlap (typo tolerance)
        prefixes = get_prefix_keys(name_tokens)
        if country in prefix_index:
            for pfx in prefixes:
                if pfx in prefix_index[country]:
                    for cid in prefix_index[country][pfx]:
                        prefix_scores[cid] += 1

        # Strategy 3: Address number overlap
        addr_nums = extract_numbers(addr)
        if addr_nums and country in number_index:
            for num in addr_nums:
                if num in number_index[country]:
                    for cid in number_index[country][num]:
                        number_scores[cid] += 1

        # Strategy 4: Address word overlap (catches DBA/trade-name cases)
        addr_tokens = get_address_blocking_tokens(addr)
        if addr_tokens and country in addr_word_index:
            for tok in addr_tokens:
                if tok in addr_word_index[country]:
                    for cid in addr_word_index[country][tok]:
                        addr_word_scores[cid] += 1

        # INDEPENDENT top-K from each strategy, then union
        result = set()
        # Name tokens: most reliable, give more slots
        result.update(cid for cid, _ in token_scores.most_common(min(top_k, 100)))
        # Prefixes: catches typos
        result.update(cid for cid, _ in prefix_scores.most_common(min(top_k, 80)))
        # Address numbers: catches different-name-same-address
        result.update(cid for cid, _ in number_scores.most_common(min(top_k, 30)))
        # Address words: catches DBA/trade names at same location
        result.update(cid for cid, _ in addr_word_scores.most_common(min(top_k, 30)))

        # Cap total if way too many
        if len(result) > top_k:
            # Rank by combined score and keep top_k
            combined = Counter()
            for cid in result:
                combined[cid] = (token_scores[cid] * 3 +
                                 prefix_scores[cid] +
                                 number_scores[cid] * 2 +
                                 addr_word_scores[cid] * 2)
            result = set(cid for cid, _ in combined.most_common(top_k))

        candidates[s1_id] = list(result)
        total_candidates += len(result)

    elapsed = time.time() - t0
    n_with = sum(1 for v in candidates.values() if v)
    avg_cands = total_candidates / max(n_total, 1)
    log.info(f"  Done in {elapsed:.1f}s — {n_with:,}/{n_total:,} have candidates "
             f"({100*n_with/n_total:.1f}%), avg {avg_cands:.1f} candidates/entity")

    return candidates


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

FEATURE_NAMES = [
    'name_jw', 'name_ratio', 'name_partial', 'name_token_sort', 'name_token_set',
    'addr_jw', 'addr_ratio', 'addr_partial', 'addr_token_sort', 'addr_token_set',
    'name_jaccard', 'addr_jaccard',
    'num_match', 'name_len_ratio', 'addr_len_ratio',
    'shared_token_frac',
]


def compute_pair_features(s1_name, s1_addr, s23_name, s23_addr):
    """Compute similarity features for one (S1, S2/S3) pair. Returns list."""
    # Name similarities
    name_jw = JaroWinkler.similarity(s1_name, s23_name) if s1_name and s23_name else 0.0
    name_ratio = fuzz.ratio(s1_name, s23_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, s23_name) / 100.0
    name_tsort = fuzz.token_sort_ratio(s1_name, s23_name) / 100.0
    name_tset = fuzz.token_set_ratio(s1_name, s23_name) / 100.0

    # Address similarities
    addr_jw = JaroWinkler.similarity(s1_addr, s23_addr) if s1_addr and s23_addr else 0.0
    addr_ratio = fuzz.ratio(s1_addr, s23_addr) / 100.0
    addr_partial = fuzz.partial_ratio(s1_addr, s23_addr) / 100.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, s23_addr) / 100.0
    addr_tset = fuzz.token_set_ratio(s1_addr, s23_addr) / 100.0

    # Token Jaccard (name)
    s1_ntok = set(s1_name.split()) if s1_name else set()
    s23_ntok = set(s23_name.split()) if s23_name else set()
    union_n = len(s1_ntok | s23_ntok)
    name_jaccard = len(s1_ntok & s23_ntok) / union_n if union_n else 0.0

    # Token Jaccard (address)
    s1_atok = set(s1_addr.split()) if s1_addr else set()
    s23_atok = set(s23_addr.split()) if s23_addr else set()
    union_a = len(s1_atok | s23_atok)
    addr_jaccard = len(s1_atok & s23_atok) / union_a if union_a else 0.0

    # Numeric overlap in address
    s1_nums = extract_numbers(s1_addr)
    s23_nums = extract_numbers(s23_addr)
    union_nums = len(s1_nums | s23_nums)
    num_match = len(s1_nums & s23_nums) / union_nums if union_nums else 0.0

    # Length ratios
    name_lr = min(len(s1_name), len(s23_name)) / max(len(s1_name), len(s23_name), 1)
    addr_lr = min(len(s1_addr), len(s23_addr)) / max(len(s1_addr), len(s23_addr), 1)

    # Shared blocking token fraction
    s1_btok = set(get_blocking_tokens(s1_name))
    s23_btok = set(get_blocking_tokens(s23_name))
    union_b = len(s1_btok | s23_btok)
    shared_frac = len(s1_btok & s23_btok) / union_b if union_b else 0.0

    return [
        name_jw, name_ratio, name_partial, name_tsort, name_tset,
        addr_jw, addr_ratio, addr_partial, addr_tsort, addr_tset,
        name_jaccard, addr_jaccard,
        num_match, name_lr, addr_lr,
        shared_frac,
    ]


def compute_baseline_score(features):
    """Simple weighted score for baseline (no ML model)."""
    # features indices: 3=name_token_sort, 4=name_token_set,
    #                   8=addr_token_sort, 10=name_jaccard, 15=shared_token_frac
    name_score = 0.5 * features[3] + 0.5 * features[4]   # token_sort + token_set
    addr_score = 0.5 * features[8] + 0.5 * features[9]    # addr_token_sort + addr_token_set
    token_score = features[15]                              # shared blocking token frac
    return 0.45 * name_score + 0.35 * addr_score + 0.20 * token_score


# ============================================================================
# SCORING / EVALUATION
# ============================================================================

def f05_score(precision, recall):
    """Compute F0.5 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_predictions(predictions, ground_truth):
    """
    Compute macro-averaged F0.5.
    predictions: dict s1_id -> set(predicted_ids)
    ground_truth: dict s1_id -> set(true_ids)
    """
    scores = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())

        if len(true_ids) == 0 and len(pred_ids) == 0:
            scores.append(1.0)  # singleton correctly predicted
        elif len(true_ids) == 0 and len(pred_ids) > 0:
            scores.append(0.0)  # false merge on singleton
        elif len(true_ids) > 0 and len(pred_ids) == 0:
            scores.append(0.0)  # missed all matches
        else:
            tp = len(pred_ids & true_ids)
            precision = tp / len(pred_ids) if pred_ids else 0.0
            recall = tp / len(true_ids) if true_ids else 0.0
            scores.append(f05_score(precision, recall))

    macro_f05 = np.mean(scores)
    return macro_f05


def tune_threshold(all_scores, ground_truth, s1_ids):
    """
    Sweep thresholds to find optimal F0.5.
    all_scores: dict s1_id -> list of (cand_id, score)
    Returns: best_threshold, best_f05
    """
    log.info("Tuning threshold for F0.5...")
    thresholds = np.arange(0.30, 0.95, 0.02)
    best_t, best_f = 0.5, 0.0

    for t in thresholds:
        preds = {}
        for s1_id in s1_ids:
            if s1_id in all_scores:
                preds[s1_id] = set(
                    cid for cid, sc in all_scores[s1_id] if sc >= t
                )
            else:
                preds[s1_id] = set()

        f05 = evaluate_predictions(preds, ground_truth)
        if f05 > best_f:
            best_f = f05
            best_t = t

    log.info(f"  Best threshold: {best_t:.3f} → F0.5 = {best_f:.4f}")
    return best_t, best_f


# ============================================================================
# ML MODEL (LightGBM)
# ============================================================================

def prepare_training_data(s1_df, s23_lookup, candidates, ground_truth, config):
    """
    Build training dataset from blocked candidates + ground truth.
    Returns: X (features array), y (labels), pair_info (for debugging)
    """
    log.info("Preparing training data...")
    t0 = time.time()
    neg_ratio = config.get('neg_ratio', 5)

    X_rows = []
    y_labels = []
    pair_info = []

    n_pos = 0
    n_neg = 0

    for s1_id in candidates:
        if s1_id not in ground_truth:
            continue

        true_ids = ground_truth[s1_id]
        s1_rec = None
        # Find s1 record
        mask = s1_df['entity_id'] == s1_id
        if mask.any():
            s1_row = s1_df[mask].iloc[0]
            s1_name = s1_row['name_norm']
            s1_addr = s1_row['addr_norm']
        else:
            continue

        cand_ids = candidates[s1_id]
        neg_cands_this = []

        for cand_id in cand_ids:
            if cand_id not in s23_lookup:
                continue
            s23_name, s23_addr = s23_lookup[cand_id]
            feats = compute_pair_features(s1_name, s1_addr, s23_name, s23_addr)

            if cand_id in true_ids:
                X_rows.append(feats)
                y_labels.append(1)
                pair_info.append((s1_id, cand_id))
                n_pos += 1
            else:
                neg_cands_this.append((feats, cand_id))

        # Subsample negatives
        np.random.shuffle(neg_cands_this)
        max_neg = max(neg_ratio * max(len(true_ids), 1), 3)
        for feats, cid in neg_cands_this[:max_neg]:
            X_rows.append(feats)
            y_labels.append(0)
            pair_info.append((s1_id, cid))
            n_neg += 1

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.int32)

    elapsed = time.time() - t0
    log.info(f"  Training data: {n_pos:,} positives, {n_neg:,} negatives "
             f"({len(X):,} total) in {elapsed:.1f}s")

    return X, y, pair_info


def train_lgb_model(X, y, config):
    """Train LightGBM model for match/non-match classification."""
    log.info("Training LightGBM model...")
    import lightgbm as lgb

    # Split into train/val for early stopping
    val_frac = config.get('val_fraction', 0.15)
    n_val = int(len(X) * val_frac)
    indices = np.random.permutation(len(X))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos = neg_count / max(pos_count, 1)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos,
        'verbose': -1,
        'n_jobs': -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50),
        ],
    )

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    imp_sorted = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
    log.info("  Feature importance (gain):")
    for fname, fimp in imp_sorted[:8]:
        log.info(f"    {fname}: {fimp:.1f}")

    return model


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_s23_lookup(s23_df):
    """Build dict: entity_id -> (name_norm, addr_norm) for fast lookups."""
    log.info("Building S2/S3 lookup table...")
    t0 = time.time()
    lookup = {}
    for row in s23_df.itertuples(index=False):
        lookup[row.entity_id] = (row.name_norm, row.addr_norm)
    log.info(f"  {len(lookup):,} records indexed in {time.time()-t0:.1f}s")
    return lookup


def build_s1_lookup(s1_df):
    """Build dict: entity_id -> (name_norm, addr_norm, country_norm)."""
    lookup = {}
    for row in s1_df.itertuples(index=False):
        lookup[row.entity_id] = (row.name_norm, row.addr_norm, row.country_norm)
    return lookup


def score_candidates(s1_df, s23_lookup, candidates, model=None):
    """
    Score all candidate pairs. Returns dict: s1_id -> [(cand_id, score), ...]
    If model is provided, use ML scoring. Otherwise, use baseline heuristic.
    """
    log.info(f"Scoring candidates ({'ML' if model else 'baseline'})...")
    t0 = time.time()

    s1_lookup = build_s1_lookup(s1_df)
    all_scores = {}
    n_total = len(candidates)
    n_pairs = 0

    for i, (s1_id, cand_ids) in enumerate(candidates.items()):
        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            log.info(f"  Scoring {i+1:,}/{n_total:,} ({100*(i+1)/n_total:.1f}%) "
                     f"[{rate:.0f} ent/s, ETA {eta/60:.1f}min]")

        if not cand_ids:
            all_scores[s1_id] = []
            continue

        s1_name, s1_addr, _ = s1_lookup[s1_id]

        if model:
            # Batch compute features for all candidates of this entity
            feat_batch = []
            valid_cids = []
            for cid in cand_ids:
                if cid in s23_lookup:
                    s23_name, s23_addr = s23_lookup[cid]
                    feat_batch.append(compute_pair_features(s1_name, s1_addr, s23_name, s23_addr))
                    valid_cids.append(cid)

            if feat_batch:
                X = np.array(feat_batch, dtype=np.float32)
                probs = model.predict(X)
                scored = list(zip(valid_cids, probs.tolist()))
            else:
                scored = []
        else:
            # Baseline scoring
            scored = []
            for cid in cand_ids:
                if cid in s23_lookup:
                    s23_name, s23_addr = s23_lookup[cid]
                    feats = compute_pair_features(s1_name, s1_addr, s23_name, s23_addr)
                    score = compute_baseline_score(feats)
                    scored.append((cid, score))

        all_scores[s1_id] = scored
        n_pairs += len(scored)

    elapsed = time.time() - t0
    log.info(f"  Scored {n_pairs:,} pairs in {elapsed:.1f}s")
    return all_scores


def apply_threshold(all_scores, threshold, all_s1_ids):
    """Apply threshold to scores, return predictions dict."""
    predictions = {}
    for s1_id in all_s1_ids:
        if s1_id in all_scores:
            matched = set(cid for cid, sc in all_scores[s1_id] if sc >= threshold)
        else:
            matched = set()
        predictions[s1_id] = matched
    return predictions


def write_output(predictions, candidates, all_s1_ids, output_dir):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    os.makedirs(output_dir, exist_ok=True)

    # matching_results.tsv
    match_path = os.path.join(output_dir, 'matching_results.tsv')
    log.info(f"Writing {match_path}...")
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for s1_id in sorted(all_s1_ids):
            matched = predictions.get(s1_id, set())
            matched_str = ','.join(sorted(matched))
            f.write(f'{s1_id}\t{matched_str}\n')

    n_matched = sum(1 for v in predictions.values() if v)
    n_empty = len(all_s1_ids) - n_matched
    log.info(f"  {n_matched:,} matched, {n_empty:,} singletons, {len(all_s1_ids):,} total")

    # candidate_pairs.tsv
    cand_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    log.info(f"Writing {cand_path}...")
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for s1_id in sorted(all_s1_ids):
            cands = candidates.get(s1_id, [])
            cand_str = ','.join(sorted(cands))
            f.write(f'{s1_id}\t{cand_str}\n')

    log.info("  Output files written successfully.")
    return match_path, cand_path


# ============================================================================
# BLOCKING RECALL MEASUREMENT
# ============================================================================

def measure_blocking_recall(candidates, ground_truth):
    """Measure what fraction of true matches are in the candidate sets."""
    total_true = 0
    found = 0
    missed_examples = []

    for s1_id, true_ids in ground_truth.items():
        if not true_ids:
            continue
        total_true += len(true_ids)
        cand_set = set(candidates.get(s1_id, []))
        found_ids = true_ids & cand_set
        found += len(found_ids)
        missed = true_ids - cand_set
        if missed and len(missed_examples) < 5:
            missed_examples.append((s1_id, missed))

    recall = found / total_true if total_true else 0.0
    log.info(f"  Blocking recall: {found:,}/{total_true:,} = {recall:.4f} "
             f"({100*recall:.2f}%)")
    if missed_examples:
        log.info("  Example missed matches:")
        for s1_id, missed in missed_examples[:3]:
            log.info(f"    {s1_id} missed: {missed}")

    return recall


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

def save_progress(progress_file, stage, info):
    """Append progress info to a JSON-lines file."""
    entry = {'stage': stage, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **info}
    with open(progress_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# ============================================================================
# MAIN ENTRY POINTS
# ============================================================================

def run_baseline(args, config):
    """Run baseline pipeline (blocking + threshold, no ML)."""
    data_dir = args.data_dir
    output_dir = args.output_dir
    sample = args.sample
    split = args.split
    progress_file = os.path.join(output_dir, 'progress.jsonl')

    log.info(f"{'='*60}")
    log.info(f"BASELINE PIPELINE — split={split}, sample={sample}")
    log.info(f"{'='*60}")

    # Load data — always load FULL S2/S3 (matches scattered across entire dataset)
    s1 = load_source(os.path.join(data_dir, f'{split}_source1.tsv'), sample)
    s2 = load_source(os.path.join(data_dir, f'{split}_source2.tsv'))
    s3 = load_source(os.path.join(data_dir, f'{split}_source3.tsv'))

    # Preprocess
    s1 = preprocess_df(s1)
    s2 = preprocess_df(s2)
    s3 = preprocess_df(s3)

    # Combine S2 + S3
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    save_progress(progress_file, 'data_loaded',
                  {'s1_rows': len(s1), 's23_rows': len(s23)})

    # Build blocking index
    token_idx, prefix_idx, number_idx, addr_idx = build_blocking_index(s23, config)
    save_progress(progress_file, 'index_built', {})

    # Generate candidates
    candidates = generate_candidates(s1, token_idx, prefix_idx, number_idx, addr_idx, config)

    # Free index memory
    del token_idx, prefix_idx, number_idx, addr_idx

    # If training split, measure blocking recall
    gt = None
    if split == 'train':
        gt_path = os.path.join(data_dir, f'{split}_ground_truth.tsv')
        if os.path.exists(gt_path):
            gt = load_ground_truth(gt_path, set(s1['entity_id']))
            blocking_recall = measure_blocking_recall(candidates, gt)
            save_progress(progress_file, 'blocking_recall',
                          {'recall': blocking_recall})

    # Build lookup and score
    s23_lookup = build_s23_lookup(s23)
    del s23

    all_scores = score_candidates(s1, s23_lookup, candidates, model=None)

    # Determine threshold
    threshold = config['baseline_threshold']
    if gt:
        # Tune threshold on the data
        threshold, best_f05 = tune_threshold(all_scores, gt, set(s1['entity_id']))
        save_progress(progress_file, 'threshold_tuned',
                      {'threshold': threshold, 'f05': best_f05})

    # Apply threshold
    all_s1_ids = set(s1['entity_id'])
    predictions = apply_threshold(all_scores, threshold, all_s1_ids)

    # Evaluate if ground truth available
    if gt:
        f05 = evaluate_predictions(predictions, gt)
        log.info(f"  *** Validation F0.5 = {f05:.4f} ***")
        save_progress(progress_file, 'evaluation', {'f05': f05, 'threshold': threshold})

    # Write output
    match_path, cand_path = write_output(predictions, candidates, all_s1_ids, output_dir)
    save_progress(progress_file, 'output_written',
                  {'match_path': match_path, 'cand_path': cand_path})

    return match_path, cand_path


def run_ml(args, config):
    """Run ML pipeline: train on train split, predict on test split."""
    data_dir = args.data_dir
    output_dir = args.output_dir
    sample = args.sample
    model_path = args.model_path
    progress_file = os.path.join(output_dir, 'progress.jsonl')

    log.info(f"{'='*60}")
    log.info(f"ML PIPELINE — sample={sample}")
    log.info(f"{'='*60}")

    # ---- PHASE 1: Train model on training data ----
    if model_path and os.path.exists(model_path):
        log.info(f"Loading pre-trained model from {model_path}...")
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        model = saved['model']
        threshold = saved['threshold']
        log.info(f"  Loaded model, threshold={threshold:.3f}")
    else:
        log.info("PHASE 1: Training on train split...")

        # Load training data — FULL S2/S3 (matches scattered)
        train_dir = os.path.join(data_dir, 'train') if os.path.isdir(os.path.join(data_dir, 'train')) else data_dir
        train_s1 = load_source(os.path.join(train_dir, 'train_source1.tsv'), sample)
        train_s2 = load_source(os.path.join(train_dir, 'train_source2.tsv'))
        train_s3 = load_source(os.path.join(train_dir, 'train_source3.tsv'))

        train_s1 = preprocess_df(train_s1)
        train_s2 = preprocess_df(train_s2)
        train_s3 = preprocess_df(train_s3)

        train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)
        del train_s2, train_s3

        # Ground truth
        gt = load_ground_truth(
            os.path.join(train_dir, 'train_ground_truth.tsv'),
            set(train_s1['entity_id'])
        )

        # Block
        tok_idx, pfx_idx, num_idx, addr_idx = build_blocking_index(train_s23, config)
        train_candidates = generate_candidates(
            train_s1, tok_idx, pfx_idx, num_idx, addr_idx, config
        )
        del tok_idx, pfx_idx, num_idx, addr_idx

        blocking_recall = measure_blocking_recall(train_candidates, gt)
        save_progress(progress_file, 'train_blocking_recall', {'recall': blocking_recall})

        # Build lookup
        train_s23_lookup = build_s23_lookup(train_s23)
        del train_s23

        # Prepare training data (features + labels)
        X, y, pair_info = prepare_training_data(
            train_s1, train_s23_lookup, train_candidates, gt, config
        )

        # Train model
        model = train_lgb_model(X, y, config)

        # Tune threshold on training data scores
        # Score a subset for threshold tuning
        log.info("Scoring training candidates for threshold tuning...")
        val_s1_ids = list(gt.keys())
        np.random.shuffle(val_s1_ids)
        val_s1_ids = val_s1_ids[:min(50000, len(val_s1_ids))]
        val_s1_set = set(val_s1_ids)
        val_s1_df = train_s1[train_s1['entity_id'].isin(val_s1_set)]
        val_candidates = {k: v for k, v in train_candidates.items() if k in val_s1_set}

        val_scores = score_candidates(val_s1_df, train_s23_lookup, val_candidates, model)
        val_gt = {k: v for k, v in gt.items() if k in val_s1_set}
        threshold, val_f05 = tune_threshold(val_scores, val_gt, val_s1_set)

        save_progress(progress_file, 'model_trained',
                      {'threshold': threshold, 'val_f05': val_f05,
                       'n_train': len(X), 'blocking_recall': blocking_recall})

        # Save model
        model_save_path = os.path.join(output_dir, 'model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump({'model': model, 'threshold': threshold}, f)
        log.info(f"  Model saved to {model_save_path}")

        del train_s1, train_s23_lookup, train_candidates, X, y

    # ---- PHASE 2: Predict on test data ----
    log.info("PHASE 2: Predicting on test split...")

    test_dir = os.path.join(data_dir, 'test') if os.path.isdir(os.path.join(data_dir, 'test')) else data_dir
    test_s1 = load_source(os.path.join(test_dir, 'test_source1.tsv'))
    test_s2 = load_source(os.path.join(test_dir, 'test_source2.tsv'))
    test_s3 = load_source(os.path.join(test_dir, 'test_source3.tsv'))

    test_s1 = preprocess_df(test_s1)
    test_s2 = preprocess_df(test_s2)
    test_s3 = preprocess_df(test_s3)

    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)
    del test_s2, test_s3

    # Block
    tok_idx, pfx_idx, num_idx, addr_idx = build_blocking_index(test_s23, config)
    test_candidates = generate_candidates(test_s1, tok_idx, pfx_idx, num_idx, addr_idx, config)
    del tok_idx, pfx_idx, num_idx, addr_idx

    # Score
    test_s23_lookup = build_s23_lookup(test_s23)
    del test_s23

    test_scores = score_candidates(test_s1, test_s23_lookup, test_candidates, model)
    del test_s23_lookup

    # Apply threshold
    all_s1_ids = set(test_s1['entity_id'])
    predictions = apply_threshold(test_scores, threshold, all_s1_ids)

    # Write output
    match_path, cand_path = write_output(predictions, test_candidates, all_s1_ids, output_dir)
    save_progress(progress_file, 'test_output_written',
                  {'match_path': match_path, 'cand_path': cand_path, 'threshold': threshold})

    return match_path, cand_path


def run_evaluate(args, config):
    """Evaluate on training data with cross-validation style split."""
    data_dir = args.data_dir
    sample = args.sample or 20000  # default to smaller sample for eval
    output_dir = args.output_dir

    log.info(f"{'='*60}")
    log.info(f"EVALUATION MODE — sample={sample}")
    log.info(f"{'='*60}")

    # Load training data — FULL S2/S3 (matches are scattered)
    s1 = load_source(os.path.join(data_dir, 'train_source1.tsv'), sample)
    s2 = load_source(os.path.join(data_dir, 'train_source2.tsv'))
    s3 = load_source(os.path.join(data_dir, 'train_source3.tsv'))

    s1 = preprocess_df(s1)
    s2 = preprocess_df(s2)
    s3 = preprocess_df(s3)

    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3

    # Ground truth
    gt = load_ground_truth(
        os.path.join(data_dir, 'train_ground_truth.tsv'),
        set(s1['entity_id'])
    )

    # Block
    tok_idx, pfx_idx, num_idx, addr_idx = build_blocking_index(s23, config)
    candidates = generate_candidates(s1, tok_idx, pfx_idx, num_idx, addr_idx, config)
    del tok_idx, pfx_idx, num_idx, addr_idx

    # Blocking recall
    blocking_recall = measure_blocking_recall(candidates, gt)

    # Build lookup and score (baseline)
    s23_lookup = build_s23_lookup(s23)
    del s23

    all_scores = score_candidates(s1, s23_lookup, candidates, model=None)

    # Tune threshold
    s1_ids = set(s1['entity_id'])
    threshold, best_f05 = tune_threshold(all_scores, gt, s1_ids)

    # Detailed analysis
    predictions = apply_threshold(all_scores, threshold, s1_ids)
    f05 = evaluate_predictions(predictions, gt)

    n_predicted = sum(len(v) for v in predictions.values())
    n_true = sum(len(v) for v in gt.values())
    n_correct = sum(len(predictions.get(s1, set()) & gt[s1]) for s1 in gt)

    log.info(f"\n{'='*40}")
    log.info(f"EVALUATION RESULTS")
    log.info(f"{'='*40}")
    log.info(f"  Blocking recall:     {blocking_recall:.4f}")
    log.info(f"  Best threshold:      {threshold:.3f}")
    log.info(f"  F0.5 score:          {f05:.4f}")
    log.info(f"  Total predicted:     {n_predicted:,}")
    log.info(f"  Total true matches:  {n_true:,}")
    log.info(f"  Correct matches:     {n_correct:,}")
    log.info(f"  Overall precision:   {n_correct/max(n_predicted,1):.4f}")
    log.info(f"  Overall recall:      {n_correct/max(n_true,1):.4f}")

    return f05


def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('mode', choices=['baseline', 'ml', 'evaluate'],
                        help='Pipeline mode')
    parser.add_argument('--data-dir', default='../../dataset/train',
                        help='Path to data directory (containing source TSV files)')
    parser.add_argument('--output-dir', default='../../output',
                        help='Output directory for results')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N rows from S1 (and N*3 from S2/S3)')
    parser.add_argument('--split', default='test',
                        help='Data split: train or test')
    parser.add_argument('--model-path', default=None,
                        help='Path to pre-trained model (for ml mode)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override matching threshold')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Override top_k candidates')

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.threshold:
        config['baseline_threshold'] = args.threshold
    if args.top_k:
        config['top_k'] = args.top_k

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()

    if args.mode == 'baseline':
        run_baseline(args, config)
    elif args.mode == 'ml':
        run_ml(args, config)
    elif args.mode == 'evaluate':
        run_evaluate(args, config)

    total_time = time.time() - t_start
    log.info(f"\nTotal pipeline time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
