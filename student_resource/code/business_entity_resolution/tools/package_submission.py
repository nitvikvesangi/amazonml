#!/usr/bin/env python3
"""Build the team submission package: staging tree + zip, in the organisers' layout.

    <team>_submission.zip
      output/matching_results.tsv        # byte-identical to the leaderboard upload
      output/candidate_pairs.tsv         # the blocking candidate set fed to the model
      code/business_entity_resolution/
        src/                             # ALL source: library modules + CLI drivers,
        |                                # campaign shell wrappers in src/campaign_scripts/
        artifacts/                       # reference artefacts of the shipped run
        README.md                        # end-to-end reproduction instructions
        requirements.txt                 # pinned dependencies / environment
      Documentation_template.md          # filled-in methodology write-up
      MANIFEST.json                      # provenance: sizes, sha256, reports, versions

The staging tree is written first (so it can be inspected and smoke-tested), then zipped.
Relative paths inside the tree match the development checkout, so every command in the
README works from `code/business_entity_resolution/` with `dataset/` and `utils/` next to
`output/` at the archive root.

    python3 tools/package_submission.py --team myteam \
        --stage /Users/you/submission_stage \
        --out /Users/you/myteam_submission.zip \
        --cand-parts ../../output/pred500k_C/candidate_pairs.tsv \
                     ../../output/pred500k_D/candidate_pairs.tsv \
        --compress --note "LB 0.914, 6th submission"
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

STAGE_MARKER = '.submission_stage'
SKIP_DIRS = {'__pycache__', '.venv-sem', 'logs', '.git'}
SKIP_FILES = {'.DS_Store', STAGE_MARKER}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def place(src, dst, link=False):
    """Copy `src` to `dst` (link=True hardlinks instead, for throwaway big files)."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        raise SystemExit(f"refusing to overwrite {dst} (two sources share a name?)")
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def merge_tsvs(parts, out):
    """Concatenate TSVs keeping exactly one header (the candidate halves)."""
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for i, part in enumerate(parts):
            with open(part, encoding='utf-8') as g:
                first = g.readline()
                if not first.startswith('source1_entity_id'):
                    raise SystemExit(f"{part}: unexpected header {first[:60]!r}")
                if i == 0:
                    f.write(first)
                for line in g:
                    f.write(line)
                    n += 1
    return n


def reset_stage(stage):
    if os.path.exists(stage):
        if not (os.path.exists(os.path.join(stage, STAGE_MARKER))
                or not os.listdir(stage)):
            raise SystemExit(f"refusing to delete {stage}: not a previous staging tree")
        log(f"clearing previous staging tree {stage}")
        shutil.rmtree(stage)
    os.makedirs(stage)
    open(os.path.join(stage, STAGE_MARKER), 'w').close()


def copy_source(code_dir, stage, manifest):
    """src/ = the whole pipeline: src/*.py + tools/*.py flat, tools/*.sh as campaign scripts."""
    dst_src = os.path.join(stage, 'code', 'business_entity_resolution', 'src')
    os.makedirs(dst_src, exist_ok=True)
    n_py = n_sh = 0
    for sub in ('src', 'tools'):
        d = os.path.join(code_dir, sub)
        for fn in sorted(os.listdir(d)):
            full = os.path.join(d, fn)
            if not os.path.isfile(full) or fn.startswith('.'):
                continue
            if fn.endswith('.py'):
                place(full, os.path.join(dst_src, fn), link=False)
                n_py += 1
    wrappers = os.path.join(dst_src, 'campaign_scripts')
    for fn in sorted(os.listdir(os.path.join(code_dir, 'tools'))):
        if fn.endswith('.sh'):
            place(os.path.join(code_dir, 'tools', fn), os.path.join(wrappers, fn), link=False)
            n_sh += 1
    manifest['source'] = {'python_files': n_py, 'campaign_scripts': n_sh}
    log(f"src/: {n_py} python files + {n_sh} campaign scripts")


def copy_artifacts(root, code_dir, stage, manifest, score_parts=()):
    """Reference artefacts: shipped model, mined CE pairs, CE score cache, run reports."""
    art = os.path.join(stage, 'code', 'business_entity_resolution', 'artifacts')
    items = []
    srcs = [
        ('model_base41_500k/model_fast.pkl', f'{root}/output/model_base41_500k/model_fast.pkl'),
        ('model_base41_500k/train_report.json', f'{root}/output/model_base41_500k/train_report.json'),
        ('ce_train/ce_train.jsonl', f'{root}/cache/ce_train.jsonl'),
        ('ce_train/train_report.json', f'{root}/output/ce_model/train_report.json'),
        ('ce_train/tokenizer_config.json', f'{root}/output/ce_model/tokenizer_config.json'),
    ]
    for name, path in srcs:
        if os.path.exists(path):
            place(path, os.path.join(art, name))
            items.append(name)
        else:
            log(f"  !! missing artefact {path}")
    cache_dir = f'{root}/output/ce_apply/cache'
    if os.path.isdir(cache_dir):
        for fn in sorted(os.listdir(cache_dir)):
            if fn.endswith('.npz'):
                place(os.path.join(cache_dir, fn), os.path.join(art, 'ce_scores_cache', fn))
                items.append(f'ce_scores_cache/{fn}')
    reports = {
        'verify.txt': f'{root}/output/ce_apply/verify.txt',
        'threshold_report.txt': f'{root}/output/ce_apply/threshold_report.txt',
        'ship_table.txt': f'{root}/output/ce_apply/ship_table.txt',
        'ship_diff.txt': f'{root}/output/ce_apply/ship_diff.txt',
        'ce_holdout_eval.json': f'{root}/output/ce_holdout_eval.json',
        'check_outputs_final.txt': f'{root}/output/ce_apply/check_outputs.txt',
        'verify_package.txt': f'{root}/output/ce_apply/verify_package.txt',
    }
    for name, path in reports.items():
        if os.path.exists(path):
            place(path, os.path.join(art, 'reports', name))
            items.append(f'reports/{name}')
    for path in score_parts:
        path = os.path.abspath(path)
        # both predict shards have a file called scores.tsv - name them after their shard dir
        tag = os.path.basename(os.path.dirname(path)) or 'shard'
        name = f'gbdt_scores/{tag}.tsv'
        place(path, os.path.join(art, name))
        items.append(name)
    log(f"artifacts/: {len(items)} files")


def write_zip(stage, out, compress, manifest):
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    t0 = time.time()
    members = 0
    with zipfile.ZipFile(out, 'w', mode, compresslevel=6) as z:
        for root, dirs, files in os.walk(stage):
            dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and not d.startswith('.')]
            for fn in sorted(files):
                if fn in SKIP_FILES or fn.startswith('.'):
                    continue
                full = os.path.join(root, fn)
                if os.path.islink(full):
                    continue
                z.write(full, os.path.relpath(full, stage))
                members += 1
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=1))
    log(f"wrote {out}: {members + 1} members, "
        f"{os.path.getsize(out)/1e9:.2f} GB in {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', default='team')
    ap.add_argument('--root', default='../..', help='source tree root (student_resource)')
    ap.add_argument('--code-dir', default=None,
                    help='pipeline folder (default <root>/code/business_entity_resolution)')
    ap.add_argument('--stage', required=True, help='staging directory to build the tree in')
    ap.add_argument('--out', required=True)
    ap.add_argument('--matching', default=None, help='default <root>/output/matching_results.tsv')
    ap.add_argument('--cand-parts', nargs='+', default=None,
                    help='candidate_pairs.tsv halves; merged with one header')
    ap.add_argument('--candidate', default=None,
                    help='single candidate file (mutually exclusive with --cand-parts)')
    ap.add_argument('--score-parts', nargs='+', default=None,
                    help='GBDT score-matrix shards of the shipped run, bundled under '
                         'artifacts/gbdt_scores/ so the CE stage and the threshold rule can be '
                         'replayed without re-predicting the test split')
    ap.add_argument('--compress', action='store_true', help='deflate members')
    ap.add_argument('--note', default=None, help='provenance note stored in MANIFEST.json')
    ap.add_argument('--zip-only', action='store_true', help='reuse an existing staging tree')
    args = ap.parse_args()

    root = args.root
    code_dir = args.code_dir or os.path.join(root, 'code', 'business_entity_resolution')
    matching = args.matching or os.path.join(root, 'output', 'matching_results.tsv')
    cand_parts = args.cand_parts or [os.path.join(root, 'output', 'candidate_pairs.tsv')]
    if args.candidate:
        cand_parts = [args.candidate]

    manifest = {
        'team': args.team,
        'packaged_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': args.note,
        'python': sys.version.split()[0],
    }

    if not args.zip_only:
        reset_stage(args.stage)
        out_dir = os.path.join(args.stage, 'output')
        place(os.path.abspath(matching), os.path.join(out_dir, 'matching_results.tsv'))
        cand_out = os.path.join(out_dir, 'candidate_pairs.tsv')
        if len(cand_parts) > 1:
            rows = merge_tsvs([os.path.abspath(p) for p in cand_parts], cand_out)
            log(f"merged {len(cand_parts)} candidate halves: {rows:,} rows")
        else:
            place(os.path.abspath(cand_parts[0]), cand_out)
        copy_source(code_dir, args.stage, manifest)
        copy_artifacts(root, code_dir, args.stage, manifest, args.score_parts or ())
        dst_code = os.path.join(args.stage, 'code', 'business_entity_resolution')
        for dst, src in ((os.path.join(dst_code, 'README.md'),
                          os.path.join(code_dir, 'README.md')),
                         (os.path.join(dst_code, 'requirements.txt'),
                          os.path.join(code_dir, 'requirements.txt')),
                         (os.path.join(args.stage, 'Documentation_template.md'),
                          os.path.join(root, 'Documentation_template.md'))):
            if not os.path.exists(src):
                raise SystemExit(f"missing {src}")
            place(src, dst)
        log("README.md, requirements.txt, Documentation_template.md placed")

    outputs = {}
    for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
        p = os.path.join(args.stage, 'output', name)
        log(f"hashing {name} ({os.path.getsize(p)/1e9:.2f} GB)")
        outputs[name] = {'bytes': os.path.getsize(p), 'sha256': sha256(p),
                         'rows': sum(1 for _ in open(p, encoding='utf-8')) - 1}
    manifest['outputs'] = outputs
    for name, info in outputs.items():
        log(f"  {name}: {info['rows']:,} rows, {info['bytes']:,} B, {info['sha256'][:16]}…")
    code_root = os.path.join(args.stage, 'code', 'business_entity_resolution')
    src_dir = os.path.join(code_root, 'src')
    manifest['source'] = {
        'python_files': len([f for f in os.listdir(src_dir) if f.endswith('.py')]),
        'campaign_scripts': len([f for f in os.listdir(os.path.join(src_dir, 'campaign_scripts'))
                                 if f.endswith('.sh')]),
    }
    # hash everything a reviewer may want to check: artifacts/**, README, requirements
    hashes = {}
    for root_, _, files in os.walk(os.path.join(code_root, 'artifacts')):
        for fn in sorted(files):
            p = os.path.join(root_, fn)
            rel = os.path.relpath(p, code_root)
            hashes[rel] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    for rel in ('README.md', 'requirements.txt'):
        p = os.path.join(code_root, rel)
        hashes[rel] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    manifest['artifact_sha256'] = hashes
    manifest['artifacts'] = sorted(k for k in hashes if k.startswith('artifacts/'))
    ver = os.path.join(code_root, 'artifacts', 'reports', 'verify.txt')
    if os.path.exists(ver):
        manifest['validation'] = open(ver).read().strip()
    log(f"hashed {len(hashes)} files (src = {manifest['source']['python_files']} .py + "
        f"{manifest['source']['campaign_scripts']} .sh)")
    write_zip(args.stage, args.out, args.compress, manifest)
    log(f"done — staging tree: {args.stage}")


if __name__ == '__main__':
    main()
