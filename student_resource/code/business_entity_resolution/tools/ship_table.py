#!/usr/bin/env python3
"""Per-country shape of submission files: match rate + ids per matched row.

Quick sanity on a ship file before upload: the champion's LB-validated rule
(France 0.97 / US+India 0.94) produced France mr ~3.4x ids per matched row on
the test set; a variant whose France shape is far off deserved a second look.

  python3 tools/ship_table.py ../../output/ship500k_fr097.tsv ../../output/ship1m_fr097.tsv
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# tools/ -> business_entity_resolution -> code -> student_resource
S1 = os.path.join(HERE, '..', '..', '..', 'dataset', 'test', 'test_source1.tsv')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    s1 = pd.read_csv(S1, sep='\t', usecols=['entity_id', 'country'],
                     dtype=str, keep_default_na=False, na_filter=False)
    for path in sys.argv[1:]:
        try:
            m = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                            na_filter=False)
        except FileNotFoundError:
            print(f"{path}: MISSING")
            continue
        df = s1.merge(pd.DataFrame({'entity_id': m['source1_entity_id'],
                                    'ids': m['matched_entity_ids']}), on='entity_id')
        n = df['ids'].map(lambda s: 0 if s == '' else s.count(',') + 1)
        df = df.assign(n=n)
        g = df.groupby('country').agg(rows=('n', 'size'),
                                      matched=('n', lambda s: int((s > 0).sum())),
                                      ids=('n', 'sum'))
        g['mr'] = (g.matched / g.rows).round(3)
        g['ipr'] = (g.ids / g.matched.clip(lower=1)).round(2)
        body = ' | '.join(f"{c}: mr={r.mr} ipr={r.ipr}" for c, r in g.iterrows())
        print(f"{os.path.basename(path)}: {body}  "
              f"(total ids {int(g.ids.sum()):,}, rows {len(df):,})", flush=True)


if __name__ == '__main__':
    main()
