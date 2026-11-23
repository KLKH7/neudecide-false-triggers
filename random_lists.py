"""How unstable is the false-trigger rate across tool lists? Forty random lists from a pool of 17 tools.

    python random_lists.py      # writes results/random_lists.json

Each list is 3 to 8 tools drawn at random, in random order. Each is run on the same 50 sound
clips and 50 speech clips, with and without a `noAction` tool placed last.
"""
import json
import random
import statistics

from neudecide import NeuDecide

from controls import PHONE, VENDOR
from run import GATES, ROOT, SEED, TOOLS, clips

POOL = TOOLS + PHONE + VENDOR
rng = random.Random(SEED)
model = NeuDecide.from_pretrained()
sample = {'sounds': [], 'speech': []}
for meta, audio, rate in clips(300):
    if meta['set'] in sample and len(sample[meta['set']]) < 50:
        sample[meta['set']].append((audio, rate))


def fired(tools, rows):
    return sum(any(c['name'] != 'noAction' for c in model.generate(a, tools, sample_rate=r)) for a, r in rows)


lists = []
for i in range(40):
    tools = rng.sample(POOL, rng.randint(3, 8))
    row = {'tools': [t['name'] for t in tools]}
    for s, rows in sample.items():
        row[s] = fired(tools, rows)
        row[s + '_gate'] = fired(tools + [GATES['gate']], rows)
    lists.append(row)
    print(i, row, flush=True)


def spread(key):
    v = sorted(r[key] * 2 for r in lists)
    return {'min': v[0], 'quartile_1': v[10], 'median': statistics.median(v), 'quartile_3': v[30], 'max': v[-1],
            'lists_at_most_10pct': sum(x <= 10 for x in v), 'lists_at_least_90pct': sum(x >= 90 for x in v)}


presence = {}
for tool in POOL:
    with_, without = ([r['sounds'] * 2 for r in lists if (tool['name'] in r['tools']) == flag] for flag in (True, False))
    presence[tool['name']] = {'lists_with': len(with_), 'mean_pct_with': round(statistics.mean(with_), 1) if with_ else None,
                              'mean_pct_without': round(statistics.mean(without), 1)}
out = {'percent_fired': {k: spread(k) for k in ['sounds', 'sounds_gate', 'speech', 'speech_gate']},
       'gate_fixed_sounds': {'lists_at_least_50pct_without_gate': sum(r['sounds'] >= 25 for r in lists),
                             'of_those_at_most_10pct_with_gate': sum(r['sounds'] >= 25 and r['sounds_gate'] <= 5 for r in lists),
                             'of_those_still_at_least_50pct': sum(r['sounds'] >= 25 and r['sounds_gate'] >= 25 for r in lists)},
       'sounds_by_tool_present': presence, 'lists': lists}
(ROOT / 'results' / 'random_lists.json').write_text(json.dumps(out, indent=1) + '\n')
print('SUMMARY', json.dumps({k: out[k] for k in ['percent_fired', 'gate_fixed_sounds', 'sounds_by_tool_present']}))
