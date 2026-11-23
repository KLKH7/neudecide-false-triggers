"""Follow-up checks on results/calls.jsonl: invalid arguments, paired changes, and tool order.

    python checks.py            # writes results/checks.json
"""
import collections
import json
import random
from pathlib import Path

from run import GATE_NAMES, GATES, ROOT, SEED, TOOL_LISTS, TOOLS, acted, clips, share

rows = [json.loads(line) for line in open(ROOT / 'results' / 'calls.jsonl')]
SPEC = {t['name']: t['parameters'] for name in TOOL_LISTS for t in TOOL_LISTS[name]}
out = {}

# 1. Calls whose arguments break the tool's own schema.
invalid = {}
for name in TOOL_LISTS:
    for group, sets in [('commands', ['commands']), ('non_commands', ['speech', 'sounds', 'synthetic'])]:
        missing, outside, total, examples = 0, 0, 0, collections.Counter()
        for r in rows:
            if r['set'] not in sets:
                continue
            for call in r['calls'][name]:
                if call['name'] in GATE_NAMES:
                    continue
                total += 1
                spec = SPEC[call['name']]
                if any(k not in call['arguments'] for k in spec.get('required', [])):
                    missing += 1
                bad = [(k, v) for k, v in call['arguments'].items()
                       if v not in spec['properties'].get(k, {}).get('enum', [v])]
                outside += bool(bad)
                examples.update(f'{k}={v}' for k, v in bad)
        invalid[f'{name}/{group}'] = {'calls': total, 'missing_required': missing,
                                      'value_outside_enum': outside, 'examples': examples.most_common(6)}
out['invalid_arguments'] = invalid

# 2. Paired change when the gate is added (same clips): acted without -> with.
paired = {}
for gate in ['gate', 'gate_alt']:
    for s in ['commands', 'speech', 'sounds']:
        sub = [r for r in rows if r['set'] == s]
        stopped = sum(acted(r['calls']['plain']) and not acted(r['calls'][gate]) for r in sub)
        started = sum(not acted(r['calls']['plain']) and acted(r['calls'][gate]) for r in sub)
        paired[f'{gate}/{s}'] = {'stopped_acting': stopped, 'started_acting': started}
out['paired'] = paired

# 3. Tool order: same tools reversed, and the gate placed first instead of last.
from neudecide import NeuDecide
model = NeuDecide.from_pretrained()
orders = {'reversed': TOOLS[::-1], 'gate_first': [GATES['gate']] + TOOLS, 'gate_alt_first': [GATES['gate_alt']] + TOOLS}
order = {k: collections.defaultdict(list) for k in orders}
picks = {k: collections.Counter() for k in orders}
for meta, audio, rate in clips(300):
    for k, tools in orders.items():
        calls = model.generate(audio, tools, sample_rate=rate)
        order[k][meta['set']].append(acted(calls))
        if meta['set'] != 'commands':
            picks[k].update(c['name'] for c in calls)
        elif k == 'gate_first':
            order['gate_first']['commands_right'].append(bool(calls) and calls[0]['name'] == meta['gold'])
out['tool_order'] = {k: {s: share(v) for s, v in d.items()} for k, d in order.items()}
out['tool_order_false_call_tools'] = {k: v.most_common() for k, v in picks.items()}
(ROOT / 'results' / 'checks.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out))

# 4. A schema check after the model: drop any call missing a required argument or using a value outside its enum.
def valid(call):
    spec = SPEC[call['name']]
    return (call['name'] not in GATE_NAMES and all(k in call['arguments'] for k in spec.get('required', []))
            and all(v in spec['properties'].get(k, {}).get('enum', [v]) for k, v in call['arguments'].items()))

checked = {}
for name in TOOL_LISTS:
    for s in ['commands', 'speech', 'sounds', 'synthetic']:
        sub = [r for r in rows if r['set'] == s]
        entry = {'acted': share([any(valid(c) for c in r['calls'][name]) for r in sub])}
        if s == 'commands':
            entry['right_tool'] = share([bool(r['calls'][name]) and valid(r['calls'][name][0])
                                         and r['calls'][name][0]['name'] == r['gold'] for r in sub])
        checked[f'{name}/{s}'] = entry
out['with_schema_check'] = checked
(ROOT / 'results' / 'checks.json').write_text(json.dumps(out, indent=1) + '\n')
for k, v in checked.items():
    print(k, {m: (x['count'], x['of'], x['interval_95']) for m, x in v.items()})
