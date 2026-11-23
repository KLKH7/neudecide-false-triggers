"""Is the gate tool's failure about my wording? Five wordings on 12 random tool lists.

    python wordings.py          # writes results/wordings.json

Each list is run on 40 sound clips and 40 meeting utterances, with each gate wording placed last.
"""
import json
import random
import statistics

from neudecide import NeuDecide

from controls import PHONE, VENDOR
from run import ROOT, SEED, TOOLS, clips
from vad_conv import meeting_clips

EMPTY = {'type': 'object', 'properties': {}}
WORDINGS = {
    'noAction': 'Use when the audio is not a command for this device: conversation, background noise or silence.',
    'no_action': 'No action. Call this when no other tool applies.',
    'none': 'None of the other tools apply.',
    'notACommand': 'The speaker is not giving this device a command.',
    'doNothing': 'Do nothing.',
}


def main():
    rng = random.Random(SEED + 1)
    model = NeuDecide.from_pretrained()
    sample = {'sounds': [], 'meeting': [(x, r) for _, x, r in meeting_clips(300)][:40]}
    for meta, audio, rate in clips(300):
        if meta['set'] == 'sounds' and len(sample['sounds']) < 40:
            sample['sounds'].append((audio, rate))


    def fired(tools, rows, gate=None):
        return sum(any(c['name'] != gate for c in model.generate(a, tools, sample_rate=r)) for a, r in rows)


    rows = []
    for i in range(12):
        tools = rng.sample(TOOLS + PHONE + VENDOR, rng.randint(3, 8))
        row = {'tools': [t['name'] for t in tools]}
        for s, clips_ in sample.items():
            row[s] = {'none_added': fired(tools, clips_)}
            for name, text in WORDINGS.items():
                row[s][name] = fired(tools + [{'name': name, 'description': text, 'parameters': EMPTY}], clips_, name)
        rows.append(row)
        print(i, row, flush=True)
    out = {'of': 40, 'median_fired': {s: {k: statistics.median(r[s][k] for r in rows) for k in rows[0][s]} for s in sample},
           'lists_cut_to_10pct_or_less': {s: {k: sum(r[s][k] <= 4 for r in rows) for k in rows[0][s]} for s in sample},
           'lists': rows}
    (ROOT / 'results' / 'wordings.json').write_text(json.dumps(out, indent=1) + '\n')
    print('SUMMARY', json.dumps({k: out[k] for k in ['median_fired', 'lists_cut_to_10pct_or_less']}))


if __name__ == '__main__':
    main()
