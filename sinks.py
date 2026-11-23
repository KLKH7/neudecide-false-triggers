"""Which tools attract calls on audio with no command in it, and why?

    python sinks.py             # writes results/sinks.json

Uses the first 100 ESC-50 sound clips and 100 LibriSpeech clips from run.py's sample.
  alone      each tool offered on its own
  variants   the seven tools with `bringItem` renamed, re-described, or given no required argument
  quantised  the same clip as float samples and as 16-bit samples
"""
import copy
import json

import numpy as np
from neudecide import NeuDecide

from controls import PHONE, VENDOR
from run import ROOT, TOOLS, clips


def variant(**change):
    tools = copy.deepcopy(TOOLS)
    bring = next(t for t in tools if t['name'] == 'bringItem')
    if change.pop('optional', False):
        bring['parameters'].pop('required')
    bring.update(change)
    return tools


VARIANTS = {
    'original': TOOLS,
    'renamed_fetchItem': variant(name='fetchItem'),
    'renamed_orderTaxi': variant(name='orderTaxi'),
    'new_description': variant(description='Fetches one of the listed things from another room.'),
    'argument_optional': variant(optional=True),
    'bring_first': [TOOLS[5]] + TOOLS[:5] + TOOLS[6:],
}


def main():
    model = NeuDecide.from_pretrained()
    sample = {'sounds': [], 'speech': []}
    for meta, audio, rate in clips(300):
        if meta['set'] in sample and len(sample[meta['set']]) < 100:
            sample[meta['set']].append((audio, rate))
    out = {'alone': {}, 'variants': {}}
    for source, tools in [('home', TOOLS), ('vendor', VENDOR), ('phone', PHONE)]:
        for tool in tools:
            out['alone'][f"{source}/{tool['name']}"] = {
                s: sum(bool(model.generate(a, [tool], sample_rate=r)) for a, r in rows) for s, rows in sample.items()}
    for name, tools in VARIANTS.items():
        out['variants'][name] = {s: sum(bool(model.generate(a, tools, sample_rate=r)) for a, r in rows)
                                 for s, rows in sample.items()}
    flips = 0
    for a, r in sample['sounds'] + sample['speech']:
        as_int = np.round(np.clip(a, -1, 1) * 32767).astype(np.int16)
        flips += model.generate(a, TOOLS, sample_rate=r) != model.generate(as_int, TOOLS, sample_rate=r)
    out['quantised_changed_answer'] = [int(flips), 200]
    (ROOT / 'results' / 'sinks.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out['alone'].items():
        print('alone', k, v)
    for k, v in out['variants'].items():
        print('variant', k, v)
    print('quantised', out['quantised_changed_answer'])


if __name__ == '__main__':
    main()
