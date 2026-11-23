"""Is it one tool, or the number of tools? Swap and drop tools on 100 sound clips.

    python swap.py              # writes results/swap.json
"""
import collections
import copy
import json

from neudecide import NeuDecide

from controls import PHONE
from run import ROOT, TOOLS, clips
from sinks import VARIANTS

bring = next(t for t in TOOLS if t['name'] == 'bringItem')
rest = [t for t in TOOLS if t['name'] != 'bringItem']
no_enum = copy.deepcopy(bring)
no_enum['parameters']['properties']['item'] = {'type': 'string'}
LISTS = {
    'seven': TOOLS,
    'without_bringItem': rest,
    'without_setLamp': [t for t in TOOLS if t['name'] != 'setLamp'],
    'without_setLanguage': [t for t in TOOLS if t['name'] != 'setLanguage'],
    'bringItem_swapped_for_takePhoto': rest + [PHONE[4]],
    'bringItem_swapped_for_openApp': rest + [PHONE[6]],
    'bringItem_free_text_item': rest + [no_enum],
    'renamed_orderTaxi': VARIANTS['renamed_orderTaxi'],
    'new_description': VARIANTS['new_description'],
}
model = NeuDecide.from_pretrained()
sounds = []
for meta, audio, rate in clips(300):
    if meta['set'] == 'sounds' and len(sounds) < 100:
        sounds.append((audio, rate))
out = {}
for name, tools in LISTS.items():
    picked = collections.Counter()
    fired = 0
    for a, r in sounds:
        calls = model.generate(a, tools, sample_rate=r)
        fired += bool(calls)
        picked.update(c['name'] for c in calls)
    out[name] = {'fired_of_100': fired, 'tools': picked.most_common(3)}
    print(name, out[name])
(ROOT / 'results' / 'swap.json').write_text(json.dumps(out, indent=1) + '\n')
