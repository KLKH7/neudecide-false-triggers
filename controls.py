"""Controls that could explain the headline result away.

    python controls.py          # writes results/controls.json

  same_input    a WAV file path and the array give the same answer (rules out my audio handling)
  unconstrained the answer without the output grammar (does the grammar force calls?)
  vendor        Neuphonic's own example tools, with and without the gate
  other_domain  a second, unrelated tool list (phone assistant), with and without the gate
  sink          a tool with no parameters that is NOT a gate (`getBattery`): does it absorb calls like the gate does?
  no_bring      the seven tools without `bringItem`
  short_speech  speech cut to its first 3 seconds
"""
import collections
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from neudecide import NeuDecide

from run import GATES, ROOT, TOOLS, acted, clips, share

VENDOR = json.loads((ROOT / 'upstream' / 'examples' / 'tools.json').read_text())
BATTERY = {'name': 'getBattery', 'description': "Reports the device's battery level.",
           'parameters': {'type': 'object', 'properties': {}}}
PHONE = [
    {'name': 'setTimer', 'description': 'Starts a countdown timer.',
     'parameters': {'type': 'object', 'properties': {'minutes': {'type': 'integer'}}, 'required': ['minutes']}},
    {'name': 'getWeather', 'description': 'Reads out the weather forecast for a city.',
     'parameters': {'type': 'object', 'properties': {'city': {'type': 'string', 'enum': ['London', 'New York', 'Paris', 'Tokyo']}}, 'required': ['city']}},
    {'name': 'callContact', 'description': 'Phones one of the saved contacts.',
     'parameters': {'type': 'object', 'properties': {'contact': {'type': 'string', 'enum': ['mum', 'dad', 'work', 'doctor']}}, 'required': ['contact']}},
    {'name': 'setAlarm', 'description': 'Sets a wake-up alarm.',
     'parameters': {'type': 'object', 'properties': {'hour': {'type': 'integer'}}, 'required': ['hour']}},
    {'name': 'takePhoto', 'description': 'Takes a photo with the camera.',
     'parameters': {'type': 'object', 'properties': {'camera': {'type': 'string', 'enum': ['front', 'back']}}, 'required': ['camera']}},
    {'name': 'setBrightness', 'description': 'Changes the screen brightness.',
     'parameters': {'type': 'object', 'properties': {'direction': {'type': 'string', 'enum': ['up', 'down']}}, 'required': ['direction']}},
    {'name': 'openApp', 'description': 'Opens an app.',
     'parameters': {'type': 'object', 'properties': {'app': {'type': 'string', 'enum': ['maps', 'mail', 'calendar', 'notes']}}, 'required': ['app']}},
]
LISTS = {
    'vendor': VENDOR, 'vendor_gate': VENDOR + [GATES['gate']],
    'other_domain': PHONE, 'other_domain_gate': PHONE + [GATES['gate']],
    'sink_last': TOOLS + [BATTERY], 'sink_first': [BATTERY] + TOOLS,
    'no_bring': [t for t in TOOLS if t['name'] != 'bringItem'],
}
GATE = {'noAction'}


def did(calls):
    return any(c['name'] not in GATE for c in calls)


def main():
    model = NeuDecide.from_pretrained()
    tally = {k: collections.defaultdict(list) for k in LISTS}
    names = {k: collections.Counter() for k in LISTS}
    extra = collections.defaultdict(list)
    same, text = [], collections.Counter()
    counts = collections.Counter()
    tmp = Path(tempfile.mkdtemp())
    for meta, audio, rate in clips(300):
        s = meta['set']
        counts[s] += 1
        for k, tools in LISTS.items():
            calls = model.generate(audio, tools, sample_rate=rate)
            tally[k][s].append(did(calls))
            if s != 'commands':
                names[k].update(c['name'] for c in calls)
            elif k.startswith('sink') or k == 'no_bring':
                extra[k + '/right_tool'].append(bool(calls) and calls[0]['name'] == meta['gold'])
        if counts[s] <= 25:  # same answer from a file path as from the array?
            path = tmp / 'clip.wav'
            sf.write(path, audio, rate, subtype='PCM_16')
            same.append(model.generate(str(path), TOOLS) == model.generate(audio, TOOLS, sample_rate=rate))
        if s in ('sounds', 'synthetic') and counts[s] <= 100:
            raw = model.generate_text(audio, TOOLS, sample_rate=rate, constrained=False)
            extra['unconstrained_acted/' + s].append(raw.strip() not in ('[]', ''))
            text[raw[:60]] += 1
        if s == 'speech':
            short = audio[: 3 * rate]
            extra['short_speech/plain'].append(acted(model.generate(short, TOOLS, sample_rate=rate)))
            extra['short_speech/gate'].append(acted(model.generate(short, TOOLS + [GATES['gate']], sample_rate=rate)))
    out = {'same_input': share(same),
           'lists': {k: {s: share(v) for s, v in d.items()} for k, d in tally.items()},
           'false_call_tools': {k: v.most_common(5) for k, v in names.items()},
           'extra': {k: share(v) for k, v in extra.items()},
           'unconstrained_examples': text.most_common(6)}
    (ROOT / 'results' / 'controls.json').write_text(json.dumps(out, indent=1) + '\n')
    print('same', out['same_input']['count'], out['same_input']['of'])
    for k, d in out['lists'].items():
        print(k, {s: (x['count'], x['of']) for s, x in d.items()}, out['false_call_tools'][k])
    for k, x in out['extra'].items():
        print(k, x['count'], x['of'])
    print(out['unconstrained_examples'])


if __name__ == '__main__':
    main()
