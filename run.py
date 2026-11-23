"""How often does NeuDecide call a tool when nobody gave a command, and does a "no action" tool stop it?

    python run.py --dry-run     # load and count the audio only; no model needed
    python run.py               # run every clip under every tool list, write results/

Audio (all public, fetched by fetch.sh into data/):
  commands   200 Fluent Speech Commands test clips (smart-home commands, 73 speakers)
  speech     300 LibriSpeech dev-clean-2 clips (read audiobook speech: no command in it)
  sounds     300 ESC-50 clips (dogs, rain, engines, keyboards: no speech)
  synthetic  silence, white noise and pink noise at three levels

Tool lists:
  plain      seven smart-home tools
  gate       the same plus a `noAction` tool, as the NeuDecide README suggests
  gate_alt   the same gate under another name and description, to see if wording matters

The model decodes greedily, so one pass per clip is the whole distribution.
"""
import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SEED = 20261123
ROOMS = ['kitchen', 'bedroom', 'washroom']

TOOLS = [
    {'name': 'setLights', 'description': 'Turns the ceiling lights on or off, in one room or everywhere.',
     'parameters': {'type': 'object', 'properties': {
         'state': {'type': 'string', 'enum': ['on', 'off']},
         'room': {'type': 'string', 'enum': ROOMS}}, 'required': ['state']}},
    {'name': 'setLamp', 'description': 'Turns the lamp on or off.',
     'parameters': {'type': 'object', 'properties': {
         'state': {'type': 'string', 'enum': ['on', 'off']}}, 'required': ['state']}},
    {'name': 'setMusic', 'description': 'Plays or pauses the music.',
     'parameters': {'type': 'object', 'properties': {
         'state': {'type': 'string', 'enum': ['play', 'pause']}}, 'required': ['state']}},
    {'name': 'changeHeat', 'description': 'Turns the heating up or down, in one room or everywhere.',
     'parameters': {'type': 'object', 'properties': {
         'direction': {'type': 'string', 'enum': ['up', 'down']},
         'room': {'type': 'string', 'enum': ROOMS}}, 'required': ['direction']}},
    {'name': 'changeVolume', 'description': 'Turns the sound volume up or down.',
     'parameters': {'type': 'object', 'properties': {
         'direction': {'type': 'string', 'enum': ['up', 'down']}}, 'required': ['direction']}},
    {'name': 'bringItem', 'description': 'Asks the robot to fetch an item.',
     'parameters': {'type': 'object', 'properties': {
         'item': {'type': 'string', 'enum': ['shoes', 'socks', 'newspaper', 'juice']}}, 'required': ['item']}},
    {'name': 'setLanguage', 'description': 'Changes the device language.',
     'parameters': {'type': 'object', 'properties': {
         'language': {'type': 'string', 'enum': ['English', 'German', 'Korean', 'Chinese']}}}},
]
GATES = {
    'gate': {'name': 'noAction',
             'description': 'Use when the audio is not a command for this device: conversation, background noise or silence.',
             'parameters': {'type': 'object', 'properties': {}}},
    'gate_alt': {'name': 'ignore',
                 'description': 'Does nothing. Choose this if nobody asked the device to do anything.',
                 'parameters': {'type': 'object', 'properties': {}}},
}
TOOL_LISTS = {'plain': TOOLS, 'gate': TOOLS + [GATES['gate']], 'gate_alt': TOOLS + [GATES['gate_alt']]}
GATE_NAMES = {g['name'] for g in GATES.values()}


def gold_tool(text):
    """The tool a Fluent Speech Commands sentence asks for, from its wording."""
    t = text.lower()
    if any(w in t for w in ['bring', 'fetch', 'get me', 'go get']):
        return 'bringItem'
    if 'language' in t:
        return 'setLanguage'
    if 'lamp' in t:
        return 'setLamp'
    if 'light' in t:
        return 'setLights'
    if any(w in t for w in ['heat', 'temperature', 'hotter', 'cooler']):
        return 'changeHeat'
    if any(w in t for w in ['volume', 'loud', 'quiet', 'sound', 'softer', 'turn it down']):
        return 'changeVolume'
    if any(w in t for w in ['music', 'resume', 'stop', 'pause']):
        return 'setMusic'
    raise ValueError(f'no tool for: {text}')


def decode(blob):
    audio, rate = sf.read(io.BytesIO(blob), dtype='float32', always_2d=False)
    return audio, rate


def commands():
    table = pq.read_table(next((DATA / 'IntentClassification_FluentSpeechCommands-Action').rglob('*.parquet')),
                          columns=['file', 'transcription', 'audio'])
    for row in table.to_pylist():
        audio, rate = decode(row['audio']['bytes'])
        yield {'set': 'commands', 'id': row['file'], 'text': row['transcription'],
               'gold': gold_tool(row['transcription'])}, audio, rate


def speech(n, rng):
    files = sorted((DATA / 'LibriSpeech').rglob('*.flac'))
    for path in rng.sample(files, n):
        audio, rate = sf.read(path, dtype='float32')
        yield {'set': 'speech', 'id': path.stem}, audio, rate


def sounds(n, rng):
    table = pq.read_table(next((DATA / 'esc50').rglob('*.parquet')))
    rows = table.select(['filename', 'category']).to_pylist()
    picks = sorted(rng.sample(range(len(rows)), n))
    audio_col = table.column('audio')
    for i in picks:
        audio, rate = decode(audio_col[i].as_py()['bytes'])
        yield {'set': 'sounds', 'id': rows[i]['filename'], 'kind': rows[i]['category']}, audio, rate


def synthetic(rng):
    rate = 16000
    gen = np.random.default_rng(SEED)
    for seconds in [2, 5, 10]:
        n = rate * seconds
        yield {'set': 'synthetic', 'id': f'silence_{seconds}s', 'kind': 'silence'}, np.zeros(n, 'float32'), rate
        for level in [0.003, 0.03, 0.3]:
            white = gen.standard_normal(n).astype('float32')
            spectrum = np.fft.rfft(white)
            spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
            pink = np.fft.irfft(spectrum, n).astype('float32')
            for kind, x in [('white', white), ('pink', pink)]:
                x = x / np.abs(x).max() * level
                yield {'set': 'synthetic', 'id': f'{kind}_{level}_{seconds}s', 'kind': kind}, x, rate


def clips(n):
    rng = random.Random(SEED)
    yield from commands()
    yield from speech(n, rng)
    yield from sounds(n, rng)
    yield from synthetic(rng)


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def share(flags):
    k, n = sum(flags), len(flags)
    return {'count': k, 'of': n, 'share': round(k / n, 4) if n else None, 'interval_95': wilson(k, n)}


def acted(calls):
    """True when the model asked the device to do something: any call other than the gate tool."""
    return any(c['name'] not in GATE_NAMES for c in calls)


def summarise(rows):
    out = {}
    for tools in TOOL_LISTS:
        out[tools] = {}
        for name in ['commands', 'speech', 'sounds', 'synthetic']:
            sub = [r for r in rows if r['set'] == name]
            calls = [r['calls'][tools] for r in sub]
            entry = {'acted': share([acted(c) for c in calls]),
                     'chose_gate': share([any(x['name'] in GATE_NAMES for x in c) for c in calls]),
                     'empty': share([not c for c in calls])}
            if name == 'commands':
                entry['right_tool'] = share([bool(c) and c[0]['name'] == r['gold'] for r, c in zip(sub, calls)])
            out[tools][name] = entry
    kinds = {}
    for r in rows:
        if r['set'] == 'sounds':
            kinds.setdefault(r['kind'], []).append(acted(r['calls']['plain']))
    out['sounds_by_kind_plain'] = {k: [sum(v), len(v)] for k, v in sorted(kinds.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--n', type=int, default=300, help='clips of read speech and of sounds')
    args = parser.parse_args()
    if args.dry_run:
        counts, seconds = {}, {}
        for meta, audio, rate in clips(args.n):
            counts[meta['set']] = counts.get(meta['set'], 0) + 1
            seconds[meta['set']] = seconds.get(meta['set'], 0) + len(audio) / rate
        print(counts, {k: round(v / counts[k], 1) for k, v in seconds.items()})
        return
    from neudecide import NeuDecide
    model = NeuDecide.from_pretrained()
    (ROOT / 'results').mkdir(exist_ok=True)
    rows, start = [], time.time()
    with open(ROOT / 'results' / 'calls.jsonl', 'w') as f:
        for meta, audio, rate in clips(args.n):
            meta['seconds'] = round(len(audio) / rate, 2)
            meta['calls'] = {name: model.generate(audio, tools, sample_rate=rate) for name, tools in TOOL_LISTS.items()}
            rows.append(meta)
            f.write(json.dumps(meta) + '\n')
            if len(rows) % 100 == 0:
                print(len(rows), 'clips', round(time.time() - start), 's', flush=True)
    summary = summarise(rows)
    (ROOT / 'results' / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
