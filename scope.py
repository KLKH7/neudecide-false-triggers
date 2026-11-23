"""If a wake word sits in front, the model only hears requests. What does it do with a request no tool covers?

    python scope.py             # writes results/scope.json

  scope   the 200 spoken commands against 20 random tool lists. A command is "in scope" for a list that holds
          its tool and "out of scope" otherwise. Run with no gate and with the `doNothing` gate.
  tv      a second source of speech with no command: a late-night TV monologue cut into 4-second windows.
"""
import io
import json

import pyarrow.parquet as pq
import soundfile as sf
from neudecide import NeuDecide

from do_nothing import BY_NAME, gate
from run import DATA, ROOT, TOOLS, clips, share
from silero_check import speech_seconds


def tv_clips(n):
    table = pq.read_table(next((DATA / 'meanwhile').rglob('*.parquet')), columns=['audio'])
    out = []
    for cell in table.column('audio'):
        x, rate = sf.read(io.BytesIO(cell.as_py()['bytes']), dtype='float32')
        if x.ndim == 2:
            x = x.mean(axis=1)
        for start in range(0, len(x) - 4 * rate + 1, 6 * rate):  # one 4 s window every 6 s
            out.append((x[start:start + 4 * rate], rate))
    step = max(1, len(out) // n)
    return out[::step][:n]


def main():
    model = NeuDecide.from_pretrained()
    commands = [(m, a, r) for m, a, r in clips(300) if m['set'] == 'commands']
    lists = json.loads((ROOT / 'results' / 'random_lists.json').read_text())['lists'][20:40]
    tally = {k: [] for k in ['in/right', 'in/right_gate', 'in/gated', 'out/fired', 'out/fired_gate']}
    per_list = []
    for item in lists:
        tools = [BY_NAME[n] for n in item['tools']]
        gated = tools + [gate('doNothing')]
        row = {'tools': item['tools'], 'out': 0, 'out_fired': 0, 'out_fired_gate': 0}
        for meta, audio, rate in commands:
            plain = model.generate(audio, tools, sample_rate=rate)
            with_gate = model.generate(audio, gated, sample_rate=rate)
            real = [c for c in with_gate if c['name'] != 'doNothing']
            if meta['gold'] in item['tools']:
                tally['in/right'].append(bool(plain) and plain[0]['name'] == meta['gold'])
                tally['in/right_gate'].append(bool(with_gate) and with_gate[0]['name'] == meta['gold'])
                tally['in/gated'].append(bool(with_gate) and with_gate[0]['name'] == 'doNothing')
            else:
                tally['out/fired'].append(bool(plain))
                tally['out/fired_gate'].append(bool(real))
                row['out'] += 1
                row['out_fired'] += bool(plain)
                row['out_fired_gate'] += bool(real)
        per_list.append(row)
        print(row, flush=True)
    out = {'scope': {k: share(v) for k, v in tally.items()}, 'scope_lists': per_list}

    tv = tv_clips(300)
    passes = [speech_seconds(a, r) >= 0.3 for a, r in tv]
    home_gate = TOOLS + [gate('doNothing')]
    plain = [bool(model.generate(a, TOOLS, sample_rate=r)) for a, r in tv]
    gated = [any(c['name'] != 'doNothing' for c in model.generate(a, home_gate, sample_rate=r)) for a, r in tv]
    out['tv'] = {'clips': len(tv), 'vad_says_speech': share(passes), 'fired': share(plain), 'fired_doNothing': share(gated),
                 'fired_doNothing_and_vad': share([p and g for p, g in zip(passes, gated)])}
    (ROOT / 'results' / 'scope.json').write_text(json.dumps(out, indent=1) + '\n')
    print('SUMMARY', json.dumps({'scope': {k: (v['count'], v['of'], v['interval_95']) for k, v in out['scope'].items()},
                                 'tv': {k: ((v['count'], v['of']) if isinstance(v, dict) else v) for k, v in out['tv'].items()}}))


if __name__ == '__main__':
    main()
