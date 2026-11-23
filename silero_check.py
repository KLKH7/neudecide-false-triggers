"""Redo the voice-activity check with Silero VAD, and test whether the meeting result is about loudness.

    python silero_check.py      # writes results/silero.json

Silero's ONNX model is read directly: 512-sample windows at 16 kHz with 64 samples of context.
A clip "holds speech" if at least 0.3 s of windows score above 0.5.
"""
import json

import numpy as np
import onnxruntime as ort
from neudecide import NeuDecide
from neudecide.audio import prepare_audio

from run import DATA, GATES, ROOT, TOOLS, acted, clips, share
from vad_conv import meeting_clips

session = ort.InferenceSession(str(DATA / 'silero' / 'silero_vad.onnx'))


def speech_seconds(audio, rate):
    x = prepare_audio(audio, rate, 16000).astype(np.float32)
    state, context, voiced = np.zeros((2, 1, 128), np.float32), np.zeros(64, np.float32), 0
    for i in range(0, len(x) - 511, 512):
        chunk = x[i:i + 512]
        prob, state = session.run(None, {'input': np.concatenate([context, chunk])[None, :], 'state': state,
                                         'sr': np.array(16000, np.int64)})
        context = chunk[-64:]
        voiced += int(prob[0, 0] > 0.5)
    return float(voiced * 0.032)


def main():
    calls = [json.loads(line) for line in open(ROOT / 'results' / 'calls.jsonl')]
    speechy = [speech_seconds(a, r) >= 0.3 for _, a, r in clips(300)]
    out = {'vad': {}}
    for s in ['commands', 'speech', 'sounds', 'synthetic']:
        idx = [i for i, r in enumerate(calls) if r['set'] == s]
        entry = {'vad_says_speech': share([speechy[i] for i in idx])}
        for tools in ['plain', 'gate']:
            entry[f'acted_after_vad/{tools}'] = share([speechy[i] and acted(calls[i]['calls'][tools]) for i in idx])
        if s == 'commands':
            entry['right_tool_after_vad/plain'] = share(
                [speechy[i] and bool(calls[i]['calls']['plain']) and calls[i]['calls']['plain'][0]['name'] == calls[i]['gold'] for i in idx])
        if s == 'sounds':
            kinds = {}
            for i in idx:
                if speechy[i]:
                    kinds[calls[i]['kind']] = kinds.get(calls[i]['kind'], 0) + 1
            entry['kinds_passed'] = sorted(kinds.items(), key=lambda kv: -kv[1])[:8]
        out['vad'][s] = entry

    model = NeuDecide.from_pretrained()
    no_bring = [t for t in TOOLS if t['name'] != 'bringItem']
    passes, peaks = [], []
    tally = {k: [] for k in ['home', 'home_gate', 'home_without_bringItem']}
    loud = {k: [] for k in tally}
    lists = {'home': TOOLS, 'home_gate': TOOLS + [GATES['gate']], 'home_without_bringItem': no_bring}
    for row, x, rate in meeting_clips(300):
        passes.append(speech_seconds(x, rate) >= 0.3)
        peaks.append(float(np.abs(x).max()))
        y = x / max(np.abs(x).max(), 1e-9) * 0.9
        for k, tools in lists.items():
            tally[k].append(acted(model.generate(x, tools, sample_rate=rate)))
            loud[k].append(acted(model.generate(y, tools, sample_rate=rate)))
    out['conversation'] = {
        'vad_says_speech': share(passes), 'median_peak_level': round(float(np.median(peaks)), 4),
        'acted': {k: share(v) for k, v in tally.items()},
        'acted_peak_normalised': {k: share(v) for k, v in loud.items()},
        'acted_after_vad': {k: share([p and a for p, a in zip(passes, v)]) for k, v in tally.items()}}
    (ROOT / 'results' / 'silero.json').write_text(json.dumps(out, indent=1) + '\n')
    for s, e in out['vad'].items():
        print(s, {k: ((v['count'], v['of']) if isinstance(v, dict) else v) for k, v in e.items()})
    c = out['conversation']
    print('conv', c['vad_says_speech']['count'], c['median_peak_level'])
    for k in ['acted', 'acted_peak_normalised', 'acted_after_vad']:
        print(k, {n: v['count'] for n, v in c[k].items()})


if __name__ == '__main__':
    main()
