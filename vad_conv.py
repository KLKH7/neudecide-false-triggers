"""Two things the first pass left out: a voice-activity detector in front, and conversational speech.

    python vad_conv.py          # writes results/vad_conv.json

  vad            WebRTC VAD (mode 3, the strictest) decides whether a clip holds speech: at least 0.3 s of voiced
                 30 ms frames. Combined with the answers already in results/calls.jsonl.
  conversation   300 utterances of 1.5 to 6 s from AMI meetings, recorded on a single distant microphone.
"""
import io
import json
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import webrtcvad
from neudecide import NeuDecide
from neudecide.audio import prepare_audio

from controls import PHONE, VENDOR
from run import DATA, GATES, ROOT, SEED, TOOLS, acted, clips, share

vad = webrtcvad.Vad(3)


def has_speech(audio, rate):
    x = prepare_audio(audio, rate, 16000)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    step = 480 * 2
    voiced = sum(vad.is_speech(pcm[i:i + step], 16000) for i in range(0, len(pcm) - step + 1, step))
    return voiced * 0.03 >= 0.3


def meeting_clips(n):
    table = pq.read_table(DATA / 'ami' / 'sdm' / 'validation-00004-of-00005.parquet')
    rows = table.select(['audio_id', 'text', 'begin_time', 'end_time']).to_pylist()
    keep = [i for i, r in enumerate(rows) if 1.5 <= r['end_time'] - r['begin_time'] <= 6]
    audio = table.column('audio')
    for i in sorted(random.Random(SEED).sample(keep, n)):
        x, rate = sf.read(io.BytesIO(audio[i].as_py()['bytes']), dtype='float32')
        yield rows[i], x, rate


def main():  # run with: python vad_conv.py
    out = {}
    calls = [json.loads(line) for line in open(ROOT / 'results' / 'calls.jsonl')]
    speechy = [has_speech(audio, rate) for _, audio, rate in clips(300)]
    table = {}
    for s in ['commands', 'speech', 'sounds', 'synthetic']:
        idx = [i for i, r in enumerate(calls) if r['set'] == s]
        entry = {'vad_says_speech': share([speechy[i] for i in idx])}
        for tools in ['plain', 'gate']:
            entry[f'acted_after_vad/{tools}'] = share([speechy[i] and acted(calls[i]['calls'][tools]) for i in idx])
        if s == 'commands':
            for tools in ['plain', 'gate']:
                entry[f'right_tool_after_vad/{tools}'] = share(
                    [speechy[i] and bool(calls[i]['calls'][tools]) and calls[i]['calls'][tools][0]['name'] == calls[i]['gold'] for i in idx])
        table[s] = entry
    out['vad'] = table

    model = NeuDecide.from_pretrained()
    lists = {'home': TOOLS, 'home_gate': TOOLS + [GATES['gate']],
             'home_without_bringItem': [t for t in TOOLS if t['name'] != 'bringItem'],
             'vendor': VENDOR, 'vendor_gate': VENDOR + [GATES['gate']], 'phone': PHONE, 'phone_gate': PHONE + [GATES['gate']]}
    tally = {k: [] for k in lists}
    passes, examples = [], []
    for row, x, rate in meeting_clips(300):
        passes.append(has_speech(x, rate))
        for k, tools in lists.items():
            got = model.generate(x, tools, sample_rate=rate)
            tally[k].append(acted(got))
            if k == 'home' and got and len(examples) < 12:
                examples.append([row['text'], got[0]])
    out['conversation'] = {'vad_says_speech': share(passes), 'acted': {k: share(v) for k, v in tally.items()},
                           'examples_home': examples}
    (ROOT / 'results' / 'vad_conv.json').write_text(json.dumps(out, indent=1) + '\n')
    for s, e in out['vad'].items():
        print(s, {k: (v['count'], v['of']) for k, v in e.items()})
    print('conv vad', out['conversation']['vad_says_speech']['count'])
    print({k: (v['count'], v['interval_95']) for k, v in out['conversation']['acted'].items()})
    for e in examples:
        print(e)


if __name__ == '__main__':
    main()
