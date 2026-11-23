"""Two questions about the meeting result, and one possible fix.

    python confidence.py        # writes results/confidence.json

  close_mic    the same meeting utterances from the speaker's headset microphone: is it the distance, or the talk?
  wording      meeting utterances with no word a smart-home tool could act on
  confidence   NeuDecide has no confidence score, but the decoder's own token probabilities are available.
               Does a threshold on them separate real commands from false calls? The score and threshold are
               chosen on one random half of the clips and reported on the other half.
"""
import io
import json
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from neudecide import NeuDecide
from neudecide.model import parse_calls

from run import DATA, ROOT, SEED, TOOLS, clips, share
from vad_conv import meeting_clips

HOME_WORDS = ['light', 'lamp', 'music', 'heat', 'temperature', 'volume', 'loud', 'quiet', 'sound', 'language',
              'bring', 'fetch', 'get me', 'play', 'pause', 'stop', 'turn', 'switch', 'warm', 'cold', 'hot', 'cool']


class Recorder:
    """Wraps the model's own selector and keeps the log-probability of each token it picks."""

    def __init__(self, inner):
        self.inner, self.logp = inner, []

    def select(self, logits):
        token = self.inner.select(logits)
        x = np.asarray(logits, dtype=np.float64).reshape(-1)
        x = x - x.max()
        self.logp.append(float(x[token] - np.log(np.exp(x).sum())))
        return token


def scored(model, audio, rate, tools=TOOLS):
    rec = Recorder(model.selector(tools))
    ids = model.generate_ids(audio, tools, sample_rate=rate, selector=rec)
    calls = parse_calls(model.decode(ids))
    lp = rec.logp[1:] or [0.0]  # the first token is forced
    return calls, {'mean': float(np.mean(lp)), 'min': float(np.min(lp)), 'early': float(np.mean(lp[:4]))}


def auc(pos, neg):
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    model = NeuDecide.from_pretrained()
    out = {}

    # close microphone, paired with the distant recordings
    ihm = pq.read_table(DATA / 'ami' / 'ihm' / 'validation-00004-of-00005.parquet')
    key = {(r['meeting_id'], r['speaker_id'], round(r['begin_time'], 2)): i
           for i, r in enumerate(ihm.select(['meeting_id', 'speaker_id', 'begin_time']).to_pylist())}
    sdm = pq.read_table(DATA / 'ami' / 'sdm' / 'validation-00004-of-00005.parquet',
                        columns=['audio_id', 'meeting_id', 'speaker_id', 'begin_time']).to_pylist()
    where = {r['audio_id']: key.get((r['meeting_id'], r['speaker_id'], round(r['begin_time'], 2))) for r in sdm}
    far, near, plain, records = [], [], [], []
    for row, x, rate in meeting_clips(300):
        calls, score = scored(model, x, rate)
        records.append({'set': 'meeting', 'fired': bool(calls), **score})
        wordy = any(w in row['text'].lower() for w in HOME_WORDS)
        plain.append((wordy, bool(calls)))
        j = where.get(row['audio_id'])
        if j is not None:
            y, r2 = sf.read(io.BytesIO(ihm.column('audio')[j].as_py()['bytes']), dtype='float32')
            far.append(bool(calls))
            near.append(bool(model.generate(y, TOOLS, sample_rate=r2)))
    out['close_mic'] = {'distant': share(far), 'headset': share(near),
                        'fired_on_both': sum(a and b for a, b in zip(far, near)),
                        'only_distant': sum(a and not b for a, b in zip(far, near)),
                        'only_headset': sum(b and not a for a, b in zip(far, near))}
    out['wording'] = {'no_home_word': share([f for w, f in plain if not w]), 'has_home_word': share([f for w, f in plain if w])}

    # confidence threshold
    for meta, audio, rate in clips(300):
        calls, score = scored(model, audio, rate)
        right = bool(calls) and calls[0]['name'] == meta.get('gold')
        records.append({'set': meta['set'], 'fired': bool(calls), 'right': right, **score})
    rng = random.Random(SEED)
    for r in records:
        r['half'] = rng.random() < 0.5
    fit = [r for r in records if r['half']]
    test = [r for r in records if not r['half']]

    def split(rows):
        return ([r for r in rows if r['set'] == 'commands' and r['fired']],
                [r for r in rows if r['set'] != 'commands' and r['fired']])

    pos, neg = split(fit)
    aucs = {k: round(auc([r[k] for r in pos], [r[k] for r in neg]), 3) for k in ['mean', 'min', 'early']}
    best = max(aucs, key=aucs.get)
    threshold = float(np.quantile([r[best] for r in pos], 0.05))  # keeps 95% of the fitting half's commands
    tpos, tneg = split(test)
    result = {'auc_on_fitting_half': aucs, 'score_used': best, 'threshold': round(threshold, 3),
              'auc_on_test_half': round(auc([r[best] for r in tpos], [r[best] for r in tneg]), 3)}
    for s in ['commands', 'meeting', 'speech', 'sounds', 'synthetic']:
        sub = [r for r in test if r['set'] == s]
        result[s] = {'fired': share([r['fired'] for r in sub]),
                     'fired_above_threshold': share([r['fired'] and r[best] >= threshold for r in sub])}
        if s == 'commands':
            result[s]['right_tool'] = share([r['right'] for r in sub])
            result[s]['right_tool_above_threshold'] = share([r['right'] and r[best] >= threshold for r in sub])
    out['confidence'] = result
    (ROOT / 'results' / 'confidence.json').write_text(json.dumps(out, indent=1) + '\n')
    c = out['close_mic']
    print('close_mic', c['distant']['count'], c['headset']['count'], c['distant']['of'], c['fired_on_both'], c['only_distant'], c['only_headset'])
    print('wording', {k: (v['count'], v['of']) for k, v in out['wording'].items()})
    print({k: v for k, v in result.items() if not isinstance(v, dict) or k == 'auc_on_fitting_half'})
    for s in ['commands', 'meeting', 'speech', 'sounds', 'synthetic']:
        print(s, {k: (v['count'], v['of']) for k, v in result[s].items()})


if __name__ == '__main__':
    main()
