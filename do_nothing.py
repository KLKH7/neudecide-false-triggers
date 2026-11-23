"""`doNothing` was the best of five gate wordings on 12 tool lists. Does that hold on lists it was not picked on?

    python do_nothing.py        # writes results/do_nothing.json

  fresh lists   20 of the 40 lists from random_lists.py (none used to pick the wording), 40 sounds and 40 meeting clips
  name or text  the name `doNothing` with the long description, and the name `noAction` with "Do nothing."
  cost          the home list on all 200 commands and all 300 meeting clips, with Silero VAD as well
"""
import json
import statistics

from neudecide import NeuDecide

from controls import PHONE, VENDOR
from run import ROOT, TOOLS, clips, share
from silero_check import speech_seconds
from vad_conv import meeting_clips
from wordings import EMPTY, WORDINGS

LONG = WORDINGS['noAction']
VARIANTS = {
    'doNothing': ('doNothing', 'Do nothing.'),
    'doNothing_long_text': ('doNothing', LONG),
    'noAction_short_text': ('noAction', 'Do nothing.'),
    'noAction': ('noAction', LONG),
}
BY_NAME = {t['name']: t for t in TOOLS + PHONE + VENDOR}


def gate(key):
    name, text = VARIANTS[key]
    return {'name': name, 'description': text, 'parameters': EMPTY}


def main():
    model = NeuDecide.from_pretrained()

    def fired(tools, rows, name=None):
        return [any(c['name'] != name for c in model.generate(a, tools, sample_rate=r)) for a, r in rows]

    meetings = [(x, r) for _, x, r in meeting_clips(300)]
    everything = list(clips(300))
    sounds = [(a, r) for m, a, r in everything if m['set'] == 'sounds'][:40]
    lists = json.loads((ROOT / 'results' / 'random_lists.json').read_text())['lists'][:20]
    rows = []
    for item in lists:
        tools = [BY_NAME[n] for n in item['tools']]
        row = {'tools': item['tools']}
        for s, sample in [('sounds', sounds), ('meeting', meetings[:40])]:
            row[s] = {'none_added': sum(fired(tools, sample))}
            for key in VARIANTS:
                row[s][key] = sum(fired(tools + [gate(key)], sample, VARIANTS[key][0]))
        rows.append(row)
        print(row, flush=True)
    out = {'fresh_lists': {
        'of': 40, 'lists': len(rows),
        'median_fired': {s: {k: statistics.median(r[s][k] for r in rows) for k in rows[0][s]} for s in ['sounds', 'meeting']},
        'lists_cut_to_10pct_or_less': {s: {k: sum(r[s][k] <= 4 for r in rows) for k in rows[0][s]} for s in ['sounds', 'meeting']},
        'rows': rows}}

    home = TOOLS + [gate('doNothing')]
    commands = [(m, a, r) for m, a, r in everything if m['set'] == 'commands']
    got = [model.generate(a, home, sample_rate=r) for _, a, r in commands]
    passes = [speech_seconds(a, r) >= 0.3 for a, r in meetings]
    meet = fired(home, meetings, 'doNothing')
    out['home_list'] = {
        'commands_right_tool': share([bool(c) and c[0]['name'] == m['gold'] for (m, _, _), c in zip(commands, got)]),
        'commands_sent_to_gate': share([bool(c) and c[0]['name'] == 'doNothing' for c in got]),
        'meeting_fired': share(meet), 'meeting_fired_with_vad': share([p and f for p, f in zip(passes, meet)]),
        'read_speech_fired': share(fired(home, [(a, r) for m, a, r in everything if m['set'] == 'speech'], 'doNothing')),
        'sounds_fired': share(fired(home, [(a, r) for m, a, r in everything if m['set'] == 'sounds'], 'doNothing'))}
    (ROOT / 'results' / 'do_nothing.json').write_text(json.dumps(out, indent=1) + '\n')
    print('SUMMARY', json.dumps({'median': out['fresh_lists']['median_fired'], 'cut': out['fresh_lists']['lists_cut_to_10pct_or_less'],
                                 'home': {k: (v['count'], v['of']) for k, v in out['home_list'].items()}}))


if __name__ == '__main__':
    main()
