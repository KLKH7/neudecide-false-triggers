# NeuDecide: false triggers and the "no action" gate

[NeuDecide](https://github.com/neuphonic/neudecide) turns speech straight into a tool call. Its README says it can fire on noise and suggests putting a gate in front, such as an explicit "no action" tool. This measures how often it fires when nobody gave a command, and how well that gate works.

This is a first pass with one tool list. Its limits are at the end.

## Setup

Seven smart-home tools (lights, lamp, music, heating, volume, fetch an item, language), run on four sets of audio:

| Set | Clips | What it is |
| --- | ---: | --- |
| Commands | 200 | Fluent Speech Commands test clips, 73 speakers |
| Speech | 300 | LibriSpeech read audiobook speech: talking, but no command |
| Sounds | 300 | ESC-50: dogs, rain, engines, keyboards |
| Synthetic | 21 | Silence, white noise and pink noise at three levels |

Decoding is greedy, so each clip gives one answer. Brackets are 95% Wilson intervals.

## Results

**Share of clips where the model asked the device to do something** (any call other than the gate tool):

| Tool list | Commands | Speech | Sounds | Synthetic |
| --- | ---: | ---: | ---: | ---: |
| Seven tools | 100% | 32% [27–37] | 98% [96–99] | 20 of 21 |
| + `noAction`, last | 96.5% | 40% [35–46] | 4% [3–7] | 0 of 21 |
| + `noAction`, first | 96% | 42% | 3% | 0 of 21 |
| + `ignore`, last | 97% | 23% [19–28] | 91% [87–93] | 14 of 21 |
| + `ignore`, first | 92.5% | 18% | 3% | 0 of 21 |

**Right tool on real commands:** 93% [89–96] with seven tools, 88.5% [83–92] with `noAction` added. The gate swallowed 7 of 200 real commands, all short or about volume ("Stop", "Resume", "Too quiet").

What this shows:

1. **Without a gate, almost any sound produces a call.** 294 of 300 everyday sounds and 20 of 21 silence or noise clips did.
2. **The gate works on sounds and not on speech.** `noAction` cut false calls on sounds from 98% to 4%. On speech with no command in it, false calls went from 32% to 40%: adding the gate stopped 15 clips and started 39. Voice-activity detection cannot help with this case, since it is speech.
3. **The gate depends on its name and position.** Renamed `ignore` with a different description and placed last, it left 91% of sounds firing. Placed first, the same tool cut that to 3%.
4. **One tool takes most false calls.** `bringItem` received 80% of them with seven tools and 85% with the tool order reversed, usually with no arguments.
5. **Arguments can break the schema.** 256 of 410 false calls lacked a required argument. Some values fell outside the tool's `enum`: `room: "ashroom"`, `item: "dunes"`, `item: "wolf's paws"`.

**A schema check after the model** (drop calls missing a required argument or using a value outside its enum) cuts false calls on sounds from 98% to 18% with no gate. It costs real commands, because the model also drops required arguments there ("Less heat" gives `changeHeat {}`): right tool falls from 93% to 80%.

Full figures are in `results/summary.json` and `results/checks.json`.

## Reproduce

Needs a Hugging Face account with access to `neuphonic/neudecide`.

```sh
pip install neudecide pyarrow soundfile numpy huggingface_hub
sh fetch.sh           # about 520 MB of public audio into data/
python run.py         # writes results/calls.jsonl and results/summary.json
python checks.py      # writes results/checks.json
```

## Limits

- One tool list. Which tool attracts false calls, and how often, will differ with other tools.
- Read audiobook speech is not conversation. Speech near a device, such as a television or people talking about the lights, is the harder and more useful test.
- Six commands say "bathroom", which my `room` enum does not offer (it has "washroom"). Those count as outside the enum through my tool list's fault, though the model did still emit a value the schema forbids.
- Fluent Speech Commands is a small, clean set with repeated phrasings, so 93% is not a general accuracy figure.
- Only the gate tool was tested, not voice-activity detection in front of the model, which should remove most of the sounds and noise cases.
- Two gate wordings and two positions. That is enough to show wording matters, not to say which wording is best.

## Credits

Fluent Speech Commands (Lugosch et al., 2019), LibriSpeech (Panayotov et al., 2015, CC BY 4.0), ESC-50 (Piczak, 2015, CC BY-NC).
