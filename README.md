# NeuDecide: false triggers depend on the tool list

[NeuDecide](https://github.com/neuphonic/neudecide) turns speech straight into a tool call. Its README says it can fire on noise and suggests a gate, such as an explicit "no action" tool. This is a black-box test of how often it fires when nobody gave a command, and of that gate.

**Short version: the answer swings between 2% and 99% depending on which tools are in the list, and I could not find a rule that predicts it.** The gate fixed it for two tool lists and not for a third.

An earlier version of this page led with "98% of sounds trigger a call, the gate cuts it to 4%". Both figures are real but belong to one tool list; the controls below show they do not generalise. The limits are at the end.

## Setup

| Set | Clips | What it is |
| --- | ---: | --- |
| Commands | 200 | Fluent Speech Commands test clips (smart-home commands, 73 speakers) |
| Speech | 300 | LibriSpeech read audiobook speech: talking, but no command |
| Sounds | 300 | ESC-50: dogs, rain, engines, keyboards |
| Synthetic | 21 | Silence, white noise and pink noise at three levels |

Three tool lists: seven smart-home tools I wrote ("home"), seven phone-assistant tools I wrote ("phone"), and the three robot-vacuum tools from NeuDecide's own example ("vendor"). The gate is a `noAction` tool with no parameters, placed last. Decoding is greedy, so each clip gives one answer. Brackets are 95% Wilson intervals.

## 1. It depends on the tool list

Share of clips with no command in them where the model called a tool:

| Tool list | Sounds (300) | Speech (300) |
| --- | ---: | ---: |
| Home, seven tools | 98% [96–99] | 32% [27–37] |
| Home without `bringItem`, six tools | 2% [1–5] | 6% [4–9] |
| Vendor example, three tools | 97% [95–99] | 23% [19–28] |
| Phone, seven tools | 99% [98–100] | 41% [35–46] |

Removing one tool took the home list from 98% to 2%. On the first 100 sound clips:

| Change to the home list | Fired | Tool that took the calls |
| --- | ---: | --- |
| None | 99 | `bringItem` |
| Remove `bringItem` | 2 | |
| Remove `setLamp` instead | 100 | `bringItem`, `setLanguage` |
| Rename `bringItem` to `orderTaxi` | 99 | `orderTaxi` |
| Rewrite its description | 99 | `bringItem` |
| Make its argument free text | 100 | `bringItem` |
| Swap it for `takePhoto` | 89 | `takePhoto` |
| Swap it for `openApp` | 19 | `openApp` |

So it is not the tool's name, its description, the number of tools or its position (moving it first changed nothing). Offered on its own, `bringItem` fired on 68 of 100 sounds; `takePhoto`, which took 83 calls when swapped in, fired on 0 of 100 alone. I do not know what makes a tool attract these calls.

## 2. The gate works for some tool lists

Sounds that still called a real tool with `noAction` added:

| Tool list | No gate | With `noAction` |
| --- | ---: | ---: |
| Home | 98% | 4% [3–7] |
| Vendor example | 97% | 4% [3–7] |
| Phone | 99% | 98% [96–99] |

On real commands with the home list, the right tool was chosen 93% [89–96] without the gate and 88.5% [83–92] with it. The gate took 7 of 200 real commands ("Stop", "Resume", "Too quiet").

Position matters separately from meaning. A tool with no parameters that is not a gate (`getBattery`) took 308 of the false calls when placed first and 27 when placed last. A gate named `ignore` cut sounds to 3% when first and left 91% when last, so its success when first may be position, not meaning.

## 3. Speech

False calls on speech with no command ranged from 6% to 57% across the lists above. Two things move it more than the gate does:

- **Length.** The same clips cut to their first 3 seconds fired 56% of the time, against 32% in full.
- **Any extra tool.** Adding `getBattery` last raised it from 32% to 57%. Adding `noAction` raised it to 40%, which I first read as the gate backfiring. It is inside the swing any added tool causes, so I make no claim about the gate on speech.

Commands meant for another device also fire: 42% of the smart-home commands called a vacuum tool from the vendor list.

## 4. Small input changes flip answers

The same clip as float samples and as 16-bit samples gave a different answer for 15 of 200 clips with no command in them.

## What I checked that could have made this an artefact

- **My audio handling.** A WAV file path and the array gave the same answer on 94 of 96 clips; the two differences are the 16-bit effect above.
- **The output grammar.** With constrained decoding off, 99 of 100 sounds still produced a call.
- **My tool lists.** NeuDecide's own example tools behave the same way (97%).

## What this does not show

- **Why.** This is a black-box test. I have no explanation for which tools attract calls.
- **Real deployment rates.** NeuDecide's README says to put voice-activity detection in front, which should remove most of the sounds and noise cases. I did not test with it. Read audiobook speech is also not conversation near a device.
- **A general accuracy figure.** Fluent Speech Commands is small and clean, and I wrote the mapping from sentence to tool.
- **Schema violations as a bug.** Calls sometimes lack a required argument or use a value outside an `enum`. The NeuDecide grammar is documented as constraining tool names and argument keys only, so this is by design.
- **Many tool lists.** Four lists and a handful of swaps. Enough to show the rate is unstable, not to map when.
- Intervals treat clips as independent, though ESC-50 has several clips per sound type and LibriSpeech several per speaker.

## Reproduce

Needs a Hugging Face account with access to `neuphonic/neudecide`.

```sh
pip install neudecide pyarrow soundfile numpy huggingface_hub
git clone --depth 1 https://github.com/neuphonic/neudecide upstream   # for its example tools
sh fetch.sh           # about 520 MB of public audio into data/
python run.py         # home list, with and without gates -> results/summary.json
python checks.py      # tool order, schema check -> results/checks.json
python controls.py    # other tool lists and artefact checks -> results/controls.json
python sinks.py       # each tool alone, renames -> results/sinks.json
python swap.py        # swapping and dropping tools -> results/swap.json
```

## Credits

Fluent Speech Commands (Lugosch et al., 2019), LibriSpeech (Panayotov et al., 2015, CC BY 4.0), ESC-50 (Piczak, 2015, CC BY-NC).
