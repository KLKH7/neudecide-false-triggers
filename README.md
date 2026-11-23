# NeuDecide: what stops a false trigger, and what does not

[NeuDecide](https://github.com/neuphonic/neudecide) turns speech straight into a tool call. Its README says it can fire on noise and suggests putting a gate in front: voice-activity detection, or an explicit "no action" tool. This is a black-box test of both gates on audio with no command in it.

**What I found:**

1. **A good voice-activity detector solves noise.** Silero VAD cut false calls on everyday sounds from 98% to 2%.
2. **The "no action" tool depends on its wording.** Named `doNothing` it fixed sounds for 10 of 20 random tool lists; with my first wording, 3 of 20. It costs about 6% of real commands.
3. **Speech that is not a command is the open problem.** Meeting conversation triggered a tool call 82% of the time and a TV monologue 51%. The detector, the best gate wording and a confidence threshold each helped somewhere, and none got either below about 40%.
4. **So is a request no tool covers.** With a wake word in front the model only hears requests. Asked for something its tool list cannot do, it called a tool anyway 47% of the time, 43% with the gate.

This page has been rewritten several times as controls overturned earlier readings; the history is at the end, with the limits.

## Setup

| Set | Clips | What it is |
| --- | ---: | --- |
| Commands | 200 | Fluent Speech Commands test clips (smart-home commands, 73 speakers) |
| Read speech | 300 | LibriSpeech audiobook speech |
| Meeting speech | 300 | AMI meeting utterances of 1.5 to 6 s, single distant microphone |
| TV speech | 300 | A late-night TV monologue (the "Meanwhile" set), 4-second windows |
| Sounds | 300 | ESC-50: dogs, rain, engines, keyboards |
| Synthetic | 21 | Silence, white noise and pink noise at three levels |

The main tool list is seven smart-home tools I wrote ("home"). Others: seven phone-assistant tools ("phone"), the three robot-vacuum tools from NeuDecide's own example ("vendor"), and 40 random lists of 3 to 8 tools drawn from all 17. The gate tool is `noAction`, no parameters, placed last. Decoding is greedy, so each clip gives one answer. Brackets are 95% Wilson intervals.

## 1. Voice-activity detection solves noise, if the detector is good

Share of clips where the model called a real tool, home list:

| | Sounds | Synthetic | Real commands kept |
| --- | ---: | ---: | ---: |
| No gate | 98% [96–99] | 20 of 21 | 200 of 200 |
| Silero VAD in front | 2% [1–5] | 0 of 21 | 199 of 200 |
| WebRTC VAD in front (strictest mode) | 80% [75–84] | 3 of 21 | 199 of 200 |

A clip passes if the detector finds at least 0.3 s of speech. WebRTC VAD let 247 of 300 sounds through and rejected 273 of 300 meeting utterances, so the choice of detector matters.

Without a detector, firing on sounds is the norm, not a quirk of one tool list: 34 of 40 random lists fired on at least 90% of sounds, and 3 fired on 10% or fewer.

## 2. The "no action" tool depends on its wording

I first tested one wording: `noAction`, "Use when the audio is not a command for this device: conversation, background noise or silence." Across 40 random tool lists it brought sounds to 10% or fewer for 4 of the 36 lists that fired on at least half of them.

Then five wordings on 12 lists. The best was `doNothing`, "Do nothing." Checked on 20 lists it was not chosen on, 40 sound clips each:

| Gate | Median list: sounds fired | Lists at 10% or fewer |
| --- | ---: | ---: |
| None | 100% | 2 of 20 |
| `noAction`, long description | 98% | 3 of 20 |
| `noAction`, "Do nothing." | 54% | 5 of 20 |
| `doNothing`, long description | 44% | 10 of 20 |
| `doNothing`, "Do nothing." | 14% | 10 of 20 |

Both the name and the description matter, and the best wording still fails for half the lists.

It costs real commands. On the home list the right tool was chosen 93% [89–96] with no gate, 88.5% [83–92] with `noAction` and 87% [82–91] with `doNothing`, which took 14 of 200 commands for itself.

Position has its own effect. A tool with no parameters that is not a gate (`getBattery`) took most of the false calls when placed first and few when placed last.

## 3. Speech that is not a command

No detector can remove this, since it is speech. Home list, share of clips that called a real tool:

| | Read speech | Meeting speech |
| --- | ---: | ---: |
| No gate | 32% [27–37] | 82% [78–86] |
| Silero VAD in front | 32% | 72% [66–76] |
| `noAction` tool | 40% [35–46] | 58% [53–64] |
| `doNothing` tool | 36% [30–41] | 41% [35–46] |
| `doNothing` and VAD | | 39% [33–44] |

Across the 20 fresh tool lists the median list fired on 74% of meeting clips with no gate and 49% with `doNothing`. No list got to 10% or fewer. (On the 12 lists used to pick the wording, 4 did; that did not repeat.) The `doNothing` name with the long description did better on meeting clips, 31%, with 2 of 20 lists at 10% or fewer, and worse on sounds, so no one wording wins on both.

What the meeting result is not:

- **Not distance.** The same utterances from the speakers' headset microphones fired 73% of the time, against 82% from across the room (293 pairs).
- **Not loudness.** Meeting clips are quiet (median peak 0.02); scaled to full level they fired 81%.
- **Not command-like words.** Leaving out every utterance with a word a smart-home tool could act on ("volume", "light", "turn" and so on), 81% of the remaining 278 fired.
- **Not one tool list.** Vendor 67%, phone 89%.

**A confidence threshold helps partly.** NeuDecide has no confidence score, but the decoder's token probabilities are available. The mean log-probability of the first four tokens separates real commands from false calls with an AUC of 0.93. A threshold chosen on half the clips to keep 95% of commands, applied to the other half, kept 96 of 101 commands and cut meeting false calls from 85% to 54%.

Other things that move the speech figure:

- **Tool list.** Read speech ran from 2% to 74% across 40 random lists, median 32%.
- **Length.** Read speech cut to its first 3 seconds fired 56% of the time.
- **Another device's commands.** 42% of the smart-home commands called a vacuum tool from the vendor list.

**A second source agrees on the floor, not on the gate.** The TV monologue fired 51% [45–56] with no gate and 48% [42–54] with `doNothing`; the detector passed 294 of 300 windows. The gate that halved the meeting figure did almost nothing here.

Examples from meetings, home list: "Thank you very much" called `bringItem(socks)`; "Welcome back" called `setLights(on)`.

## 4. A request no tool covers

A wake word or a push-to-talk button would stop most of the speech above from reaching the model. What remains is a real request the tool list cannot serve. NeuDecide's README says it returns `[]` when no tool applies.

The 200 spoken commands against 20 random tool lists: a command is in scope for a list that holds its tool, out of scope otherwise.

| | No gate | With `doNothing` |
| --- | ---: | ---: |
| In scope: right tool | 89% | 86.5% |
| Out of scope: called a tool anyway | 47% | 43% |

By list, the out-of-scope figure ran from 14% to 79% (median 47%) with no gate, and from 12% to 76% (median 43%) with it. I give no intervals because the same 200 commands recur across lists.

## Checks that this is not my own artefact

- A WAV file path and the array give the same answer on 94 of 96 clips. The two differences come from 16-bit rounding, which changes the answer on 15 of 200 non-command clips.
- With the output grammar off, 99 of 100 sounds still produce a call.
- NeuDecide's own example tools behave like mine.

## Limits

- **Black box.** I do not know why a tool list fires or not. One list of six fired on 2% of sounds; adding one tool took it to 98%; renaming that tool changed nothing.
- **Tools and wording are mine.** 17 tools in one writing style and five gate wordings. A sixth wording could do better than `doNothing`; nothing here says which wording is best in general.
- **The confidence threshold is fitted and tested on one tool list.**
- **No wake word in sections 1 to 3.** A deployed device may only pass audio to the model after one, which would remove most of that speech. Section 4 is the case that survives it.
- **Meeting speech is a hard case.** AMI is recorded far from the speakers, and the meetings are about designing a remote control, so words like "volume" come up. One meeting split, 300 utterances.
- **The 0.3 s detector rule is my choice.** A stricter rule would pass less and lose more commands.
- **No general accuracy figure.** Fluent Speech Commands is small and clean, and I wrote the sentence-to-tool mapping.
- **Not a bug list.** Calls sometimes lack a required argument or use a value outside an `enum`; the grammar is documented as constraining names and keys only.
- Intervals treat clips as independent, though the sets have several clips per sound type or speaker.

## How this page changed

1. First: "98% of sounds trigger a call; the `noAction` tool cuts it to 4%." True for one tool list.
2. Then: "the rate swings from 2% to 99% with the tool list." Four lists made that look typical. Forty random lists show nearly all fire, and the 2% list was the outlier.
3. Then: "the `noAction` tool fixed 4 of 36 lists, so the gate tool mostly does not work." True of that wording. A different wording fixed half.
4. Now: as above, with a second speech source and out-of-scope requests added.

## Reproduce

Needs a Hugging Face account with access to `neuphonic/neudecide`.

```sh
pip install neudecide pyarrow soundfile numpy huggingface_hub webrtcvad-wheels
git clone --depth 1 https://github.com/neuphonic/neudecide upstream   # for its example tools
sh fetch.sh              # about 980 MB of public audio and the Silero model into data/
python run.py            # home list, with and without the gate tool
python checks.py         # tool order
python controls.py       # other tool lists and artefact checks
python sinks.py          # each tool alone, renames
python swap.py           # swapping and dropping tools
python random_lists.py   # 40 random tool lists (slow)
python vad_conv.py       # WebRTC VAD, meeting speech
python silero_check.py   # Silero VAD, loudness check
python wordings.py       # five gate wordings on 12 lists (slow)
python do_nothing.py     # the best wording on 20 fresh lists, and its cost
python confidence.py     # headset microphone, command words, confidence threshold
python scope.py          # requests no tool covers, TV speech (slow)
```

Results are in `results/`.

## Credits

Fluent Speech Commands (Lugosch et al., 2019), LibriSpeech (Panayotov et al., 2015, CC BY 4.0), AMI Meeting Corpus (Carletta et al., 2005, CC BY 4.0), the Meanwhile set (distil-whisper), ESC-50 (Piczak, 2015, CC BY-NC), Silero VAD (MIT).
