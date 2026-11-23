# NeuDecide: what stops a false trigger, and what does not

[NeuDecide](https://github.com/neuphonic/neudecide) turns speech straight into a tool call. Its README says it can fire on noise and suggests putting a gate in front: voice-activity detection, or an explicit "no action" tool. This is a black-box test of both gates on audio with no command in it.

**What I found:**

1. **A good voice-activity detector solves noise.** Silero VAD cut false calls on everyday sounds from 98% to 2%.
2. **The "no action" tool mostly does not work.** Across 40 random tool lists it fixed 4 of the 36 that needed it.
3. **Speech that is not a command is the open problem.** Neither gate helps. Meeting conversation recorded across a room triggered a tool call 82% of the time, and 72% with the detector in front.

This page has been rewritten twice as controls overturned earlier readings; the history is at the end, with the limits.

## Setup

| Set | Clips | What it is |
| --- | ---: | --- |
| Commands | 200 | Fluent Speech Commands test clips (smart-home commands, 73 speakers) |
| Read speech | 300 | LibriSpeech audiobook speech |
| Meeting speech | 300 | AMI meeting utterances of 1.5 to 6 s, single distant microphone |
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

## 2. The "no action" tool is not a reliable gate

Across the 40 random tool lists, 50 sound clips each:

| | Lists |
| --- | ---: |
| Fired on at least half the sounds without the gate | 36 |
| … of those, 10% or fewer with `noAction` added | 4 |
| … of those, still at least half | 28 |

The median list fired on 100% of sounds without the gate and 97% with it. The two lists I tested first (home and vendor) both happened to be among the few it fixes, 98% to 4%. With the phone list 98% still fired.

When it does work it costs real commands: right tool 93% [89–96] without it, 88.5% [83–92] with it, on the home list. It took 7 of 200 commands ("Stop", "Resume", "Too quiet").

Position has its own effect. A tool with no parameters that is not a gate (`getBattery`) took most of the false calls when placed first and few when placed last.

## 3. Speech that is not a command

No detector can remove this, since it is speech.

| Home list | Read speech | Meeting speech |
| --- | ---: | ---: |
| No gate | 32% [27–37] | 82% [78–86] |
| `noAction` tool | 40% [35–46] | 58% [53–64] |
| Silero VAD in front | 32% | 72% [66–76] |
| Both | 40% | 55% [50–61] |

- **Across tool lists** the read-speech figure runs from 2% to 74%, median 32% without the gate and 33% with it.
- **Shorter is worse.** Read speech cut to its first 3 seconds fired 56% of the time.
- **It is not loudness.** Meeting clips are quiet (median peak 0.02); scaled to full level they fired 81% of the time.
- **Other lists agree on meetings:** vendor 67%, phone 89%; with `noAction` 30% and 61%.
- **Commands for another device fire too.** 42% of the smart-home commands called a vacuum tool from the vendor list.

Examples from meetings, home list: "Thank you very much" called `bringItem(socks)`; "Welcome back" called `setLights(on)`.

## Checks that this is not my own artefact

- A WAV file path and the array give the same answer on 94 of 96 clips. The two differences come from 16-bit rounding, which changes the answer on 15 of 200 non-command clips.
- With the output grammar off, 99 of 100 sounds still produce a call.
- NeuDecide's own example tools behave like mine.

## Limits

- **Black box.** I do not know why a tool list fires or not. One list of six fired on 2% of sounds; adding one tool took it to 98%; renaming that tool changed nothing.
- **Tools and wording are mine.** 17 tools in one writing style and one `noAction` wording. A second wording (`ignore`) did worse.
- **Meeting speech is a hard case.** AMI is recorded far from the speakers, and the meetings are about designing a remote control, so words like "volume" come up. One meeting split, 300 utterances.
- **The 0.3 s detector rule is my choice.** A stricter rule would pass less and lose more commands.
- **No general accuracy figure.** Fluent Speech Commands is small and clean, and I wrote the sentence-to-tool mapping.
- **Not a bug list.** Calls sometimes lack a required argument or use a value outside an `enum`; the grammar is documented as constraining names and keys only.
- Intervals treat clips as independent, though the sets have several clips per sound type or speaker.

## How this page changed

1. First: "98% of sounds trigger a call; the `noAction` tool cuts it to 4%." True for one tool list.
2. Then: "the rate swings from 2% to 99% with the tool list." Four lists made that look typical. Forty random lists show nearly all fire, and the 2% list was the outlier.
3. Now: as above, after adding the detector and meeting speech.

## Reproduce

Needs a Hugging Face account with access to `neuphonic/neudecide`.

```sh
pip install neudecide pyarrow soundfile numpy huggingface_hub webrtcvad-wheels
git clone --depth 1 https://github.com/neuphonic/neudecide upstream   # for its example tools
sh fetch.sh              # about 740 MB of public audio and the Silero model into data/
python run.py            # home list, with and without the gate tool
python checks.py         # tool order
python controls.py       # other tool lists and artefact checks
python sinks.py          # each tool alone, renames
python swap.py           # swapping and dropping tools
python random_lists.py   # 40 random tool lists (slow)
python vad_conv.py       # WebRTC VAD, meeting speech
python silero_check.py   # Silero VAD, loudness check
```

Results are in `results/`.

## Credits

Fluent Speech Commands (Lugosch et al., 2019), LibriSpeech (Panayotov et al., 2015, CC BY 4.0), AMI Meeting Corpus (Carletta et al., 2005, CC BY 4.0), ESC-50 (Piczak, 2015, CC BY-NC), Silero VAD (MIT).
