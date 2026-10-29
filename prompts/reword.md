# Rewordings for ε

`reword.jsonl` holds 300 SEED validation event labels (picked by `eddy.py reword-sources --n 300`, seed 0) and 5 rewordings of each. They set ε in [step 4](../docs/steps/04-evaluate.md#set-ε-from-rewordings). They are never used for training.

- **LLM:** Claude Opus 5.5 (`claude-opus-5-5`), run as three Claude Code subagents of 100 labels each, on 2026-10-28.
- **Instruction:** the text below, given word for word to every subagent.

## Instruction

You get motion labels from a motion-capture dataset. Write 5 rewordings of each label.

A rewording must describe exactly the same motion. A motion generator that reads it must produce the same motion as for the original label.

Keep every detail:
- body side (left / right), body part, direction, facing, and turns,
- counts (steps, repetitions), speed, size of the movement,
- props, injuries, emotions, and character type ("a person", "an injured person", "a dancer"),
- the order of the actions.

Do not add details that are not in the label. Do not remove details.

Change the wording: use other verbs, other word order, other sentence structure. Make the 5 rewordings different from each other and from the label.

Keep the style of the label: third person, plain English, one or two sentences, ending with a period, about the same length. Start with "A person", "The person", "Someone", or the label's own subject, and vary the start.

Do not fix the label's typos in a way that changes the meaning. Do not copy the label unchanged.
