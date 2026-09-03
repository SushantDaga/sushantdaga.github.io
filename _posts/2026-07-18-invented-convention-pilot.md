---
title: "GRPO Has a Silent Failure Mode. Here's How to Spot It."
description: "Correct answers kept landing in zero-advantage tied groups, so one sub-skill never moved while the headline score climbed; the tied-group share is cheap to compute on your own logs."
tags: [blog, RL, negative-results]
date: 2026-07-28 21:00:00 +0000
---

This is a single small experiment. One random seed, one toy task, one setting, run on a
laptop-class machine. Treat it as a first probe, not a settled result.
The short version: during training, the model produced the correct answer to the hardest part
of its task about 15% of the time, and learned nothing from any of those successes. Not because
the answers were wrong. Because of where they landed.

## Setup: an invented code the model has to learn

We invented a secret code that swaps digits. Each of the ten digits 0 through 9 maps to some
other digit: 4 becomes 9, 7 becomes 2, and so on. The map is one fixed random shuffle, chosen
once and used for the whole experiment. Eight of the ten mappings are printed in the prompt the
model sees. Two are held back, and they are held back in every prompt the model ever sees, in
training and at evaluation alike. The model gets a short string of digits and must output the
same string with every digit swapped by the code.

Think of a decoder ring with ten spokes. Eight spokes are labeled where the reader can see
them. Two spokes have the label rubbed off. The model can read the visible spokes straight from
the prompt. The two rubbed-off spokes are the test.

The actual code, with the two hidden mappings marked:

| Source digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Becomes | 3 | 6 | 7 | 0 | 9 | 1 | 8 | 2 | 4 | 5 |
| Printed in the prompt? | yes | **no** | yes | yes | yes | yes | **no** | yes | yes | yes |

The prompt also says the code is a one-to-one shuffle, so with eight mappings visible, the two
hidden ones can in principle be narrowed by elimination.
Read the second row of that table. The eight visible mappings account for eight of the ten
output digits, and the two they leave unspoken, 6 and 8, are exactly the two hidden targets. So
the task is not a test of whether the hidden mappings are strictly unguessable. It is a test of
whether training teaches them.

Try it yourself. Input "4 7" becomes "9 2": the row for 4 and the row for 7 are both printed,
so this is a lookup. Input "1 6" should become "6 8", but nothing printed in the prompt tells
you that. Source digits "1" and "6" are the two hidden mappings.

We built this small on purpose. Before training, we checked how likely the base model,
Qwen2.5-0.5B-Instruct, was to produce the exactly-correct output cold. The measurement
is a teacher-forced probability, the joint probability the model assigns to the whole correct
output string. It was taken over 80 fresh random draws of the code, the revealed subset, and
the input, not on the one fixed code the runs below train against. The median came out to
8.97e-4, about 1 in 1,100 (geometric mean 8.49e-4). Low, but far from impossible.

Two-digit inputs were the shortest length whose starting odds landed in that band. At 8 to 12
digits, our first attempt, the base model's odds fell as low as 1e-11, and it does not apply
even the revealed mappings reliably out of the box. Shrinking to two digits has one cost. The
input space is only 100 strings, so "held-out" means a disjoint 70/30 split of those 100
strings, not strings the model has never encountered in any form.

Training used GRPO (group-relative policy optimization): for each prompt, sample a group of
answers, score each one, subtract the group's own mean score, and push the model toward the
answers that scored above their group's average. We used LoRA (low-rank adaptation) to keep it
cheap: a small set of trainable low-rank matrices bolted onto the attention layers, updating a
few million parameters instead of all 0.5B. Eight answers per prompt, four prompts per step,
300 steps, checked every 25 steps on the held-out split, on a local Apple-silicon GPU.

Two decoding settings appear below. During training the model's answers are sampled at
temperature 1.0, which is what gives a group of eight answers any variety at all. Every
evaluation on the held-out split uses greedy decoding, the single highest-probability
continuation. Numbers from the two settings are not interchangeable.

## Three training variants, one shared freeze

We track one number per evaluation: the fraction of the 30 held-out strings the model
translates exactly right, both digits correct. Call it the combined score. We ran three
variants, all grading the same fixed 8-of-10 code, same budget:

- **A per-character grader.** Reward is the fraction of output digits correct, partial credit.
- **An exact-match grader.** Reward is 1 only if every digit is right, else 0.
- **The exact-match grader with the group doubled from 8 to 16 answers.** Same total compute,
  group size the only changed variable.

By the combined score, all three learn fine. Each climbs from 0 to roughly 0.6 to 0.7 over the
first 75 to 100 steps and stays in that band.

That plateau is a trap. Of the 30 held-out strings, 20 contain no hidden source digit at all,
and getting those right already buys a combined score of 0.667 on its own. A flat line at 0.667
can hide a channel that never moved.

![Fig 1. Held-out combined exact-match score for the three training variants over 300 steps: the per-character grader, the exact-match grader, and the exact-match grader with 16-answer groups. All three rise to about 0.667 by step 100 and stay near it, with the exact-match grader reaching 0.700 at steps 250 and 275 before settling back to 0.667. The dotted line marks 0.667, which is what the 20 held-out strings with no hidden digit are worth on their own. Two of the three variants sit on that line at step 300 by exactly that route, translating all 20 and missing all 10 of the strings that contain a hidden digit. The exact-match grader arrives at the same 0.667 by a different route: it misses one of the 20 and gets one of the 10 right.](/assets/images/writing/invented-convention-pilot/fig1_three_variants.png)

The other 10 held-out strings are where the hidden mappings get tested. Those 10 strings
contain 11 chances to produce a hidden mapping, because one string, "1 1," contains a hidden
source digit twice. We call each chance a slot: 11 slots across 10 strings. The 11 slots are
not 11 independent trials, and the structure matters for reading any count below. Nine of them
test the same hidden mapping, source digit "1." The remaining two test the other hidden
mapping, source digit "6." A model that learns one mapping and not the other moves nine slots
or two slots, not some fraction spread evenly across 11.

Split the score by channel and the plateau comes apart. The 8 revealed digits are one channel,
the 2 hidden digits the other. Every variant's revealed-channel accuracy climbs to 94 to 100%.
The hidden channel, at step 300, under greedy decoding, reads:

| variant | source "1" slots (9) | source "6" slots (2) | all hidden slots (11) |
|---|---|---|---|
| per-character grader | 0 correct | 0 correct | 0 of 11 |
| exact-match grader | 0 correct | 1 correct | 1 of 11 |
| exact-match grader, 16-answer groups | 0 correct | 0 correct | 0 of 11 |

Every variant starts at 0 of 11 at step 0. At step 300 no variant has a correct answer on the
nine-slot mapping, and one variant lands one of the two slots on the other mapping. The table
is a step-300 statement: the exact-match grader read 0.700 at steps 250 and 275, so one
hidden-digit string was right there too, and per-string answers were stored only at 0 and 300.

Here is what that looks like on real held-out strings at step 300, with the correct output on
the left and each variant's step-300 answer beside it.

| input | correct output | per-character grader | exact-match grader | 16-answer groups |
|---|---|---|---|---|
| 4 1 | 9 6 | 9 1 | 9 1 | 9 1 |
| 1 0 | 6 3 | 0 3 | 0 3 | 0 3 |
| 2 6 | 7 8 | 7 9 | **7 8** | 7 9 |
| 1 1 (both digits hidden) | 6 6 | 1 1 | 0 0 | 1 1 |

In every row that has a revealed digit, all three variants get that digit right. On "4 1" they
all answer "9 1," echoing the input "1" back rather than translating it to "6." The bolded cell
is the only hidden slot any variant got right here, and it sits on the two-slot mapping.

So a richer grade and a bigger group each had a real chance to break the freeze, and neither
did. Both only help if the model's answers still disagree on the hidden digit, which is where
the next section starts.

## The mechanism: right answers, scored as worth nothing

The model was not simply unable to produce the right answer. Every answer sampled in training
puts some digit in each hidden position, and each of those positions is a draw we can score.
Pooled over all 300 steps, the correct hidden digit came up often: 15.3% of hidden slot draws
under the exact-match grader (611 of 3,992), 8.0% under the per-character grader (318 of
3,992), and 2.8% under the 16-answer variant (111 of 3,952). The right token was there. The
question is what happened to it.

The 15.3% needs a yardstick before it means anything, and the yardstick is the model's own
uncertainty about what to put in a hidden position. Measure the spread of digits it actually emits there and you get an
entropy in bits, and bits convert to a candidate count by raising 2 to them: 3.06 bits is about
8 candidates, 1.76 bits about 3.4. Under the exact-match grader that spread reads 3.06 bits over
the first 25 steps and 1.76 bits over the last 25, so the model went from something like 8 live
candidates per hidden position to something like 3.4. Guess evenly among 8 and you hit the right
digit 12% of the time. Among 3.4, 30%. 15.3% sits inside that window. A model that used the
one-to-one structure and picked between the only two digits the prompt leaves unspoken would be
right 50% of the time, so it is well short of that as well.

That does not weaken the finding. It sharpens it. The correct hidden digits are consistent with
lucky guesses, and a lucky guess is exactly the raw material a group-relative update exists to
convert into learning: one right answer sitting beside seven wrong ones is the cheapest
teaching signal there is. It was there 611 times, and almost none of those times counted, for a
reason that has nothing to do with the answer being right.

The other two variants read below their own windows. The per-character grader's hidden-position
spread stayed between 2.65 and 3.20 bits for the whole run, about 6 to 9 candidates, which puts
even guessing at 11 to 16% against its observed 8.0%. No spread was recorded at hidden positions
for the 16-answer variant, so its 2.8% has no window to sit in. The record does not explain
either reading.

GRPO grades on a curve inside each group. Score each answer, subtract the group's mean, and the
result is that answer's advantage, the number the update is proportional to. When every answer
in a group gets an identical score, the group mean equals every score in it, so every
advantage in that group is exactly zero. The correct answer sitting in that group is worth
precisely as much as the wrong answers beside it: nothing. That exact-zero statement holds for
the plain group-relative objective used here, with no KL or entropy term added. Implementations
that add one still get nothing that ranks the correct answer above the wrong ones, because the
ranking part of the update is what went to zero.

That is where most of the correct guesses went:

| variant | correct hidden guesses in training | share landing in an all-same-grade group |
|---|---|---|
| per-character grader | 318 | 88.1% (280) |
| exact-match grader | 611 | 87.7% (536) |
| exact-match grader, 16-answer groups | 111 | 97.3% (108) |

![Fig 2. Where every correct hidden-digit guess landed during the full 300-step run, for each of the three variants. Each bar splits the variant's correct guesses into the share that landed in a group where every answer got the same grade, which carries zero advantage, and the share that landed in a group with a spread of grades, which the update can act on. The per-character grader reads 88.1% tied of 318 guesses, the exact-match grader 87.7% of 611, and the 16-answer variant 97.3% of 111.](/assets/images/writing/invented-convention-pilot/fig2_hit_landing.png)

The rest landed in a group whose answers did not all score the same, 38 of 318, 75 of 611, and
3 of 111, and those are the ones the update could act on. The mechanism that could teach the
hidden mapping is real. It is just rare, and the rarity is doing the damage.

Doubling the group from 8 to 16 answers is the fix most people reach for, and it did not help:
97.3% tied, the highest of the three, with twice as many answers per group to break a tie with.
One reading of that points back at the narrowing described above. Once a distribution has
consolidated onto a few candidates, right or wrong, sampling more answers from it mostly
returns more copies of the same guess, not more disagreement. That narrowing was measured on the
exact-match variant, so carrying the explanation over to the 16-answer variant is an inference
from its tied share rather than a separate measurement on it.

The comparison has a second problem. The 16-answer variant produced a correct hidden digit 111
times across the run, against the exact-match grader's 611 at the same total compute, a factor
of 5.5. Its 97.3% is a share of those 111 events, and the two variants differ in how often the
right answer showed up at all, well before any question of which group it landed in. Group size
is not the only thing that changed between them.

Groups with no spread in their scores are a known problem in this family of methods. DAPO
([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) resamples during training so that such groups are filtered out of the batch
before the update. What this experiment adds is a count, on a sub-skill that never learns, of
how often the correct answer itself was sitting inside a group with no spread at all.

## The diagnostic: two numbers, read together

If you are training with a GRPO-family method and one sub-skill refuses to move while your
headline metric climbs, there is a check you can run on logs you probably already store. You
need the per-answer scores, grouped as they were scored, plus some way to tell which answers
were correct on the sub-skill you care about. Then:

1. Filter to the prompts that exercise the frozen sub-skill.
2. For each group, compute the spread of scores inside it. A group where every answer scored
   the same contributes nothing to the update, whatever those answers say.
3. Among only the correct or high-scoring answers, count the share that sat in one of those
   all-same-grade groups.

The three variants here came out at 88.1%, 87.7%, and 97.3%, and none of them learned the
frozen mapping. The tempting next step is to treat a high number as the alarm on its own. Run
the same count on the channel that did learn and that reading falls apart.

The 8 revealed mappings are the control. That is the skill the model acquired, from 6.1% of
revealed positions correct on held-out strings at step 0 to 94 to 100% at step 300. Its correct
training draws landed in all-same-grade groups just as often:

| variant | frozen channel, correct draws in all-same-grade groups | learned channel, same count |
|---|---|---|
| per-character grader | 88.1% (280 of 318) | 91.8% (12,528 of 13,641) |
| exact-match grader | 87.7% (536 of 611) | 91.9% (11,919 of 12,966) |
| exact-match grader, 16-answer groups | 97.3% (108 of 111) | 90.1% (11,354 of 12,603) |

Two of the three read higher on the skill that worked. The matching shares are not a flaw in the
count. They are what learning looks like from inside a group. A skill the model has acquired
converges to groups
where every answer is right, and a group where every answer is right is a group where every
answer scores the same. Those ties carry exactly zero advantage for the same arithmetic reason
as the others, and here that is the correct outcome: nothing is left to fix. A frozen skill
converges on ties from the opposite direction, every answer wrong. The count alone cannot tell
the two apart, because both of them are ties.

What tells them apart is what the held-out score on that sub-skill is doing while the ties pile
up. Take the exact-match grader over its last 25 training steps. Ties were the normal case in
both channels by then: 96.1% of its correct revealed draws and 85.7% of its correct hidden draws
sat in a group with no spread. The difference is everything around those two numbers. Its
held-out revealed accuracy had gone from 6.1% to 93.9% over the run. Its held-out hidden
accuracy read 0 of 11 slots at step 0 and 1 of 11 at step 300.

So the reading is a pair, not a number. Ties in a sub-skill whose held-out score has climbed are
convergence, and there is nothing to fix. Ties in a sub-skill whose held-out score never left
the floor are the reason it never left. The two numbers come from different places and you have
to keep them straight: the tied share is computed on training draws at sampling temperature,
the score on held-out greedy decoding. Neither one means anything without the other.

If your frozen sub-skill shows that pair, it is not failing because the model cannot produce the
right answer. It is failing because the right answer arrives in a group that cannot express a
preference.

A richer grader and a bigger group are the usual first moves, and both got a fair shot here, on
the same task and budget, without moving the frozen mapping. The first thing to reach for
instead is dynamic sampling, DAPO's name for the resampling described earlier: keep drawing
answers for a prompt until its group holds both a right answer and a wrong one, and drop the
prompt from the batch if it never does. Every group that survives can then express a preference,
which is exactly what the groups around the correct hidden guesses here could not, with the boundary
that a prompt the model almost never gets right is dropped rather than taught. Beyond that, what
is worth trying is anything that widens the spread of scores inside a group on those prompts,
whether a grading scale with more distinguishable levels than your answers currently land on, or
restructuring which prompts share a group so that a group is not built entirely out of answers
the model finds equally easy. The companion piece on reward shopping walks through the same
zero-spread mechanism on a language task with a trained reward model.

## Scope

One seed, one toy task, one model, one setting (8 of 10 mappings revealed, two-digit inputs),
about 4.9 hours of local machine time across the three variants (2.0, 1.9, and 1.1 hours), no
paid compute. The freeze is demonstrated on this task and nothing wider, and 11 held-out slots
behind 2 mappings is a small measurement. The transferable part is not the result, it is the
diagnostic: the tied-group share is cheap to compute on any GRPO-family run, and read next to
the sub-skill's own held-out score it separates "the model cannot do this" from "the model did
it and the update discarded it."
