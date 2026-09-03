---
title: "12 of 15 Published Margins Are Smaller Than Their Own Benchmark Can See"
description: "What a benchmark's size lets it show, and why 12 of 15 recent RL papers' headline margins fall below it."
tags: [paper, reward-models, evaluation]
date: 2026-07-28 23:00:00 +0000
---

## What this paper shows

- Every benchmark has a smallest gap it can reliably show, and that gap is
  computable from two numbers a paper usually prints, the question count and the
  baseline accuracy. Section 1 builds this from scratch, because the habit of
  asking for it is not yet standard anywhere.
- We computed that smallest-visible gap for every recent RL-for-reasoning paper
  in our corpus whose headline is a single-benchmark margin. All 15 scoreable
  claims could be scored, several only after digging the missing numbers out of
  the papers' own tables and code releases. 12 of the 15 margins are smaller
  than what their own benchmark can reliably show. The median margin is a fifth
  of its floor.
- The 3 margins that clear their floor tell one story between them. Two are
  landslides from switching RL on at all, 46.8 and 55.4 points, big enough to
  survive a 30-question benchmark. One is an ordinary-sized refinement that
  clears because it was measured on 600 questions instead of 30. What separates
  visible margins from invisible ones is how the margin compares to a floor the
  benchmark's size sets, not how impressive the method is.
- The same missing arithmetic appears upstream, where reward models, the graders
  inside RL training, get chosen. All 6 public reward-model releases we audited
  support "ours is better" with single point estimates. No seeds, no confidence
  intervals, on the headline claim.
- Two failure mechanisms measured on real training runs show what that hides. A
  reward model carrying half its real information fought the full-strength
  version to an exact 108-to-108 draw on one of three otherwise-identical runs.
  And the answer to which of two reward models looks better at a given training
  checkpoint reversed sign up to 3 times over a run, with at most 1 of 6
  checkpoint readings separating from zero.
- A comparison discipline that avoids both failure modes costs single-digit
  dollars at the scale we ran it. Applied to our own two live reward-model
  comparisons, it reads both as ties, and a length audit inside it showed that
  one of our own strongest results was measuring response length, not reward
  quality.

## 1. Every benchmark has a smallest gap it can see, and its question count sets it

A benchmark is a measuring instrument, and like any instrument it can only show
differences down to a certain size. A kitchen scale that reads in whole grams
cannot settle an argument about half a gram, and no amount of confidence in the
person doing the weighing changes that. For a benchmark, the smallest showable
difference is set by how many questions it has.

A benchmark score moves in steps of one question. On a 30-question test, each
question is worth 3.3 points, so the score cannot move in units finer than that.
The score also wobbles. Ask the same model the same 30 questions on another day,
with another sampling seed, and it will get a couple of questions right that it
missed before, and miss a couple it got right. That ordinary wobble moves a
30-question score by several points in either direction without anything about
the model changing. A real difference between two models has to be bigger than
the wobble before the test can show it reliably, where reliably means the test
would show it again on a second run.

Statistics has a standard tool for exactly this, built for drug trials and
opinion surveys. It is called the minimum detectable effect, and it is the
smallest true difference a test of a given size can be counted on to find.
Computing it takes two numbers, both usually printed in a paper: how many
questions the test has, N, and roughly how often the baseline gets them right,
p. Those two give the score's natural wobble, and the convention multiplies that
wobble by 2.80. The 2.80 is two standard terms added together. The 1.96 keeps
the chance of a false alarm at 5 percent, and the 0.84 on top buys an 80
percent chance that a real gap of this size shows up again on a rerun.

    smallest reliable gap = 2.80 x sqrt(2 p (1 - p) / N)

This paper calls that number the detection floor. It is a property of the test,
not of the models being tested.

The floor is a reliability bar, not a significance bar, and the two are
different heights because they are built from different pieces of that 2.80.
Significance asks whether the gap you measured once was probably not luck, and
that costs only the 1.96. The floor asks more, that a gap of this size would
also show up again on a second run, four times out of five, and that costs the
full 2.80. Dividing the two gives the point where the bars cross, 1.96 / 2.80
= 0.70. A margin at 70 percent of its floor sits exactly at the significance
bar. Between 70 percent and the full floor, both readings are true at once,
and this is the confusing zone worth naming plainly. A margin there is
significant on the run that produced it and still too small to count on seeing
twice. Below 70 percent it fails both bars. The floor is also easy to compute
for someone else's result. It does not need their model, their code, or their
compute, only the two numbers already printed in their paper.

Now apply it to one real published result. DCPO reports beating its closest
baseline by 6.7 points on AIME24, from a baseline accuracy of 40 percent.
AIME24 has 30 questions, so p = 0.4 and N = 30, and the formula gives a floor
of 35.4 points. The reported margin is 6.7, which is 0.19 of the smallest gap
this test can reliably show. In question terms, the margin is about two
questions, on a test whose ordinary wobble is bigger than two questions.

That number does not say the margin is fake. The improvement may be real. It
says something narrower. The instrument that produced this number was never
capable of reliably showing a difference this small, so if the same comparison
were run again, the odds of seeing this margin again are poor, and that is true
even if the method genuinely works. From this test alone, there is no way to
tell a result like this apart from luck.

The rest of this paper does three things with that one idea. It runs the same
two-number check on every recent paper in our corpus that makes this kind of
claim. It looks at where reward models, the graders inside RL training, get
chosen, and finds the same arithmetic missing there. And it lays out a cheaper
way to run these comparisons, with what each step costs, including what that
way showed about two comparisons of our own.

## 2. The same check on 15 recent papers: 12 margins are below their own floor

Of 30 recent RL-for-reasoning papers in our corpus, 16 lead with a
single-benchmark margin rather than a cross-benchmark average or a ratio such
as a data-efficiency multiplier. One of the 16, REINFORCE++, is set aside for a
reason section 2.1 explains, and it is a reason in the paper's favor. The other
15 could all be scored. For 10 of them the two needed numbers were stated
outright or settled by public knowledge of the benchmark, the way AIME24 is
known to have 30 questions and the GSM8K test set 1,319. For the other 5 the
numbers were recoverable with more digging, from the papers' own results
tables, from an official code release, or from a benchmark's released files,
and section 2.1 records where each recovered number came from. Nothing in this
table rests on a guess.

| Paper | Benchmark | Margin (pp) | Baseline acc | N | N source | Floor (pp) | Margin / floor |
|---|---|---|---|---|---|---|---|
| LongRLVR | RULER-QA (14B) | 15.73 | 0.732 | 600 | released predictions | 7.17 | 2.19 |
| DeepSeek-R1 | AIME 2024 | 55.4 | 0.156 | 30 | known | 26.25 | 2.11 |
| Magistral | AIME-24 | 46.8 | 0.268 | 30 | known | 32.04 | 1.46 |
| Tulu 3 (RLVR, 8B) | GSM8K | 3.3 | 0.843 | 1,319 | known | 3.97 | 0.83 |
| NSR | MATH (Pass@256) | 0.6 | 0.963 | 5,000 | official eval config | 1.06 | 0.57 |
| VAPO | AIME 2024 | 10.4 | 0.500 | 30 | known | 36.17 | 0.29 |
| Beyond the 80/20 Rule | AIME'24 | 7.71 | 0.558 | 30 | known | 35.92 | 0.21 |
| DQO | GSM8K (greedy) | 1.05 | 0.635 | 1,319 | known | 5.25 | 0.20 |
| DCPO | AIME24 | 6.7 | 0.400 | 30 | stated | 35.44 | 0.19 |
| AGPO | MATH-500 | 1.4 | 0.732 | 500 | known | 7.85 | 0.18 |
| DAPO | AIME 2024 | 3.0 | 0.470 | 30 | known | 36.10 | 0.08 |
| Entropy Mechanism (KL-Cov, 7B) | AIME24 | 1.4 | 0.212 | 30 | known | 29.57 | 0.05 |
| Light-R1 | AIME24 | 1.7 | 0.723 | 30 | known | 32.37 | 0.05 |
| Open-Reasoner-Zero | AIME2024 | 1.1 | 0.470 | 30 | known | 36.10 | 0.03 |
| Skywork-OR1 | AIME24 | 0.8 | 0.814 | 30 | known | 28.15 | 0.03 |

*Floors are computed with the unrounded constant, 1.9600 + 0.8416 = 2.8016.
Section 1 shows 2.80 rounded for display. Margins follow one convention
throughout, the paper's headline number against its closest baseline method.
For VAPO the table prints integers while its abstract carries the decimal, so
the margin is 10.4 against DAPO's 50. Full-precision figures ship in the
companion repository, whose floor calculator recomputes every row and every
figure from the recorded inputs.*

![Horizontal lollipop chart, headline margin divided by detection floor, one row per paper, sorted descending, with a ratio-equals-1 reference line. Three rows clear their floor: LongRLVR at 2.19, DeepSeek-R1 at 2.11, and Magistral at 1.46. Twelve rows sit below the line, from Tulu 3 at 0.83 down through NSR at 0.57, VAPO at 0.29, Beyond the 80/20 Rule at 0.21, DQO at 0.20, DCPO at 0.19, AGPO at 0.18, DAPO at 0.08, the Entropy Mechanism and Light-R1 at 0.05, and Open-Reasoner-Zero and Skywork-OR1 at 0.03.](/assets/images/writing/reward-shopping-audit/fig_margin_vs_floor.png)

12 of the 15 margins sit below their own floor. The median margin is 0.20 of
its floor, a fifth of the smallest gap its own benchmark could reliably show.

The 3 rows that clear their floor say more together than any of them says
alone. DeepSeek-R1 and Magistral are landslides, 55.4 and 46.8 points, the
one-time leap from switching RL on at all. On a 30-question benchmark, where
floors run 26 to 36 points, only a landslide can clear. LongRLVR is the
opposite case and the more instructive one. Its 15.7-point margin is an
ordinary-sized refinement, one method improving on another, and it clears at
2.2 times its floor because its benchmark has 600 scored items, which puts the
floor at 7.2 points instead of 36. That reading survives a worst-case check.
If the benchmark's three context lengths secretly reuse the same 200
underlying questions, the honest N drops to 200 and the ratio drops to 1.27,
still clear of its floor. So the dividing line in this table is not how
impressive the method is. It is whether the margin is large relative to a
floor the benchmark's size sets. A big margin clears a small test, a modest
margin can clear a big test, and a small margin clears nothing. Size alone
rescues no one either. NSR was measured on 5,000 questions, the largest N in
the table, and still sits at 0.57 of its floor, because its margin over its
closest trained baseline is 0.6 points against a 1.1-point floor. One more
fact belongs to that row. At this metric the untrained base model scores 96.9,
the same as NSR itself, so the gain over the base model at this particular
endpoint is zero, and the 0.6 is its edge over PPO, not over doing nothing.

A margin computed from an averaged metric needs one caveat stated once. Most
AIME24 rows report an average over multiple sampled attempts per question,
written avg@32 or similar. Averaging over attempts shrinks the sampling part
of the wobble, so the true floor for those rows sits somewhat below the
computed one, bounded from below by how much questions genuinely differ. The
computed floors are therefore upper bounds and the ratios lower bounds, and
the caveat applies equally to every AIME24 row in the table, the old ones and
the recovered ones alike. It does not move any row across the line, because
the below-floor AIME24 rows sit at 0.29 of their floor or lower, and
question-to-question difficulty differences, which averaging cannot remove,
keep real AIME floors far above margins of a point or two.

The nearest prior statement of this problem is Hochlehnert et al.
([arXiv:2504.07086](https://arxiv.org/abs/2504.07086)), who make the small-benchmark argument for AIME directly
and first. Re-running released checkpoints under standardized decoding, they
find that one question moves pass@1 by 2.5 to 3.3 points on benchmarks of the
size they study, that pass@1 standard deviations across random seeds run as
high as 15 points on AIME'24, and that the improvements recent methods report
fall inside that variance. Ten of the fifteen rows above are AIME24 rows, so
their argument covers most of this table and the priority is theirs. Their
evidence needs a released checkpoint and the compute to re-run it. The
two-number check needs neither, which makes it available for releases with no
public checkpoint and for benchmarks nobody has re-evaluated.

### 2.1 Recovered numbers, and the one paper set aside

An earlier version of this census could score only 10 of the 16 papers and
excluded the rest for missing numbers. A challenge to go verify the exclusions
at the sources dissolved all of them, and the corrections belong on the
record. Three papers said to omit their baseline accuracy state it in their
own results tables. DQO's table gives the DPO baseline at 63.46 percent
against its 64.51, exactly the 1.05-point margin its text claims. Beyond the
80/20 Rule's table gives 55.83 against 63.54. VAPO's abstract and table give
60.4 against DAPO's 50. Two more needed the authors' released artifacts
rather than the paper text. NSR's "MATH" turns out to mean the full
5,000-question test set, settled by the dataset its official evaluation
script loads. LongRLVR's evaluation size is 600 items, settled by the
prediction files shipped with its release, 100 per task per context length.
Recovering numbers this way cuts against the paper's own convenience, since
every recovered row was a chance for the census pattern to break, and one
recovered row, LongRLVR, did break the earlier version's cleanest
generalization and forced the size-sets-the-line reading above.

The one paper set aside is REINFORCE++, and it is set aside because it makes
no margin claim to score. Its own text reports 46.7 against GRPO's 46.8 and
calls the result a tie, which is the correct reading of a 0.1-point gap on
that benchmark. One disclosure point still belongs here. The benchmark it
used, Arena-Hard-Auto, ships bootstrap confidence intervals in its standard
tooling, and the paper reports the scores without them. A reader deserves the
interval a benchmark's own harness would have printed, most of all when the
honest conclusion is a tie. The paper also spans nine arXiv versions in which
this evaluation appears from the second year onward, so any citation of those
numbers should pin the version it read.

### 2.2 The other number nobody reports

The floor needs N and p. An error bar needs a third number, how many training
seeds the headline run used and how much the result moved between them. A
companion census over the same 30-paper corpus checked for exactly that. One
of the 30 has no headline claim to score, Kimi K2 ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)), which
reports overall model capability against other released models rather than a
method-versus-baseline comparison. Of the 29 that remain, 27 report no
training-seed count at all for their headline result, 93 percent of the
corpus, and only 2 pair a seed count with its own between-seed spread. Put
the two censuses together and a reader of this literature usually cannot
reconstruct an error bar in either direction, not across questions, because N
goes unstated, and not across runs, because the seeds do. That is a statement
about what the documents contain, not about anyone's intent.

## 3. The models grading your training are chosen the same blind way: 6 of 6 releases, one number, no error bars

The check so far scores papers about training methods. The same question can be
asked one level up. Reward models, the trained graders that steer RL training
wherever no automatic checker exists, are themselves compared, ranked, and
chosen, and the release documents that support those choices can be read with
the same two questions in mind. How many runs, and where is the error bar.

We read the evaluation sections of six public reward-model releases from 2024
and 2025: Skywork-Reward-V2, the Tulu/RewardBench 2 line, GRM,
Nemotron-4-340B-Reward with HelpSteer2, ArmoRM, and INF-ORM-Llama3.1-70B. For
each we recorded which benchmarks it reports, how many seeds, what training
horizon, and whether variance or a confidence interval appears anywhere near
the headline number.

| Release | Headline practice | Downstream (RLHF/BoN) validation | Seeds reported | Variance/CI reported |
|---|---|---|---|---|
| Skywork-Reward-V2 ([arXiv:2507.01352](https://arxiv.org/abs/2507.01352)) | 7 to 8 benchmarks, point estimates | Best-of-N only. No full RLHF run in main text | None | No. A 0.6-point ablation spread exists, unattached to the leaderboard claim |
| AI2 Tulu / RewardBench 2 ([arXiv:2506.01937](https://arxiv.org/abs/2506.01937)) | RewardBench 2, 6 domains, point estimates | Most extensive. 113 best-of-N, 17 PPO vs Tulu 3 | None stated per configuration | No, but one saturation caveat volunteered (quoted below) |
| GRM ([arXiv:2406.10216](https://arxiv.org/abs/2406.10216)) | RewardBench + 2 out-of-distribution sets, point estimates | Yes. Full PPO trajectory vs. gemma-2b-it | 1 per trajectory | No confidence bands on the curves |
| Nemotron-4-340B-Reward / HelpSteer2 ([arXiv:2406.11704](https://arxiv.org/abs/2406.11704) / [2406.08673](https://arxiv.org/abs/2406.08673)) | RewardBench only, overall plus per-category scores, all point estimates | None isolated | None | No |
| ArmoRM ([arXiv:2406.12845](https://arxiv.org/abs/2406.12845)) | RewardBench only, overall plus per-category scores, all point estimates | None | None | No |
| INF-ORM-Llama3.1-70B (HuggingFace model card) | RewardBench only, single dated-snapshot number | None | None | No |

Across all six, the pattern is one pattern. Every release supports its "ours
is better" claim with point-estimate scores on a static benchmark, headlined
by a single overall number, and none reports a standard deviation, a
confidence interval, or a multi-seed replication for that headline number.
Several break the benchmark out by category, which is more disclosure than a
bare overall score and is still more point estimates rather than an error
bar. Where a lab's own data does surface run-to-run variation, the variation
stays out of the headline. Skywork's own ablation measures a 0.6-point spread
across data subsets and never attaches it to the seven-benchmark leaderboard
table. ArmoRM calls its 0.3-point gap over Nemotron-4-340B-Reward "nearly on
par" with no uncertainty estimate in either direction, and those two numbers
sit closer together than any variance either lab discloses for its own
measurement.

Downstream validation, actually training a policy with the reward model and
looking at what comes out, is rare, and where it exists it is real. AI2
trains and evaluates 113 best-of-N and 17 PPO configurations against actual
Tulu 3 policies. GRM plots reward divergence across a full PPO training
horizon. Both engage with the downstream question in a way the other four
papers do not attempt. Nemotron-4-340B-Reward is not separately validated at
all. Its reward model is folded into a training pipeline and judged only
through the trained policy's aggregate scores.

AI2's disclosure goes furthest, and one sentence of it deserves quoting in
full, because it is the only place across the six releases where a lab
volunteers that its own evidence may not distinguish between the reward
models its leaderboard ranks:

> "PPO performance quickly saturates to a similarly good performance matching that of Tulu 3
> 8B DPO for all decent-to-good reward models whose RewardBench 2 scores range from 49.8 to
> 68.5."

Read plainly, that sentence says downstream policy quality stops responding
to reward-model differences above some quality bar, which would mean the
leaderboard differences among good reward models are not the thing driving
downstream outcomes. Even this, the cleanest practice of the six, states no
seed counts per configuration and picks its reported checkpoint after the
fact rather than fixing a horizon in advance. GRM earns credit for a
different reason. It is the only release that treats the training horizon as
an object of study, plotting a whole trajectory rather than an endpoint, and
it does so from a single seed. The one paper that shows a trajectory at all
is also the clearest illustration of why seed count and horizon need
attention together. Section 4 measures exactly that, with real training runs
and real reward models.

## 4. Two ways a cheap comparison lies: an unlucky seed can erase a real gap, and an early reading does not predict the end

Two training runs can differ for three mundane reasons before any interesting
one: the data order the seed sets, the weight initialization, and the random
draws behind each sampled rollout. A gap between two runs says nothing about
the thing being compared until all three are ruled out. The seed-masking runs
below rule out all three between the compared variants. Both variants draw
their per-token sampling randomness from the same keyed noise stream, replayed
bit for bit, so at the first training step both variants sample identical
token sequences, checked directly on the machine before any scored run
launched. From that point on, a gap between them has one remaining
explanation, the reward signal. The horizon-flip runs are different and the
difference limits how hard they can be read. They were trained earlier in
this project, before that coupling existed, and the question asked of them is
not which variant won but whether an early reading points the same way the
converged reading does, both taken from the same recorded histories.

### An unlucky seed can erase a real, known difference

We built a reward model that is measurably worse than a real one, rather than
guessing at how a weak model would behave. It blends GRM-Gemma2-2.6B's own
scores half-and-half with a within-group shuffled copy of themselves, which
destroys half the information while keeping the score's scale and texture.
This half-signal variant trained against the full-strength GRM as the other
variant, across three training seeds, judged blind on held-out prompts by two
reward models that trained neither variant. The ground truth is known by
construction. The full-strength variant should win, because half the signal
it carries was deliberately destroyed in the other.

![Grouped bar chart, three training seeds. Seed 1: full-strength reward model 184 wins to half-signal reward model's 53 (p = 3.96e-18). Seed 2: 108 wins to 108 wins, an exact tie (p = 1.0), annotated "known real difference, statistically invisible on this draw." Seed 3: 200 wins to 36 (p = 9.44e-29). All bars are the cross-check judge's raw vote.](/assets/images/writing/reward-shopping-audit/fig1_seed_masking.png)

Two of the three seeds detect the difference overwhelmingly. Under the raw
judge vote, seed 1 favors the full-strength variant 184 wins to 53, at p =
3.96 x 10⁻¹⁸ on the Mistral judge. Seed 3 favors it 200 to 36 on the same
judge, at p = 9.44 x 10⁻²⁹, and 202 to 41 on the other. Seed 2 shows nothing
at all. One judge's tally lands at exactly 108 wins apiece, a dead heat, p =
1.0. The other judge reads the same seed 106 to 109, leaning the wrong way.
This is an engineered, known information deficit, and on this one seed
neither of two independent judges can see it even at the level of raw
preference counts. Under the length-corrected primary scoring rule that
section 5 introduces, the picture does not improve. Seed 2 is the one seed
whose sign flips against the other two on both judges. The pooled reading
across all three seeds stays significant in the correct direction, at p =
0.008 to 0.003 depending on judge, and the seeds do not all agree with it,
so under the two-part rule defined in section 5 no winner is called.

The instrument is not blind. Two of three seeds detected a half-strength
reward model at significance levels that would be extraordinary in most
empirical settings. The finding is narrower and worse for common practice. A
single training seed is not a reliable unit of evidence for a reward-model
comparison, because the seed-to-seed variance in how a comparison plays out
is large enough for one unlucky draw to erase a real, substantial, known
difference. A lab that reports "we compared reward model A to reward model B
and A won," the practice every release in section 3 follows, has a meaningful
chance of reporting a property of its seed rather than a property of the two
reward models, and which seed a lab happened to run is not disclosed in any
of the six releases.

### An early reading does not reliably predict where a run ends up

The training horizon is a second, independent axis on which a single reading
can mislead. We used recorded evaluation histories from three training
variants on a checkable task, three seeds each, with a real early-stopping
rule, so that "converged" and "peak" are both well-defined per run. The
variants are a checker-only variant, rewarded by GRPO's binary right/wrong
verifier, a full-reward-model variant, rewarded by a dense reward model on
every rollout, and a localized-reward-model variant that applies the dense
model only to break ties the checker cannot. The question asked is
descriptive. If a comparison between two variants had been read at an
intermediate checkpoint, would it point the same way the converged reading
does. The converged reading is itself a point estimate rather than a ground
truth, and the one converged comparison that was properly tested reads as a
statistical tie at p = 0.33, so agreement here means two point estimates
sharing a sign, nothing stronger.

![Two rows of six markers, one marker per training checkpoint from step 25 to step 150. Each marker scores whether that checkpoint's reading has the same sign as the same pairing's converged point estimate, which is itself a tie under the proper test rather than a known correct answer. The top row shows the full-reward-model variant against the checker-only variant. It matches the converged sign at steps 25, 75, and 100, and takes the opposite sign at steps 50, 125, and 150. The bottom row shows the localized-reward-model variant against the checker-only variant. It takes the opposite sign at steps 25, 50, 75, and 150, lands on an exact tie at step 125, and matches at step 100.](/assets/images/writing/reward-shopping-audit/fig2_horizon_flip.png)

Across six checkpoints spanning steps 25 through 150, the answer is no for
two of the three pairings. The full-reward-model variant against the
checker-only variant flips sign three times across the six checkpoints and
never settles into the direction it eventually converges to. The point
estimates show the full-reward-model variant with a transient mid-training
lead of 1.8 to 3.8 points across the three seeds, read at each variant's own
peak. That lead does not clear the rule, landing at p = 0.196 pooled under the same
rule used everywhere else in this paper. By convergence the point estimates
lean the other way, and the three seeds split in sign. The
localized-reward-model variant against the checker-only variant matches its
converged sign at exactly one of the six checkpoints, on a gap of 0.13
points, too small to separate from zero. The third pairing, localized
against full, is the one that partly settles. It lands on the converged side
at three of six checkpoints including both of the last two, and it still
points the wrong way at three of the first four.

A follow-up interval analysis on the same histories makes the practical
reading blunter. With a 95 percent confidence interval on each checkpoint
comparison, at most one of the six checkpoints in the window separates from
zero for any pairing, and no pairing shows two significant readings with
opposite signs. The sign flips are point estimates wobbling inside their own
intervals. That hardens the finding rather than softening it. A pilot
reading a single checkpoint in this window is not reading an unstable truth.
It is reading noise that happens to wear a sign. The one checkpoint that
does separate cleanly is also the most misleading one. At step 150, pooled
across seeds, the full-reward-model variant leads by 2.9 points, interval
0.4 to 5.5. By convergence that lead is gone, and the properly tested
endpoint comparison is a tie.

One tempting fix does not work. Reading a summary of the trajectory up to a
horizon, rather than a single point on it, was tested directly against the
same data. The best of four trajectory summaries tried, a linear slope over
recent checkpoints, beats a plain point-in-time reading in exactly one of
fifteen pair-by-horizon comparisons, which is not an improvement at that
sample size. A cumulative running average gets the eventual winner's sign
wrong at all fifteen, for a diagnosable reason. It systematically punishes a
variant that starts slowest and finishes best, which is exactly the variant
that wins this comparison. If truncating a run early creates this much risk,
the fix is not a cleverer way to read a truncated run. It is not truncating
it, and that choice is where the next section starts.

## 5. The fix costs single-digit dollars: six steps, each with its cost

If a single seed and a single horizon are each independently unreliable, the
response is not a cleverer way to read one seed at one horizon. It is a
design that does not depend on either being reliable alone. That design
follows, as six numbered steps with what each one costs.

![Six-box pipeline diagram in a two-row grid. Top row, left to right: Step 1, Couple and judge blind. Step 2, Race to convergence. Step 3, Seeds first, to a point. An arrow drops from Step 3 straight down to Step 4, positioned below it in the bottom row. Bottom row, right to left, continuing the flow: Step 4, Pass a known null, then re-check it, shown in red as the one step whose first pass was revised after a length confound. Step 5, Watch the gap, not the hack. Step 6, Date the flag, on the far left. Reading the bottom row right to left keeps the sequence 4 to 5 to 6 unbroken.](/assets/images/writing/reward-shopping-audit/fig6_protocol_pipeline.png)

| Step | What it does | What it costs |
|---|---|---|
| 1. Couple and judge blind | Two training variants share weight init, data order, and (until they diverge) token draws. Judged blind by two reward models with no training lineage to either variant. | About $13 to $29 per live comparison, training and blind judging together |
| 2. Race to convergence | Each seed reads at its own converged endpoint, never a fixed horizon. | Nothing beyond step 1. It is a stopping-rule choice |
| 3. Seeds first, to a point | The next dollar buys a seed up to the third seed, and evaluation prompts after that. A second judge comes last under either goal. | $3.50 per seed, $0.50 per 100 prompts, $0.30 for a second judge |
| 4. Pass a known null, then re-check it | A known-null control must pass, and then gets re-checked for nuisance axes like length before it is trusted. | $19.39 for the plant control, $0 for the re-analysis |
| 5. Watch the gap, not the hack | A reward score that climbs while an independent read does not is the alarm, whatever is causing it. | $0, a re-analysis of already-recorded drift data |
| 6. Date the flag | Every headline margin's floor ratio (section 2) is computed and dated before any re-evaluation exists, then scored against whatever lands later. | $0, reading only |

**Step 1: couple two training variants and judge them blind.** The task is
the non-verifiable slice of an instruction dataset, prompts with no checkable
answer, 4,666 items at a pinned revision. One deterministic construction,
fixed before any run, takes 500 items as the training pool and a disjoint
block as the held-out evaluation pool. The first live comparison in section 6
read on a 200-item evaluation pool. The second read on a 325-item pool built
by the same construction with a larger cut, whose first 200 items are
byte-identical to the earlier pool. The policy is Qwen2.5-1.5B-Instruct,
trained by LoRA under GRPO with eight rollouts per prompt. Within a training
seed, the two compared variants share the same adapter initialization and the
same per-step data order, and both variants' token sampling is driven by the
same fixed pseudo-random draw, computed per prompt, step, sample, and decode
position. The trick is borrowed from communication-free noise coupling built
for speculative decoding (Daliri, Musco, and Suresh, [arXiv:2408.07978](https://arxiv.org/abs/2408.07978)). At
the first training step, before the weights have diverged, the two variants'
sampled token sequences are bit-for-bit identical, checked directly on the
machine. Any difference downstream of step one is attributable to the
reward, not to rollout luck.

Before training, all four scoring models in the pipeline, the two reward
models being compared and the two judges, are measured for their own
repeat-scoring noise. Each of 64 held-out items is scored five times, and 3
times the 95th-percentile spread across those scorings becomes that scorer's
noise floor. GRM-Gemma2-2.6B's floor is 0.0393. Skywork-8B's is 0.415, 10.5
times larger. The two judges land at 0.274 and 0.281. Any score gap smaller
than a scorer's own floor is treated as a tie everywhere downstream. Two
notes belong to that measurement. The 3x multiplier is a disclosed design
constant written into the spec before any floor was measured, deliberately
generous so the band errs toward calling a real difference a tie. And a
reward model is a deterministic function of its input, so a spread across
repeat scorings of the same text needs a mechanism. The mechanism is batch
composition. The five scorings differ only in what else shares the batch,
and changing the batch changes its padded shape and the order in which
floating-point additions happen on the accelerator. Those additions are not
associative, so the same text scored beside different neighbors comes back
with slightly different scores. This is the known explanation for the
effect, not something this project discovered. The protocol measures its
size by scoring each item alone and then at four positions inside an
eight-sequence batch.

At read time, both variants generate greedy responses on the held-out
prompts, pooled with the untrained baseline's own responses, shuffled, and
scored by two reward models that trained neither variant and share no
training lineage with either one, the contamination risk Preference Leakage
names directly ([arXiv:2502.01534](https://arxiv.org/abs/2502.01534)). Variant identity is never shown to either
judge. One judge is primary and the other is a cross-check, and a result the
cross-check disagrees with is reported as a disagreement, not a finding. The
verdict rule works on prompts, not scores. Within a seed, a prompt counts as
a win for whichever variant its judge scored higher, unless the two scores
fall inside the judge's own floor, in which case the prompt is a tie and
drops out. Each prompt then votes once, by majority across the seeds. Two
conditions must hold together for a comparison to count as a called win. The pooled
per-prompt majority vote must be significant on an exact two-sided binomial
test, and every seed's own sign must agree with the majority. That pair of
conditions is the two-part rule, and every result in this paper is held
to it. Section 4's seed-masking result shows the two conditions doing their
jobs. The pooled vote there is significant at p = 0.008, so the first
condition is met, and seed 2's own sign points against the majority, so the
second fails and no winner is called.

**Step 2: race each variant to its own convergence, never a fixed horizon.**
The horizon findings in section 4 force this choice. Each seed reads at its
own converged endpoint, whatever the step count. Running the comparison
small already saves roughly 50 times the parameters of a production
fine-tune. Stopping early would save a further 2.5 times and is exactly
where the horizon risk lives, so this design gives that discount back. A
worked check on the section 4 data shows the stakes, using the same three
variants. Reading the three seeds' endpoints on 2,000 questions, the
full-reward-model variant and the checker-only variant are statistically
tied, p = 0.33 pooled. Reading them at a shared mid-training checkpoint on
the original 500 questions instead, the full-reward-model variant leads by
2.0 to 4.6 points at pooled p = 0.025, a lead it gives back by convergence.
Selecting a checkpoint by validation accuracy on the original 500, a rule
fixed before 1,500 fresh questions were scored, and then reading only the
fresh 1,500, the full-reward-model variant wins 211 of 365 separating
questions at pooled p = 0.0033, all three seeds agreeing. Reading each
variant's final, unselected checkpoint on the same fresh 1,500 instead, it
trails by 0.8 to 1.3 points with all three seeds agreeing on the losing
side, p = 0.36. Same runs, four defensible readings, three different
stories. On a checkable task the checker itself finds the peak for free. On
a task with no checker, nothing that cheap exists, because the reward
model's own training score keeps climbing after real accuracy has turned
over, so finding the peak needs a second, independent signal, and paying
for that signal is the expense a reward model was supposed to remove.

**Step 3: spend the next dollar on a seed, up to the third seed, and on
prompts after that.** The pooled verdict's uncertainty splits into three
measured parts, seed-to-seed variation, per-prompt sampling noise, and
judge-scoring jitter. Across four independent readings of that split, the
seed part runs roughly 4 to 30 times larger than the prompt part and orders
of magnitude larger than the judge part. The judge part is small partly
because the two judges are not very independent. Their score differences
correlate at r = 0.91 to 0.92, which makes two judges count as about 1.05
independent opinions, and it means the cross-check this protocol leans on is
weaker corroboration than "two judges agreed" sounds. In money, one more
training seed costs about $3.50 at this scale and buys roughly 5 times the
uncertainty reduction that $0.50 of additional prompts buys. A second
cross-check judge costs about $0.30 and buys almost nothing, for the same
correlation reason.

![Bar chart, three bars. Before spending the $5, the pooled verdict's standard error is 7.25 percentage points. Spending the $5 on one more seed plus 300 more prompts tightens it to 6.24 percentage points. Spending the same $5 entirely on 1,000 more prompts, with no new seed, leaves it at 7.11 percentage points, barely moved.](/assets/images/writing/reward-shopping-audit/fig3_seeds_first_budget.png)

A worked $5 split makes it concrete. One more seed plus 300 more prompts
tightens the pooled standard error from 7.25 to 6.24 percentage points. The
same $5 spent entirely on 1,000 more prompts leaves it at 7.11, barely
moved. The rule has two regimes, and stating it as "seeds before everything"
would misdescribe this project's own numbers. While a comparison runs on one
or two seeds, the next dollar buys a seed, because the seed term dominates
and nothing else touches it. At three seeds the seed term stops paying.
Under the two-part rule, each added seed sharpens the pooled test and
simultaneously adds one more chance for a seed to point the other way, and
the two effects roughly cancel. The smallest win-rate gap the rule
reliably catches is 8.7 percentage points at 3 seeds and still 8.2 at 5, while doubling
the per-seed prompt count moves the same floor from 8.7 to 6.1. Seeds up to
three to know where you stand, prompts after that to nail down what you
found, and a second judge last under either goal, at this project's own
measured scale. One honest boundary. This ranking counts a second judge
only for noise reduction, not for its separate value in catching a shared
bias or contamination that a single judge cannot see by construction, and
step 5 shows a concrete case of that separate value. A projection of this
arithmetic was also written down and dated before the 1,500 extra questions
above were bought, at 0.990 for a 3-point gap and 0.833 for 2 points. The
realized numbers came out at 0.990 and 0.826.

**Step 4: pass a known-null control, then check what the passing grade is
measuring.** Two controls ran against the verdict machinery
before any live comparison. A positive control confirmed the judges detect a
known, large effect, each trained checkpoint against the untrained starting
policy, and it cleared all six pairings tested at primary-judge p-values
from 2.6 x 10⁻³² to 3.3 x 10⁻²². A plant control replaced one variant's
reward with a permutation of its own scores, which carries zero usable
information by construction, and the judges caught it just as decisively.
The pooled vote favored the honestly-scored variant 143 to 21 across 164
decided prompts, p = 1.7 x 10⁻²³, with the cross-check judge agreeing at 152
to 19, p = 5.9 x 10⁻²⁷. One note on these p-values. The code that first
computed them used an absolute floating-point tolerance that floored any
result below about 10⁻¹⁵, so the stored outputs understated how extreme the
strongest tests were. The counts are unaffected, and the values printed
here are recomputed exactly.

Then the passing grade itself got checked, and this is where the protocol
caught one of its own results. A practitioner's instinct says reward models
reward length, a bias documented for LLM judges generally (Zheng et al.,
[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)), and the instinct turned out substantially right. The
honestly-scored variant's responses grow to a mean of 121 to 134 tokens by
read time while the plant-scored variant's stay near 70, and both judges
carry a length-correlated scoring bias of similar size, r of roughly 0.36
to 0.47. Restricting the plant control to response pairs of matched length
collapses the result. At a 20 percent length band the vote no longer clears
significance, p = 0.418, down from 10⁻²³, on 38 decided prompts. At the
tightest band the direction reverses on 21 prompts, with about 86 percent
of pairs discarded by the matching, a weak reading on a small slice rather
than a finding. A regression that keeps every pair says the same thing more
precisely. Length alone explains 40 percent of the score-difference
variance, and the length-independent remainder is 0.309, standard error
0.086, which sits barely above the judges' own noise floor of 0.274. A
result that stood 21 to 25 orders of magnitude past significance survives,
once length is accounted for, only as a small, barely-above-floor residual.

Regressing length out is not automatically safe either, and this is the
part that stops the fix from being a reflex. On three settings with
checkable correctness, the reward model audited showed no positive length
preference at fixed correctness, and residualizing anyway hurt
truth-agreement in two of them, by 0.8 and 3.0 points overall and by 10.3
points in the worst condition, because in those settings short answers are
more often right, so subtracting length subtracts real signal. The rule
this forces: measure a judge's length slope at fixed quality wherever a
checker makes that possible, and residualize only where the slope is
positive. Where no checker exists the slope is a disclosed, unverifiable
quantity, not a license to residualize by default. This rule was written
after the first of section 6's two live comparisons had already run, so
only the second was scored under it from the start, and the same
length-driven mechanism turns out to explain the one anomalous seed in the
first comparison. Whenever a judge's score tracks some axis that is not the
thing being measured, any drift between two variants along that axis gets
converted into what looks like reward-quality signal. We call this
nuisance-axis amplification, and it is the reading most consistent with
two single-instance findings rather than an independently proven mechanism.

**Step 5: watch for a reward score that climbs while an independent read
does not.** Nobody needs to identify which failure mode a reward model has
fallen into before being warned something is wrong. Tested on recorded
training data from a checkable task: in a variant trained with a real
reward model in the loop, the reward model's own score on already-wrong
outputs drifts upward over training by 123 times its own measured noise
floor, and the same drift appears on outputs that were already correct.
That is correctness-blind score inflation, not a narrow, item-specific
hack. A matched control with no reward model in the loop shows no drift,
a point estimate near zero with an interval spanning zero. The alarm this
licenses is cheap. Watch the reward model's own score against an
independent, cheaper read, a checker where one exists or a held-out judge
where none does, over the course of a small race. A reward score that
climbs while the independent read does not is the signature, whatever the
underlying hack. Two limits, stated plainly. The signature has been
demonstrated for one judge family, and per that judge's own disclosed
rider, a Skywork-class judge is blind to this signal class, so the alarm
is mechanism-agnostic but not yet shown to be judge-agnostic. Diagnosing
which hack is happening takes more work. Detecting that one is happening
does not.

**Step 6: turn the floor arithmetic into a dated, falsifiable ledger.**
Given a release's stated margin and evaluation size, its margin-to-floor
ratio can be computed, dated, and published before any independent
re-evaluation exists, then scored against whatever re-evaluation lands
later. The kind of gap this catches is not hypothetical. Independent
re-runs of recent reasoning RL releases have measured seed-to-seed AIME24
standard deviations of 2.9 to 6.3 percentage points (Hochlehnert et al.,
[arXiv:2504.07086](https://arxiv.org/abs/2504.07086)), and a contamination-resistant benchmark variant cut
three separately released models' scores by 42 to 73 percent relative to
the original leaderboard numbers (VAR-MATH, [arXiv:2507.12885](https://arxiv.org/abs/2507.12885)). Among the
census papers, DCPO is the row this ledger can act on today, because its
evaluation size appears in its own text, which is to its credit and is the
only reason its margin can be put on the scale at all. Its 6.7 points on
30-question AIME24 sits against a 35.4-point floor computed from its own
disclosed numbers, inside the published seed-to-seed swings for comparable
systems, with no public re-evaluation and no released checkpoint that
could test it today. So this stands as a dated, falsifiable flag rather
than a verdict. As of July 2026, the prediction is that this margin does
not survive a seed-varied or larger-N re-evaluation at adequate power. Any
refutation, the authors' own per-seed numbers included, will be reported
as prominently as the flag.

## 6. The same check, turned on us: our two live comparisons land in ties

Sections 4 and 5 explain how a single-seed, single-horizon comparison can
report the wrong answer, and the six releases in section 3 all run that kind
of comparison. What those sections do not say is how often real reward-model
comparisons land close enough for the failure modes to matter. We ran the
live comparison twice, on real reward models rather than engineered plants,
with the seed and length confounds controlled throughout. Both land at or
near a tie, and a third data point from outside this project reads the same
way.

![Dot plot, pooled p-value per judge for both live comparisons. None clears significance. Live comparison 1 (Skywork-Reward-V2-8B vs. GRM-Gemma2-2.6B): primary judge p = 0.698 (80 wins to 86 across 166 decided prompts), cross-check judge p = 0.213 (74 to 91 of 165 decided). Live comparison 2 (OA-DeBERTa vs. GRM-Gemma2-2.6B, length-residualized primary rule from the start): primary judge p = 0.38 (122 wins for GRM to 137 for OA-DeBERTa across 259 decided prompts), cross-check judge p = 0.58 (125 to 135 across 260 decided).](/assets/images/writing/reward-shopping-audit/fig5_near_tie_pvalues.png)

The two comparisons are not scored by the same rule, for a reason that is
chronological rather than convenient. Comparison 1 ran before the length
findings in step 4 existed and was scored under the raw vote rule in force
when it ran. Comparison 2 fixed the length-residualized rule as its primary
reading before its first training step. Each is reported under the rule that
governed it, which means comparison 1 carries a length caveat that
comparison 2's design rules out in advance.

The first comparison trained a leaderboard-favored reward model,
Skywork-Reward-V2-8B, against a cheaper one it clearly outranks on a
standard leaderboard, GRM-Gemma2-2.6B, across three seeds on the 200
held-out prompts, with blind dual judging. The primary judge's pooled vote
splits 80 wins to 86 across 166 decided prompts, p = 0.698, with a
Skywork win rate of 48.2 percent and a 95 percent interval from 40.7 to
55.7. The cross-check judge leans the same direction and is not significant
either, 74 to 91 across 165 decided prompts, p = 0.213. Its decided count
differs by one because each judge's own noise floor sets its own ties. Both
intervals straddle 50 percent, which is the same signature as a margin
sitting below its own detection floor. One seed shows a real, individually
significant lean toward Skywork, p = 0.005 on both judges, and that lean
does not survive a length-matched re-check at any band tested. It appears
substantially explained by that seed's variants drifting apart in response
length, the same nuisance axis step 4 shows can turn a training-time
difference into apparent reward-quality signal. The pooled verdict is a
near-tie, on a leaderboard's top-ranked reward model against one it clearly
outranks.

> **One held-out prompt from that comparison, both variants, both judges.**
>
> **Prompt** (held-out set, general Q&A): "What are some of the best TV shows on HBO?"
>
> **Skywork-trained variant's response** (336 characters): "Some highly acclaimed TV shows
> produced by HBO include 'Game of Thrones,' 'The Wire,' 'True Blood,' 'Westworld,' 'Girls,'
> 'Insecure,' 'Ozark,' 'Succession,' 'The Leftovers,' 'Stranger Things,' and 'Big Little Lies.'
> These series have garnered critical acclaim, won numerous awards, and have become cultural
> phenomena in their own right."
>
> **GRM-trained variant's response** (174 characters): "Some of the best TV shows on HBO
> include: 1. Game of Thrones 2. The Wire 3. True Detective 4. Westworld 5. Silicon Valley 6.
> Ozark 7. Veep 8. Barry 9. Watchmen 10. Succession"
>
> **Judge scores** (each judge's own scale, higher is better). The primary judge scores the
> Skywork variant −4.69 and the GRM variant −2.70, a 1.98-point margin favoring the shorter
> GRM response. The cross-check judge scores them 1.96 and 3.47, a 1.51-point margin, the
> same direction. Both judges favor the shorter response on this prompt, one item among the
> decided prompts the pooled p-values above are computed over.

The second comparison trained GRM-Gemma2-2.6B against OA-DeBERTa, a reward
model already known from a companion evaluation to be substantially weaker.
It ran the same design on the extended 325-prompt pool with
length-residualized judging as its primary rule from the start. It, too,
lands as a near-tie. The primary judge splits 122 wins for GRM to 137 for
OA-DeBERTa across 259 decided prompts, p = 0.38, GRM win rate 47.1 percent,
interval 41.1 to 53.2. The cross-check judge splits 125 to 135, p = 0.58.
Both intervals straddle 50 percent. The per-seed vote signs do not even
agree in direction on either judge, and neither judge clears the
two-part rule.

The floor arithmetic from section 2 applies to these two comparisons as
well, with one adjustment the different statistic needs, and skipping the
adjustment would understate our own floors. Section 2's floor is for two
independently evaluated scores. A win share is a different object. Each
decided prompt is one head-to-head trial, so the comparison is already
paired, and the compared quantity is the win-share difference D = 2w − 1,
where w is one variant's share of the decided prompts. The variance of w is
w(1 − w)/N, the variance of D is four times that, so the standard error of
D is 2 sqrt(w(1 − w)/N) and the floor is 2.8016 times that standard error.
Comparison 1's margin is 6 prompts out of 166 decided, 3.6 points of win
share, against a floor of 21.7 points, a ratio of 0.17. Comparison 2's
margin is 15 of 259, 5.8 points, against a floor of 17.4, a ratio of 0.33.
The arithmetic checks against the reported tests. Converting each ratio
back to a z-score gives 0.47 and 0.93, implying p of about 0.64 and 0.35,
consistent with the measured 0.698 and 0.38, where the unpaired formula
would have implied 0.51 and 0.19, which the measured values do not support.
Both ratios say what 12 of the 15 census ratios say. The instrument that
produced these numbers could not reliably have shown a gap this size.

The third data point is not ours. It is AI2's saturation caveat, quoted in
full in section 3, reached from 17 real PPO runs against real policies, the
largest downstream validation footprint of the six releases. A small,
seed-replicated, length-corrected instrument on one task and a 17-run PPO
sweep on another, built by different teams for different purposes, land on
the same reading. Once a reward model clears some basic quality bar,
further differences between good reward models are hard to detect
downstream, and a single-seed, single-horizon comparison that reports a
clean winner between two such models is more likely reporting sampling
noise than a property of the models.

One reading these ties do not license is a power calculation run backwards
from the observed margins, which would only restate the p-values in other
units. The sensitivity claim comes from two places fixed independently of
these results. The same instrument passes a known-null control, and it
detected an engineered half-signal reward model on 2 of 3 seeds at p values
down to 9 x 10⁻²⁹. What the instrument cannot resolve is then stated as a
floor rather than a power figure. At these decided-prompt counts, a
win-share gap under 21.7 points in the first comparison, or under 17.4 in
the second, is not reliably detectable here, and both observed margins are
far under those.

## 7. What this method cannot do

Several limits bind everything above, stated once each.

**On the comparisons that matter most, nobody knows the right answer.** The
live comparisons in section 6 are writing tasks. A writing task has no
answer key, so no checker can mark a response right or wrong. When we say
one training run beat another there, the whole claim rests on two judge
models preferring its answers. We chose judges with no shared training
history with the models being judged, so that a shared blind spot would not
get counted twice, the risk Preference Leakage names ([arXiv:2502.01534](https://arxiv.org/abs/2502.01534)).
Then we measured how independent the two judges actually are. Their scores
move together with a correlation of 0.91, and two judges that aligned
amount to about one independent opinion, not two. One more limit sits on
top of that. The length bias in step 4 was caught on a task with an answer
key, where it could be proved. The tasks here have no answer key, so we
assume the same bias operates, and that assumption is reasonable but not
provable in this setting.

**One task, one model family, one scale, few seeds.** Every live comparison
runs a policy roughly 50 times smaller than a production fine-tune, at most
three seeds, for single-digit dollars. Whether a small-scale winner
predicts a production winner is untested here, the same bet the field makes
whenever a scaling law fit on small runs plans a big one. The exact
magnitudes, the sign-flip counts, the 108-to-108 tie, the seed-variance
ratios, are this project's measurements on this setup, not universal
constants.

**The census floor assumes two independently drawn scores.** Where a
paper's two numbers share a fixed question set, the true floor runs lower,
though not by enough to close an order-of-magnitude gap. That caveat
concerns two accuracy scores on a shared question set. It does not carry
over to section 6's win shares, which are paired head-to-head counts from
the start and take the different formula derived there. The two are
separate statistics that happen to share a multiplier, not one statistic
computed twice. Section 6's formula carries its own assumption in exchange,
and a weaker one. It treats each decided prompt as an independent trial,
which pooling the same prompt across three seeds strains. Neither floor in
this paper is exact. Both are the right order of magnitude for the question
being asked, which is whether a margin is within reach of its instrument
at all.

## 8. Related work

This paper asks what six reward-model releases' evaluations actually
support, how far a single-seed comparison can be trusted, and what a
cheaper discipline looks like. The table places that against the closest
neighboring work, and the paragraphs after it walk the relationships that
need more than a cell.

| Approach | Answers | Requires | Our relation |
|---|---|---|---|
| **This work** | What 6 releases' evaluations support, single-seed reliability, a cheaper protocol | Six papers read verbatim, a null-checked coupling pipeline, no ground-truth checker | n/a |
| Small-benchmark irreproducibility in reasoning RL (Hochlehnert et al.) | Whether reported reasoning gains survive standardized re-evaluation and seed variation | Released checkpoints, re-run under controlled decoding across many seeds | The nearest prior statement of section 2's argument. We compute the floor from each paper's own stated numbers instead of re-running |
| Static reward-benchmark-to-downstream calibration (PPE, RewardBench 2, Kim et al.) | Whether a benchmark score predicts downstream RLHF/best-of-N outcomes | An existing benchmark, sometimes an end-to-end calibration run | Establishes the phenomenon we audit and test per-instance |
| Underspecification mechanism account (Eisenstein et al.) | Why in-distribution-agreeing reward models diverge under distribution shift | Analysis of reward-model families under shift | The mechanism our near-ties are consistent with |
| Verbosity bias in LLM-as-judge (Zheng et al.) | Documents that judges favor longer responses | A judge model, paired responses of varying length | The bias origin our judges also carry |
| Length-Controlled AlpacaEval (Dubois et al.) | Regresses length out of leaderboard win rates | A leaderboard, paired length and preference data | We adapt the mechanism one level up, to training |
| Length as dominant RLHF confound (Singhal et al.) | Shows length dominates reward-model training itself | Analysis of reward models under RLHF | Closest mechanism neighbor to nuisance-axis amplification |
| RM length-debias engineering (ODIN, Post-hoc Calibration) | Engineers a reward model less sensitive to length | Access to the reward model's training or scoring pipeline | Same family, applied to the comparison check instead of RM design |
| Known-null sanity checks (Adebayo et al., Hewitt and Liang, Sollenberger et al.) | Whether a self-built evaluator is sensitive to what it claims | A model, probe, or judge that can be corrupted as a null | The lineage our null-check borrows directly |
| Communication-free noise coupling (Daliri, Musco, Suresh) | How to couple samples without shared communication | Access to both distributions at inference time | The coupling primitive we repurpose for training variants |
| Preference Leakage | How judge contamination inflates apparent preference | A judge, candidates with shared or unmeasured lineage | The risk our lineage-clean judge selection guards against |

Hochlehnert et al. ([arXiv:2504.07086](https://arxiv.org/abs/2504.07086)) made the small-benchmark argument
first, from re-runs of released checkpoints across 20 seeds, and section 2
already states what their evidence shows and what the two-number check adds.
Ten of the fifteen census rows are AIME24 rows, so their argument covers
most of the table and the priority is theirs. Three static calibration
efforts, PPE (Frick et al., [arXiv:2410.14872](https://arxiv.org/abs/2410.14872)), RewardBench 2 (Malik et al.,
[arXiv:2506.01937](https://arxiv.org/abs/2506.01937)), and Kim et al. ([arXiv:2505.12763](https://arxiv.org/abs/2505.12763)), establish that a
benchmark score does not straightforwardly predict downstream outcomes,
which is the phenomenon sections 3 and 6 audit at the practice level and
test per-instance. RewardBench 2 is also one of the six releases section 3
reads, and its volunteered saturation caveat is this paper's independent
corroboration in section 6. Eisenstein et al. ([arXiv:2312.09244](https://arxiv.org/abs/2312.09244)) explain
why reward models that agree in distribution can diverge once training
shifts the distribution, a mechanism our near-ties are consistent with and
do not directly test. Zheng et al. ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)) document the
verbosity bias our judges also carry. Length-Controlled AlpacaEval (Dubois
et al., [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)) regresses length out of a static leaderboard's
win rates unconditionally. We adapt the same mechanism one level up, to a
training-variant comparison check, and find the correction is not universally
safe, since it hurts truth-agreement in checkable settings, which their
unconditional design has no way to see. Singhal et al. ([arXiv:2310.03716](https://arxiv.org/abs/2310.03716))
show length dominating reward-model training itself, the closest
mechanism-level neighbor to nuisance-axis amplification, one level down
from where we find it. ODIN (Chen et al., [arXiv:2402.07319](https://arxiv.org/abs/2402.07319)) and Post-hoc
Reward Calibration ([arXiv:2409.17407](https://arxiv.org/abs/2409.17407)) engineer the length sensitivity out
of the reward model, the same debiasing family applied to model design
rather than to the comparison check. Adebayo et al. ([arXiv:1810.03292](https://arxiv.org/abs/1810.03292)), Hewitt
and Liang ([arXiv:1909.03368](https://arxiv.org/abs/1909.03368)), and Sollenberger et al. ([arXiv:2408.11729](https://arxiv.org/abs/2408.11729))
validate evaluators against known-null controls in vision, probing, and
code verification, and the null-check discipline here borrows that
lineage directly. Applying it a second time, to our own passing controls,
is what caught the length confound. Daliri, Musco, and Suresh
([arXiv:2408.07978](https://arxiv.org/abs/2408.07978)) built the communication-free noise coupling for
speculative decoding, and we repurpose it to couple two training variants.
Preference Leakage ([arXiv:2502.01534](https://arxiv.org/abs/2502.01534)) documents the judge-contamination
risk our lineage-clean judge selection guards against, and section 7
records why that guard is partial.

Per-run records, evaluation outputs, cost accounting, and analysis scripts
for every measurement in this paper ship in the companion repository
released alongside it, one directory per claim, each with the command that
re-derives its headline number from the raw records. Two categories are
excluded rather than silently dropped, trained model weights and any single
record file above roughly 20 MB. Both are named with their sizes in the
directory that would otherwise hold them, and both remain available on
request.
