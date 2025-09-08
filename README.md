
## Strucutre
gol_mcmc/
│
├── core/
│   ├── Board.py          # Represents a Game of Life board state
│   ├── GoLEngine.py      # GPU-accelerated Game of Life stepper
│   ├── Scorer.py         # Base + concrete scoring functions
│   └── Proposal.py       # Base + concrete proposal strategies
│
├── mcmc/
│   ├── Chain.py          # Manages a single MCMC chain
│   ├── Sampler.py        # Controls multiple chains, orchestrates proposals & scoring
│   └── Scheduler.py      # Optional (simulated annealing, temperature control)
│
├── utils/
│   ├── Visualization.py  # Renders boards, makes videos/GIFs
│   ├── Storage.py        # Saving/loading states (npy, json, db)
│   └── Logging.py        # Experiment logs, scores
│
└── main.py               # CLI / runner script


## Scorer Functions:
- Alive Cells
- 
https://arxiv.org/pdf/1911.01086

Scoring Functions for Game of Life Search

Figure: The glider is a famous five-cell pattern in Conway’s Life that moves diagonally across the grid (ar5iv.labs.arxiv.org). It is a compact structure that reproduces itself while translating, illustrating the kind of “interesting” behavior we may wish to find.

In designing an MCMC or genetic search for interesting Life patterns, one must pick a score (fitness/energy) function to favor desired behaviors. A key distinction is between trajectory-free (static) metrics and trajectory-based metrics. Trajectory-based scores simulate the CA for many generations and measure properties of the evolution (e.g. lifespan, oscillation period, or transient behavior). In contrast, trajectory-free scores look only at the initial (or current) configuration or one-step statistics – for example, counting live cells or computing entropy of the pattern. Trajectory-free measures are much cheaper to compute (no long simulation needed) and map easily to GPU operations, which the user prefers. In practice one can combine both: for example, run the CA a few steps and score the result using static metrics.

Below are several classes of scoring functions that have been used or suggested for finding “interesting” Life patterns. We focus on efficiently computable, mostly trajectory-free measures:

Entropy/Information: One can quantify how ordered or random a configuration is. For instance, the Shannon entropy of the cell distribution (alive vs dead) is 
H=−plogp−(1−p)log(1−p). More generally, one can compute the conditional entropy or mean information gain of a configuration by examining local neighborhoods. Javaheri et al. show that a conditional-entropy measure (averaging the entropy of a cell given its neighbors) can distinguish structural patterns (symmetry vs randomness) in 2D CA 
eprints-gro.gold.ac.uk . In practice we can GPU-compute a histogram of live/dead states or local patch patterns and evaluate H. Purely random chaos has high entropy; pure still-lifes have very low entropy; many interesting patterns lie in between. One could reward moderate entropy or even seek extreme entropy as desired.

Activity/Growth: We often want patterns that “do something” rather than die out immediately. A simple measure is population count (number of live cells) or its peak over a few steps. For example, Dor Pascal’s GA maximized growth by combining life-span and peak size: he gave +1 to the score for each generation the pattern remained active, and then used
score = maxAlive – 2.2·startAlive + stablePeriod

where stablePeriod is the number of generations before the pattern stabilizes (longer is better), maxAlive is the maximum live-cell count reached, and startAlive is the initial population (penalized to encourage small seeds)
dorpascal.com
. This favored Methuselahs (small seeds that live long and grow large). Similarly, one can use activity = (number of births+deaths) in one step (easily computed by comparing current vs. next grid) or average over a few steps. High activity indicates chaotic change; moderate activity can signal rich dynamics. These counts are trivial to compute on a GPU via simple array operations.

Structure/Symmetry: Static structure is often correlated with “interestingness.” For example, global symmetry is easy to measure by comparing a grid to its mirror images or counting symmetric cell-pairs. Jakobj’s novelty-search code simply defined “interesting” seeds as those ending up with ≥20 live cells and at least 10 symmetric cell-pairs
github.com
. In general one can count how many cells are part of larger blocks or known motifs (blocks, blinkers, etc.). Even without pattern recognition, simple connectivity measures help: e.g. count the number of connected live clusters or the largest cluster size. Patterns that are completely fragmented (many tiny clusters) tend to be chaotic. Other geometric measures like perimeter length (boundary between live/dead) or radius of gyration can be used. All of these reduce to summing or comparing grid values and are GPU-friendly.

Known Motif Detectors: If the goal is very specific (gliders, oscillators, guns), one can score directly on the presence of those motifs. For example, after evolving a candidate a few steps, one could convolve the grid with a glider template to count gliders, or look for oscillators of a certain period. This is more expensive but can be done in parallel via convolution filters. Templates for small oscillators (blinkers, toads) or spaceships can be checked quickly for a few top candidates.

In practice, we often combine multiple objectives rather than a single score. Different scoring functions pull the search toward different phenomena. As Salo et al. note, using different score functions can produce “vastly different” discovered gadgets
villesalo.com
. In other words, one metric might find glider-producing patterns, while another finds long-lived chaotic growth. Running several MCMC chains in parallel with different scorers (entropy-based vs. activity-based vs. novelty-based) broadens the exploration.

Implementation (GPU): All the above metrics are based on simple array operations. For example, neighbor counts (for entropy or activity) can be done with a fixed convolution kernel; symmetry counts use element-wise comparisons; population is a sum; entropy is a few log multiplications on the histogram of values. These tasks are trivially parallelizable on a GPU. By contrast, deep trajectory-based evaluations (like simulating 5000 steps or flood-filling clusters) are slower, so one may only simulate a few steps or use incremental measures like the hash of seen states (as Dor Pascal did with run-length encoding)
dorpascal.com
. Ultimately one can define a composite scorer. The goal should be clear (e.g. “maximize life-span and growth” or “maximize static structure”), and each candidate is scored accordingly. For example, a hybrid score might be a weighted sum of entropy and activity, or a multi-objective scheme.

Key References: These ideas have been used in prior work. Dor Pascal’s genetic search for Methuselahs demonstrates an activity-based score
dorpascal.com
, and Jakob’s novelty search illustrates static scoring by size and symmetry
github.com
. Javaheri et al.’s work shows how conditional entropy captures pattern complexity
eprints-gro.gold.ac.uk
. Salo’s Life-search notes that different scoring functions yield different types of solutions
villesalo.com
, underlining the value of testing multiple metrics. In summary, one should choose simple, fast-to-compute measures (entropy, activity, symmetry, motif counts, etc.) aligned with the desired behavior, and potentially combine them to guide an MCMC search toward interesting Life configurations.

Sources: We summarize and cite research on CA pattern metrics
eprints-gro.gold.ac.uk
dorpascal.com
github.com
villesalo.com
 and standard Life references
ar5iv.labs.arxiv.org
 used above for context.

 https://villesalo.com/kuluma/conwaysoldiers.html

https://ar5iv.labs.arxiv.org/html/2301.03195#:~:text=There%20are%20also%20structures%20that,Conway%E2%80%99s%20Game%20of%20Life%20oscillators
https://eprints-gro.gold.ac.uk/id/eprint/17257/1/2015_EPIA.pdf#:~:text=,randomness%20of%202D%20CA%20patterns

https://villesalo.com/
https://villesalo.com/article/SaCoPitGoL.pdf#:~:text=The%20hill,the%20current%20program%20may%20produce

https://dorpascal.com/blog/tech/genetic-algorithm-for-conways-game-of-life#:~:text=a%20chromosome%20remains%20stable%2C%20it,std%3A%3Amax%281%2C
https://dorpascal.com/blog/tech/genetic-algorithm-for-conways-game-of-life#:~:text=score%20is%20determined%20by%20the,std%3A%3Amax%281%2C

 https://github.com/jakobj/novel-life

## Proposals

Ideas: 
80% local flips / local blocks
15% global flips / random blocks
5% motif injection or entropy-seeking moves
### Classical proposals

 q(x′∣x)=αq local (x′ ∣x)+βq global (x′ ∣x)+γq motif (x∣x),with weights α+β+γ=1.
1) Local Adaptations

These are moves that make small, incremental changes so the chain can refine candidate patterns and exploit promising areas.

Single-cell flips near recent activity: Flip one of the cells in the local neighborhood of the last flipped cell (or around active regions with many live cells). This keeps the proposal distribution focused.

Local block flips: Flip small clusters (2×2 blocks, short lines, L-shapes) in one move. These capture motifs that are "atoms" of larger patterns.

Perturbation of methuselahs: Identify known long-lived seeds and slightly alter them

2) Ergodicity

To ensure that any configuration can eventually be reached, proposals must not be too local. If you only ever flip neighbors of neighbors, you’ll get stuck in subregions of state space.
Uniform random flip (global move): Pick any cell uniformly at random and flip it. Even if used rarely, this ensures ergodicity.
Random block injection: Occasionally flip an arbitrary block anywhere on the board.
Random restart / mutation: With small probability, reset a region to a random high-entropy state.

3) Patches 
MakePatches working, ...

### Neural Networks for proposals
 
 Short answer: Yes — the “Meta-Learning MCMC Proposals” idea is directly relevant and can be adapted to GoL search. The original paper trains neural block proposals (mixture-density nets) to approximate block-Gibbs conditionals so learned proposals generalize across models. For Game-of-Life search you won’t be sampling from a fixed probabilistic model, but you can train a conv-based block proposer to (a) suggest promising 3×3 / 5×5 patch edits given a larger context (e.g. 9×9) and (b) be used inside a Metropolis–Hastings wrapper so correctness is preserved. The approach maps quite naturally to your idea of “input: 9×9, output: fill middle 3×3.” 
arXiv
NeurIPS Papers

Below I summarize what the paper does, how it maps to your problem, and give concrete, practical ways to train and use a learned proposal for Methuselah / entropy / change-rate search.

1) What the paper actually does (TL;DR)

The authors parametrize block proposals (i.e. propose values for many variables jointly) with a neural network and train it so the learned proposal approximates the true block conditional (so proposals are high-quality and generalize across related problems). They use mixture density networks (MDNs) and meta-training on families of models so the learned proposals transfer. The result: faster effective sampling when the learned proposals capture common local structure. 
arXiv
Yi Wu's Personal Website

Key takeaway for GoL: instead of hand-designing many specialized flip heuristics, you can train a single ConvNet that, given a context crop, proposes a distribution over possible patch edits (e.g. the 3×3 center). Then use MH to accept/reject — so the sampler remains correct while proposals are much smarter.

2) How this maps to Game of Life (practical recipe)
Proposal parametrization (what the network outputs)

Input: a context patch (example: 9×9 or 11×11) centered on the region you will change. Use 0/1 tensor, maybe + extra channels (age, recent activity).

Output: a categorical distribution over all possible states for the central block (e.g. for 3×3 center, 2^(3*3)=512 classes). Options:

Direct categorical (softmax over 512 logits) — exact and simple.

Mixture model / MDN over bitvectors (paper uses MDNs for continuous/complex conditionals).

Autoregressive patch generator (pixel-by-pixel) conditioned on context — useful for larger blocks.

Architecture: small convolutional encoder (several conv layers with residuals), global pooling into an MLP head that outputs logits over the patch configurations. Enforce equivariances by using convolution and data augmentation (rotations/reflections).

Using the learned proposal inside MH

Treat network proposal q(x'|x) as any MH proposal. Compute acceptance:

a=min(1, π(x)q(x′∣x)π(x′)q(x∣x′) )

where π is your target (if you search you can set π(x) = exp(score(x)/T) as a surrogate). Because MH corrects for biased proposals, you can train the proposer without needing it to obey detailed balance — MH will handle that if you compute q(x'|x) and q(x|x').

Training objectives (three practical routes)

Supervised conditional-matching (paper approach):

Generate training pairs (context, target_patch) from a family of examples (e.g. from random seeds, known interesting patterns, synthetic patches). Train the net to maximize the (approximate) conditional probability of the target patch given context (cross-entropy).

This approximates the true block conditional; well-matched proposals have high MH acceptance and good mixing. (Good when you can generate realistic examples.)

Source: this is what the paper does; they use MDNs / likelihood training. 
NeurIPS Papers

Offline imitation from accepted MH moves:

Run a baseline sampler (or curated edits) and log accepted (context, x' ) moves; then train the network to imitate the successful accepted edits. This biases proposals toward edits that get accepted in practice.

Reinforcement / policy gradient to maximize search objective:

Directly train the proposer to maximize expected improvement of your score after applying the proposed patch and simulating some steps. Use REINFORCE (or advantage actor-critic) with a reward equal to score(x') - score(x) or final score after simulation. Because simulation is expensive, use short rollout lengths or surrogate rewards (early activity / entropy) to keep training feasible.

This optimizes for search utility rather than conditional likelihood. Note: still wrap in MH in deployment to preserve correctness.

Practical hybrid training

Pretrain with supervised conditional matching (fast), then fine-tune with RL on your scoring objective (improves search utility).

Use data augmentation and convolutional invariances so the proposer generalizes across the board and isn’t overfit to one location.

3) Concrete design: 9×9 -> 3×3 center (practical detail)

Input: 9×9 binary, possibly with additional channels (recent activity, time since last change).

Output: logits 512 → softmax → categorical over 3×3 center states.

During proposal: sample a 3×3 candidate c from q(c | context); form x' by replacing center with c. Evaluate score(x') as needed. Accept via MH using q(x|x') which you can compute by extracting context at the same coordinates in x'.

If you use a temperature or surrogate target π(x) = exp(score(x)/T), choose T to tune acceptance.

If computing q(x|x') is expensive (because context in x' changed), you can make the network proposal depend only on outer context (the 9×9 excluding center) so q(x|x') uses the same outer context; then q(x'|x) and q(x|x') are symmetric-ish and easy to compute. But strictly speaking you must compute both densities for MH.

4) Advantages & pitfalls

Advantages

Learned block proposals can jump to promising motifs (higher acceptance than random multi-cell flips).

Convolutional encoder generalizes across spatial locations and board sizes (translation equivariance).

Because MH wraps the learner, correctness is kept (if you can evaluate q(x|x')).

Pitfalls

Low acceptance if proposal distribution drifts away from regions where the target score is high; solve via pretraining and tempered π.

Mode collapse (network always proposing same patch) — prevent via entropy regularizer or explicit entropy objective.

Costly training (needs many rollouts if using RL). Use supervised pretraining and short rollouts.

Computing q(x|x'): when block changes context (e.g. if your block is large), the reverse context differs, so you must evaluate network at x' to compute reverse probability (acceptable but doubles inference cost per MH step). Design the context so reverse evaluation is cheap (e.g. use outer ring only).

5) Evaluation & diagnostics

Acceptance rate of MH with learned proposals (too low → proposals poor).

Effective sample size or mixing time in search (time to reach high-score patterns).

Improvement per proposal: average Δscore after accepted proposals.

Diversity vs. success: fraction of unique motifs discovered.

Ablation: local-only vs mixed local+global vs learned proposals.

6) Implementation sketch (pseudocode)
# 1) architecture (PyTorch-like)
class PatchProposer(nn.Module):
    def __init__(self, in_ch=1, context=9, center=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), # global
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 512)  # logits for 2^(center*center) classes
        )
    def forward(self, ctx):  # ctx shape (B,1,9,9)
        z = self.encoder(ctx)  # -> (B,64,1,1)
        logits = self.head(z)  # (B,512)
        return logits

# 2) supervised pretraining
# Build dataset of (context, good_center) by sampling random boards,
# extracting 'good' centers (e.g. centers that after k steps produce high score),
# or from catalog of known patterns.

# 3) inference MH step
ctx = extract_context(board, pos)  # (1,1,9,9)
logits = model(ctx)
proposed_center = sample_categorical(logits)
x_prime = board.with_center(pos, proposed_center)
q_xy = softmax(logits)[proposed_center]
# compute q_yx by evaluating model on x_prime's context at pos
q_yx = model(extract_context(x_prime, pos)).softmax()[original_center]
accept_prob = min(1, exp((score(x_prime)-score(board))/T) * q_yx/q_xy)
if rand() < accept_prob:
    board = x_prime

7) Training data ideas (how to get conditional targets)

Enumerate small patches: For 3×3 you can brute force evaluate all 512 possibilities for every context and compute which produced higher downstream scores — use that as supervised target (exact conditional or target-weighted). This is feasible and powerful: for every 9×9 context, you can compute the score of all 512 centers after simulating a short rollout — then train the network to rank or predict the best centers.

Use known interesting seeds: manually curated methuselahs / R-pentomino variants as positive examples.

Self-play / bootstrapping: run initial sampler, collect proposals that lead to good outcomes, then train on them.

8) Practical hyperparameters & tips

Patch sizes: start with 3×3 center in 9×9 context. 3×3 gives 512 classes (easy). For 4×4 center (65k states) you need autoregressive or MDN approach.

Pretrain supervised on enumerated center proposals where possible (fast & stable).

Regularize output by entropy term −λ H(q(.|ctx)) to avoid mode collapse.

Use MH acceptance with surrogate π (e.g. exp(score/T)) where score is your metric (finalAlive, entropy, change-rate). Tune T.

Evaluate acceptance rate and effective improvement; adjust mixture probabilities between learned proposals and random/global flips.

9) Bottom line & recommendation

Meta-Learning MCMC Proposals (the paper) is directly applicable as inspiration: learn block proposals (a neural net mapping context → distribution over center patch) and use MH to preserve correctness. Pretrain the network to approximate good local conditionals (supervised enumeration for 3×3 is ideal), optionally fine-tune with reinforcement signals (expected score improvement). This will give you stronger local moves than blind flips while MH ensures unbiased exploration if you care about correctness. 
arXiv
NeurIPS Papers

If you want, I can:

produce a ready-to-run PyTorch implementation that (a) enumerates all 512 centers for a set of sampled contexts, (b) pretrains the proposer on conditional cross-entropy, and (c) plugs the proposer into your existing MH routine for GoL with a sample score (Methuselah / entropy).

or sketch the RL objective (REINFORCE) and the training loop if you prefer direct optimization for search utility.