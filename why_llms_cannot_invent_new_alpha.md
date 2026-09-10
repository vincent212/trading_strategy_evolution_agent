# Why LLMs Cannot Invent New Alpha

Large language models, and the agent frameworks built on them, can search for and validate
trading strategies. They cannot invent genuinely new alpha. This is not an engineering gap to
be closed with a better agent or more orchestration — it is inherent to how the models work.

## What these systems actually do

The current wave of "AI quant" systems all share one shape. A language model proposes and
mutates a candidate — a strategy, or a formulaic factor — an evaluator scores it, and the best
candidates feed back into the next prompt. This is the template DeepMind's **FunSearch**
established: the language model is a *search operator* over a space of programs, not an
oracle. LATSS (LLM-Assisted Trading Strategy Search) applies it to strategies built from a
supplied set of indicators; the alpha-mining agents (AlphaAgent, Alpha-GPT, CogAlpha,
AlphaForge, Hubble, and the rest) apply it one level down, to formulaic factors built from raw
price and volume through a fixed operator language. They differ in scope and machinery —
multi-agent hierarchies, tree search, sandboxes, novelty penalties — but they share the
paradigm: **recombination of a human-defined vocabulary, followed by certification.**

This is the same thing a coding agent does. A coding agent recombines and tests known
constructs, under supervision, against a specification; it does not invent a new programming
paradigm. LATSS recombines and tests known indicators, under supervision, against a backtest;
it does not invent a new source of edge. Both are search-and-certify engines. They are
genuinely useful, and they are bounded.

## Two limitations — only one is an engineering problem

Two well-known critiques of language models bear directly on this, and it matters that they
are different.

**Yann LeCun: no model of the world.** LeCun argues that language models have no grounded,
causal understanding — only next-token prediction over surface patterns ("word models, not
world models") — so they cannot reason about the mechanism behind an effect. This limitation
*can* be supplied externally. A framework hands the model the world it lacks: the features are
measurements of the market, and the backtest is a reality check that scores every proposal
against what actually happened. In that sense LATSS *is* a world model bolted onto the LLM.
This objection is addressable by scaffolding.

**Tom Zahavy (DeepMind), "LLMs can't jump": no abduction.** Zahavy argues that language models
handle induction (pattern-matching) and deduction (formal proof) but are structurally
incapable of abduction — the creative leap that formulates a genuinely new explanatory
hypothesis, the premises rather than the proof. His case study is Einstein's jump to
spacetime curvature: a hypothesis that the data available at the time did not compress into.
This limitation is **not** addressable by scaffolding. Inventing a new alpha — a predictive
relationship that is not a recombination of the given primitives — is exactly such a jump: a
hypothesis outside the training distribution. Giving the model a world model does not make it
jump. Grounding is necessary; it is not abduction.

That is the crux. The framework can fix the missing world model. Nothing an agent does can
manufacture the jump.

## Why alpha is the hard case: discovery, not design

It is tempting to say inventing new alpha is like asking a model to invent a new programming
language. The analogy is only half right, and it flatters the model. A programming language is
a *design artifact*: it need only be internally coherent and useful, it can be assembled as a
functional remix of known constructs, and it can be verified deterministically by running it.
A language that compiles and runs is a success.

Alpha is not a design artifact — it is an *empirical discovery*. A new factor must be **true
about the world**: a real, non-obvious relationship between what can be measured now and what
returns will be later. Truth about the world cannot be recombined into existence from priors;
it has to be discovered. And it is verified only noisily, against historical data with a low
signal-to-noise ratio, which is why a plausible-looking factor that backtests well is usually
just overfit noise. Worse, alpha is adversarial and perishable: it must be not only true but
*uncrowded*, and it decays as others find it — the opposite of a language, which grows more
valuable as it is adopted.

The right reference class is therefore not "invent a new language" but "discover a new law of
physics" or "find a drug that actually works in trials" — a true, non-obvious claim about
reality that must survive contact with noisy data, not merely be coherent. Language models
recombine what they have seen. A new law is a jump to something they have not.

## The ceiling

Put together, this defines a ceiling on how far AI can be pushed in trading-strategy
development, and it is close to where LATSS already sits. An LLM-driven framework can:

- recombine a human-supplied vocabulary of features into candidate strategies,
- fit their parameters, and
- certify the survivors against a valid skill bar that rejects overfit noise.

It cannot decide *what to measure*. Choosing the raw predictive ideas — the features, the data
sources, the economic mechanisms — is the abductive step, and it remains the human's. The
much-discussed "novelty" in alpha-mining agents (penalizing crowded formulas, requiring an
economic rationale) is novelty *within* the operator space; it steers recombination, it does
not produce a jump.

None of this is a problem to be solved with a cleverer agent, more tools, or a larger context
window. It would require a different kind of model — one capable of abduction, or one with a
genuine world model of the sort LeCun advocates. Until such a model exists, AI's contribution
to alpha is bounded by search and certification: recombining and rigorously testing human
ideas. That is valuable. It is not the creative leap, and no framework will make it one.

---

**References.** FunSearch (DeepMind, 2023). "LLMs can't jump," Tom Zahavy, DeepMind
([tomzahavy.com/projects/llms-cant-jump](https://www.tomzahavy.com/projects/llms-cant-jump)).
Yann LeCun on world models
([MIT Technology Review](https://www.technologyreview.com/2026/01/22/1131661/yann-lecuns-new-venture-ami-labs/)).
AlphaAgent — LLM-driven alpha mining with regularized exploration against alpha decay
([arXiv:2502.16789](https://arxiv.org/abs/2502.16789)).
