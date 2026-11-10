# Reinforcement Learning (chapter introduction)

Source: `chapter_reinforcement-learning/index.md` (lines 1 to 59, up to the slides marker). About 860 words: five paragraphs of introduction (about 480 words), one figure (`mdl-rl-agent-env.svg`), and a resources list of fifteen entries (about 380 words). No code. The chapter has no `further-reading.md`; the question whether it needs one is answered at the end of this file.

## Verdict

The first four paragraphs are a good introduction, and the first is the best opening paragraph of any chapter introduction I have read in this part: a language model writes a thousand tokens and a checker says once whether the answer is right, "No token has a label of its own, and each one changes what the model can write next." The reader feels the problem in three sentences, and the pole on a cart shows that it is not special to text. The two questions that follow (credit for a late reward, search without a teacher) are the right two. The introduction then stops leading. The route is eight consecutive sentences of the form "X asks how ...", the two questions are never used to sort the sections, nothing says what a reader may skip or what the chapter assumes, and the resources list is fifteen unranked entries described in catalogue prose. The most important fix is the route paragraph: organize it by the two questions, name the two running examples, and say which sections a reader bound for language-model post-training needs.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought and focus | 4 |
| Pedagogy | 3 |
| Language | 3 |
| Ease and enjoyment | 4 |

(Paragraphs one to four: 4 to 5 throughout. The route paragraph and the resources list: 2 to 3 for language.)

## What works

- **The opening.** "Every model in this book so far was trained on a dataset that sat still: nothing the model predicted changed which example came next." One sentence states what was true of the whole Basics part and what is about to change. Protect the paragraph word for word.
- **Two questions, plainly put.** "Which decisions deserve credit for a reward that arrives late? And how can a learner find good decisions when nobody shows it what they are?" These are the credit-assignment and exploration problems without the jargon.
- **History used as argument.** Bellman answers the first question "for problems whose rules are known: score each situation by the reward still to come, and a long problem becomes a chain of one-step problems"; that clause is the Bellman equation in words, and it prepares :numref:`sec_valueiter` better than a formula would. The attributions (Bellman's dynamic programming; Barto, Sutton and Anderson 1983 for the pole; Sutton 1988; Watkins 1989; Williams 1992) are right as far as I know them.
- **A judgment.** "They also explain why imitation is not enough: a model fine-tuned on an expert's text never practices recovering from its own mistakes." This tells the language-model reader why the chapter is in the book.
- **The chapter's method, stated.** "most examples are small enough to solve exactly, so that every estimate can be checked against the exact answer". This is what distinguishes the chapter from every course listed below it, and it is true of the sections.
- **Scope said honestly**: several agents are left out, hidden state and learned models are brief.

## Structural findings

1. **The route has no voice and no ranking.** See RLIDX-1 and RLIDX-2. The writing guide's section 12.1 names the "asks how" list as a defect, and the Basics review found the same template in chapter 4 (seven sentences there, eight here).
2. **The two questions are posed and then dropped.** The chapter has a real architecture that the introduction could show in two sentences: sections one to three answer the credit question (values, exactly and then from samples), section four the search question, and sections five to eight change what is learned from a table of values to a policy. The slide deck below the marker says it ("Values Answer the Credit Question, Trial and Error the Search"); the text does not. See RLIDX-3.
3. **A third theme hides in a subordinate clause.** "Because the learner collects its own data, its mistakes change what it learns from next" is the idea that ties exploration, distribution shift in imitation and the on-policy requirement together. It is the last sentence of the history paragraph, where it reads as an afterthought. See RLIDX-4.
4. **The running examples are not named.** A reader learns from the slides, not from the text, that a sixteen-cell frozen lake and a four-prompt verifier task carry the chapter. One sentence would do, and it would make "small enough to solve exactly" concrete.
5. **The resources list is a catalogue.** See RLIDX-6.

## Passage-level findings

**RLIDX-1** (moderate, language). Route paragraph.
Anchor: ":numref:`sec_mdp` asks how to write such a problem down."
Eight consecutive sentences with the verb "asks" (the word occurs eight times in one paragraph). Each question is well chosen; in a row they read as a table of contents with verbs. The slide deck already holds the same content as a table ("Eight Sections, One Question Each"), which is the right form for a list. In the text, lead instead. A draft, to show the register and not as final wording:

> The first three sections answer the credit question. :numref:`sec_mdp` writes a problem of acting over time down, on a sixteen-cell frozen lake that we can solve by hand; :numref:`sec_valueiter` computes the value of every cell when the rules are known; :numref:`sec_qlearning` learns the same values from single transitions. :numref:`sec_exploration` takes up the search question in its simplest form, a row of slot machines, where the cost of trying can be counted. The second half learns the policy itself. :numref:`sec_imitation` copies an expert and shows why copying fails over a long task; :numref:`sec_policygradient` replaces the expert by the reward; :numref:`sec_baselines` removes most of the noise from that estimate, which is the step behind the group-based updates that train language models; and :numref:`sec_deeprl` replaces the table by a network.

**RLIDX-2** (moderate, pedagogy). After the route paragraph.
Anchor: "The code is in PyTorch and JAX, and most examples are small enough to solve"
The introduction never says what matters most, what can be skipped, or what the chapter assumes. The chapter is about 42,000 words of body prose in eight sections, so the reader needs this. Proposed content, two or three sentences: (a) prerequisites: softmax regression and cross-entropy (:numref:`sec_softmax`), gradient ascent, expectations; nothing from the earlier Advanced chapters and nothing from the appendix; (b) a short route for the reader who is heading for language-model post-training: :numref:`sec_mdp`, the definitions of values in :numref:`sec_valueiter`, then :numref:`sec_imitation`, :numref:`sec_policygradient` and :numref:`sec_baselines`; (c) what may be skipped at first reading: the proofs of :numref:`sec_valueiter`, the bandit theory of :numref:`sec_exploration`. The slide "Where this fits" lists the prerequisites ("Uses: softmax regression, cross-entropy, gradient ascent, expectations"); the text should too.

**RLIDX-3** (minor, thought). Paragraph three.
Anchor: "Two questions follow. Which decisions deserve credit for a reward that arrives"
The questions are answered by name in the same paragraph (Bellman for the first, "trial and error" for the second) and never return. If RLIDX-1 is applied as drafted, they organize the route. If not, at least add a clause to the route saying which sections answer which.

**RLIDX-4** (minor, explanation). Paragraph three.
Anchor: "Because the learner collects its own data, its mistakes change what it learns from next."
Two small problems in this paragraph. This sentence opens with "Because" and no antecedent in the sentence before it (which is about Williams); it is a third point, not a consequence. Move it to follow the two questions as the thing that makes both hard ("A third fact colors both: the learner collects its own data ..."), or to the end of paragraph one, where "each one changes what the model can write next" has prepared it. And "Andrew Barto, Richard Sutton and Chris Watkins learned the same scores by trial and error" makes the researchers the learners; write "showed how an agent can learn the same scores by trial and error".

**RLIDX-5** (minor, focus). Route paragraph, last sentence.
Anchor: "and it ends with a map of the methods of this chapter and the next"
This announces the 23-row table at the end of `deep-rl.md`, more than half of whose rows name methods of the next chapter. If the map moves to the end of :numref:`chap_deep_rl`, as the section review proposes (DR-15), delete the clause. If a reduced map of this chapter's methods stays, the clause can stay with "of this chapter".

**RLIDX-6** (moderate, language and focus). Resources list.
Anchor: "The following freely available resources provide theoretical treatments, courses, implementations, and empirical guidance."
The lead sentence is filler; the list that follows has fifteen entries in four groups at equal weight, and most entries describe a table of contents ("provides a comprehensive introduction to value methods, policy methods, planning, and function approximation"; "develops reinforcement learning from the perspective of optimal control and dynamic programming"; "covers policy gradients, value methods, model-based learning, and offline reinforcement learning at the graduate level"). A reader cannot tell which one to open. Three specific points.
- Sutton and Barto is the book every reader of this chapter will compare it with, and the entry says nothing about how the two relate. Say it: this chapter follows their order through chapters 3 to 6 and 13 in a fraction of the space, proves the contraction results they cite, and adds what they do not treat (truncation, regret bounds, imitation, the language-model case); it skips what they develop at length and a reader should get from them (temporal-difference prediction with its experiments, $n$-step methods and eligibility traces, planning and Dyna, the on-policy control experiments). That is two sentences and it is the single most useful piece of guidance the list could give. "Sutton and Barto received the 2024 Turing Award for their foundational contributions to reinforcement learning" is true (the 2024 award, announced in 2025) and belongs in the history paragraph if anywhere.
- Rank. One first choice per need: Sutton and Barto to read; Szepesvári for the statements with proofs; one course, not five (the entry for CS234 already says it "is the course closest in scope to this chapter"; lead with it and give the others a clause each); Spinning Up for code.
- Three entries concern the next chapter's methods (CleanRL: "the major deep reinforcement learning algorithms discussed after :numref:`sec_deeprl`"; Stable-Baselines3; the 37 details of PPO). The next chapter's introduction relies on them being here ("The resources of :numref:`chap_reinforcement_learning`, among them CleanRL's single-file implementations and the catalog of PPO's implementation details, serve this chapter as well"). Either move them there and reverse the pointer, or say in one clause here that they are for the next chapter.
Unverified: that Szepesvári's book is "under a hundred pages" (I believe the monograph runs to about a hundred pages with its bibliography; check); the title of MIT 6.7920; all five course URLs and that every item is "freely available" (I believe all five books have free author copies).

**RLIDX-7** (minor, explanation). Last paragraph.
Anchor: "hidden state and learned models of the environment appear only briefly"
The clause follows a sentence about :numref:`chap_deep_rl` and reads as a statement about both chapters, but the next chapter gives learned models a full section (:numref:`sec_model_based`). In this chapter the statement is true (a counted model in :numref:`sec_qlearning`, belief states in one exercise of :numref:`sec_mdp`). Reorder: say what this chapter leaves out, then what the next chapter adds.

**RLIDX-8** (minor, currency). Last paragraph.
Anchor: "The Language Models part treats post-training, including learning from human"
The chapters of that part are skeletons at present. The sentence is a pointer, not a dependence, so nothing here fails if they change; it should name the part by a label once one exists, and it is repeated twice in the method table of `deep-rl.md` (DR-15) and again in the next chapter's introduction. Once per chapter is enough.

## Code and experiments

None. The figure `fig_rl_agent_env` carries the notation of the loop ($a_t$, $r_t$, $s_{t+1}$, $\tau$) and its caption is exact; it uses $r_t$ for the reward returned after action $a_t$, the convention of the sections (not Sutton and Barto's $R_{t+1}$), which is worth keeping consistent.

## Still open from October

The October review's complaint about this file was a 21-row map of algorithms in the introduction. The map has left the introduction (it now closes `deep-rl.md`, with 23 rows); what remains here is the sentence that announces it (RLIDX-5).

## Does the chapter need a `further-reading.md`?

No. The next chapter's page (`chapter_deep-reinforcement-learning/further-reading.md`, :numref:`sec_deep_rl-further-reading`) is organized by that chapter's sections, so it does not serve this chapter's methods, but its "Beyond the Chapter" paragraph already speaks for both ("Three assumptions of these two chapters can fail": several agents, hidden state, changing tasks). What this chapter itself leaves open is small and is pointed to well enough from inside its sections or from a ranked resources list: temporal-difference prediction and traces (Sutton and Barto), bandit theory (Lattimore and Szepesvári), sample-complexity rates (Agarwal et al.), inverse reinforcement learning (one exercise of `imitation.md`). A separate page would become the survey the writing guide warns against. Two things should be done instead: rank the resources list as in RLIDX-6, and add one sentence to the last paragraph of this introduction saying that :numref:`sec_deep_rl-further-reading` closes both chapters. If the method map moves out of `deep-rl.md` (DR-15), that page is its natural home.

## For the chapter review

- The introduction promises one question per section, and the sections keep the promise; the section reviews find one question answered in each. What the introduction lacks is the level above: the chapter's two halves (values, then policies) and its two running examples.
- The introduction says the code is in PyTorch and JAX. The sources agree (both tabs, `outputs/jax/` present). The project's `CLAUDE.md` still says the chapter "ships PyTorch-only"; that note is stale.
- The introduction's strongest sentence about method ("every estimate can be checked against the exact answer") is the chapter's best feature and should be the stated standard for the next chapter as well.
- No reference into the mathematics appendix and none needed here.
