# Placement Quiz Results

**Date:** 2026-05-24

## Scores

| Area | Score |
|------|-------|
| Math & Statistics | 2/2 |
| Classical ML | 1/2 |
| Deep Learning | 2/2 |
| NLP & Transformers | 1/2 |
| Applied AI | 2/2 |
| **Total** | **9/10** |

**Entry point: Phase 11 — LLM Engineering**

---

## Missed Questions

### Q4 — Classical ML (1/2)
**Question:** Which of the following is a hyperparameter of a Random Forest?
**Your answer:** The Gini impurity at each node
**Correct answer:** The number of trees

**Diagnosis:** Gini impurity is computed during training as the model learns splits — it's a learned quantity, not something you set. The number of trees is set before training begins, making it a hyperparameter. The key distinction: hyperparameters are chosen by you, learned parameters are chosen by the algorithm.

### Q7 — NLP & Transformers (1/2)
**Question:** In the Transformer architecture, what does the attention mechanism compute between?
**Your answer:** Encoder and Decoder only
**Correct answer:** Queries, Keys, and Values

**Diagnosis:** Attention operating "between encoder and decoder" describes only *cross-attention*, which is one specific use of the mechanism. Attention in general (including self-attention within the encoder or decoder alone) always operates over Queries, Keys, and Values. This is foundational to everything in Phase 11 — RAG, fine-tuning, context engineering all rely on understanding how Q/K/V work.

---

## Personalized Learning Path

| Phase | Name | Status | Est. Hours |
|-------|------|--------|------------|
| 0 | Setup & Tooling | Skip | -- |
| 1 | Math Foundations | Skip | -- |
| 2 | ML Fundamentals | Review | 21 |
| 3 | Deep Learning Core | Skip | -- |
| 4 | Computer Vision | Skip | -- |
| 5 | NLP — Foundations to Advanced | Review | 30 |
| 6 | Speech & Audio | Skip | -- |
| 7 | Transformers Deep Dive | Review | 14 |
| 8 | Generative AI | Skip | -- |
| 9 | Reinforcement Learning | Skip | -- |
| 10 | LLMs from Scratch | Skip | -- |
| 11 | LLM Engineering | Do | 17 |
| 12 | Multimodal AI | Do | 65 |
| 13 | Tools & Protocols | Do | 24.5 |
| 14 | Agent Engineering | Do | 42 |
| 15 | Autonomous Systems | Do | 20 |
| 16 | Multi-Agent & Swarms | Do | 28 |
| 17 | Infrastructure & Production | Do | 32 |
| 18 | Ethics, Safety & Alignment | Do | 31 |
| 19 | Capstone Projects | Do | 500 |

**Your personalized path: ~825 hours across 12 phases.**

---

## Recommendation

Start with **Phase 7: Transformers Deep Dive** before Phase 11. The Q7 miss — thinking attention is encoder-decoder only — is exactly the gap Phase 7 addresses, and it's foundational to RAG, fine-tuning, and context engineering in Phase 11. Phase 2 (hyperparameters vs. learned parameters) is a lighter review; skim it in parallel or skip if the Q4 miss was a vocabulary slip rather than a conceptual gap.
