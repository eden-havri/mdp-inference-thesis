# Dual-output LaTeX manuscript

This directory contains two narratives for the same scientific study:

- `thesis.tex` builds the bilingual A4 BGU thesis using the 2022 BGU template
  structure: English cover and approval pages, English front matter, Hebrew
  contents, Hebrew abstract, and Hebrew cover.
- `paper.tex` builds a concise article from `paper/`, with the core question,
  method, decisive evidence, and limitations. It is currently a research draft,
  not a completed conference submission.
- `shared/manuscript.tex` and `shared/sections/` hold the detailed thesis.
- Metadata, notation, references, and scientific conclusions are shared.
  The prose and level of detail are deliberately different. Verified numerical
  results must agree across the two narratives.
- `styles/` contains presentation-only settings for the two outputs.
- `vendor/bgu-template/` contains the attributed BGU logo and bibliography
  style distributed with the source template.
- `figures/` is reserved for generated, versioned figures.

## Build

Run either command from this `latex/` directory:

```text
latexmk -pdf thesis.tex
latexmk -pdf paper.tex
```

The source is ready for Overleaf with `paper.tex` or `thesis.tex` selected as
the main file.  The thesis uses pdfLaTeX, matching the source template.

Without latexmk/Perl, run `pdflatex -interaction=nonstopmode -halt-on-error
paper.tex`, `bibtex paper`, then pdflatex twice more; repeat with `thesis`.
From the repository root, `python experiments/write_development_latex.py`
regenerates the shared medium development table from validated run artifacts.

## BGU template provenance

The thesis wrapper is adapted from **BGU Thesis Template (New Version 2022)**
by Ilan Git, downloaded from Overleaf under CC BY 4.0.  The source and license
are recorded in `vendor/bgu-template/ATTRIBUTION.md`. The article uses a neutral
review layout until a venue and its official submission format are selected;
its current margins do not claim compliance with a conference template.

Before deposit, confirm the Hebrew spelling of the author and advisor, the
working Hebrew title translation, faculty, department, degree wording, and
submission month/year in `shared/metadata.tex`. Confirmed English names are
used on the Hebrew draft cover until their Hebrew spellings are confirmed.
The date identifies a September 2026 research draft, not an approved deposit
date. Faculty and department are assumptions awaiting confirmation.

Before a formal submission, validate page size, binding margin, line spacing,
title pages, abstract languages, declaration text, advisor wording, and
bibliography style against the current graduate-school instructions.

## Scientific narrative

Both manuscripts define `exp(beta * expected return)` before choosing an
approximation and acknowledge prior Gibbs policy search. The practical candidate
is PPO-initialized direct variational policy inference: a factorized distribution
over complete stationary tables, a linear return-score gradient, and exact
policy-space entropy. K=1 is a valid unbiased gradient setting, not an exact
exponential weight. Family and optimization error remain explicit limitations.
Policy commitment and native action resampling are both reported.

The paper now contains theory, exact correctness examples, and the evaluation
design, **not preliminary GridWorld results or development result includes**.
Its result section explicitly awaits the complete replicated held-out study.
The thesis retains the detailed gradient proof, clearly labeled development
results and failed SMC/MCMC diagnostics. Both will use the same verified final
numerical assets after confirmation; they need not share pilot material or
identical prose. The source snapshot before this separation is preserved in
`../output/source-archive/latex-before-paper-separation-20261002.zip`. See
`../docs/variational_method_contract.md`, `../docs/inference_selection_2026-09-29.md`,
and `../docs/completion_plan.md` for the current status.

## Metadata and references

`shared/metadata.tex` contains the confirmed English author and advisor names,
the unconfirmed Computer Science/Natural Sciences degree assumptions, and
explicit research-draft date labels.
`shared/references.bib` contains the primary sources cited by both wrappers.
