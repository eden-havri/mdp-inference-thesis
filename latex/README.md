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

## BGU template provenance

The thesis wrapper is adapted from **BGU Thesis Template (New Version 2022)**
by Ilan Git, downloaded from Overleaf under CC BY 4.0.  The source and license
are recorded in `vendor/bgu-template/ATTRIBUTION.md`. The article uses a neutral
review layout until a venue and its official submission format are selected;
its current margins do not claim compliance with a conference template.

Before deposit, confirm the Hebrew spelling of the author and advisor, the
working Hebrew title translation, faculty, department, degree wording, and
submission month/year in `shared/metadata.tex`.  The bracketed Hebrew-name and
date fields are intentionally visible so an unverified draft cannot be
mistaken for a submission-ready copy.

Before a formal submission, validate page size, binding margin, line spacing,
title pages, abstract languages, declaration text, advisor wording, and
bibliography style against the current graduate-school instructions.

## Scientific narrative

Both manuscripts define `exp(beta * expected return)` before choosing an
approximation and acknowledge prior Gibbs policy search. The candidate method
is guided replica-exchange policy MCMC with
exact Hastings correction. Exact dynamic-programming and fixed-tape
sample-average targets are distinguished explicitly; direct ELBO and replicated
SMC variants are retained as ablations. One sampled policy is preserved
throughout each episode, and exact finite-MDP validation is required before
scaling. Structural blocks use exact transition information even with rollout
scores. Matched total-budget enforcement and native-baseline reporting remain
prerequisites for confirmatory comparisons. See `../docs/reassessment_2026-09-29.md`
and `../docs/publication_plan.md` for the current decisions.

## Metadata and references

`shared/metadata.tex` contains the confirmed English author and advisor names,
the current Computer Science/Natural Sciences degree assumptions, and visible
placeholders for unconfirmed Hebrew names and submission dates.
`shared/references.bib` contains the primary sources cited by both wrappers.
