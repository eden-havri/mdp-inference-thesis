# Dual-output LaTeX manuscript

This directory contains one shared scientific manuscript and two thin output
wrappers:

- `thesis.tex` builds the bilingual A4 BGU thesis using the 2022 BGU template
  structure: English cover and approval pages, English front matter, Hebrew
  contents, Hebrew abstract, and Hebrew cover.
- `paper.tex` builds a compact article-style submission draft.
- `shared/` contains all scientific prose, equations, notation, and the
  bibliography database. Scientific content should be edited here, not copied
  between wrappers.
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
are recorded in `vendor/bgu-template/ATTRIBUTION.md`.  Scientific content
remains in `shared/` and is not duplicated between outputs.

Before deposit, confirm the Hebrew spelling of the author and advisor, the
working Hebrew title translation, faculty, department, degree wording, and
submission month/year in `shared/metadata.tex`.  The bracketed Hebrew-name and
date fields are intentionally visible so an unverified draft cannot be
mistaken for a submission-ready copy.

Before a formal submission, validate page size, binding margin, line spacing,
title pages, abstract languages, declaration text, advisor wording, and
bibliography style against the current graduate-school instructions.

## Scientific narrative

The shared manuscript defines `exp(beta * expected return)` before choosing an
approximation. Its primary method is guided replica-exchange policy MCMC with
exact Hastings correction. Exact dynamic-programming and fixed-tape
sample-average targets are distinguished explicitly; direct ELBO and replicated
SMC variants are retained as ablations. One sampled policy is preserved
throughout each episode, and exact finite-MDP validation is required before
scaling. Paired experiments use identical environments, seeds, evaluation
budgets, and reporting.

## Metadata and references

`shared/metadata.tex` contains the confirmed English author and advisor names,
the current Computer Science/Natural Sciences degree assumptions, and visible
placeholders for unconfirmed Hebrew names and submission dates.
`shared/references.bib` contains the primary sources cited by both wrappers.
