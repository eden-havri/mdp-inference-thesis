# Dual-output LaTeX manuscript

This directory contains one shared scientific manuscript and two thin output
wrappers:

- `thesis.tex` builds a compile-safe A4 thesis draft using the standard
  `report` class.
- `paper.tex` builds a compact article-style submission draft.
- `shared/` contains all scientific prose, equations, notation, and the
  bibliography database. Scientific content should be edited here, not copied
  between wrappers.
- `styles/` contains presentation-only settings for the two outputs.
- `figures/` is reserved for generated, versioned figures.

## Build

Run either command from this `latex/` directory:

```text
latexmk -pdf thesis.tex
latexmk -pdf paper.tex
```

No PDF is committed by this scaffold. The wrappers use standard packages and
do not depend on a private class file.

## Integrating an official BGU thesis template

The repository does not currently contain an authoritative BGU thesis class.
`thesis.tex` therefore uses `report` with conservative A4 thesis margins and
front matter. When the official template is obtained from the university:

1. Record its source, version, and redistribution terms.
2. Place authorized class/style/logo assets under `vendor/bgu-template/`.
3. Change only the `\documentclass` line and the title-page implementation in
   `thesis.tex` or `styles/thesis-fallback.tex`.
4. Map the official template's author, advisor, faculty, department, degree,
   date, English-title, and Hebrew-title commands to the neutral macros in
   `shared/metadata.tex`.
5. Keep `shared/manuscript.tex` and every file below `shared/sections/`
   unchanged. If the official class uses different heading commands, update
   only `\DocSection`, `\DocSubsection`, and `\DocSubsubsection` in the wrapper.
6. Add any required Hebrew cover and approval pages as wrapper-only front
   matter. Do not duplicate scientific chapters.

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

`shared/metadata.tex` contains the confirmed author and advisor names while
faculty, department, and degree remain placeholders. `shared/references.bib`
contains the verified primary sources cited by both output wrappers.
