# Restoration notes (private to maintainers)

This folder contains the **pre-redaction** copies of the three
reviewer-facing documents:

- `README.md` — full title, author list, dataset name, funder
- `CITATION.cff` — full title, ORCIDs, preferred citation
- `RUNBOOK.md` — full title and project paths

The repository is **public on GitHub during the peer-review cycle**, but
the paper title, author list, and dataset name have been removed from
the public README/CITATION/RUNBOOK to avoid these strings being
indexed by web search engines while the manuscript is still under
review.

## When the paper is accepted

1. Copy the three files in this folder back to the repository root,
   overwriting the redacted versions.
2. Optionally update them with the final published bibliographic
   information (DOI, volume, pages, etc.).
3. Commit with message:
   `Restore public title, authors, and dataset reference upon acceptance`.
4. Tag the commit (e.g. `accepted-camera-ready`).
5. This `.restore-on-accept/` folder can stay or be deleted.

## Why this redaction

The reviewer artifact repository must be public for the editorial
team and reviewers to inspect the code and aggregated results during
peer review. But because GitHub READMEs and CITATION.cff files are
indexed by Google Scholar and other crawlers, the paper title and
author list would be discoverable while the paper is still in review.
Some venues consider this undesirable (e.g. for blind review
processes or to avoid premature publicity). To stay on the safe side
this artifact uses surface-level redaction until acceptance.

## What is NOT redacted

- The repository **name** (`ember2024-p18-threshold-sensitivity`)
  remains, because the manuscript's Data and Code Availability section
  cites this exact URL and we cannot change a published submission's
  URL.
- The Zenodo DOI (`10.5281/zenodo.19868109`) remains, again because
  the manuscript cites it.
- Code module names, schema names, and variable names remain — they
  are code, not text.
- The git history (commit messages) is not rewritten.

The redaction is therefore **prophylactic** — enough to keep
Google/Scholar searches from surfacing the paper title, but not
absolute anonymisation.
