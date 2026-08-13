# The paper

`LaTeX/main.tex` is the current write-up. **It contains no numbers.**

Every figure, table and inline value is a macro or `\input` written from
`results/` by `experiments/make_report_tables.py`. That is a deliberate constraint:
the previous version of this project drifted from its own data because its
headline numbers were typed in by hand, and one of them came from a notebook that
hardcoded its weights. A paper that cannot be edited to disagree with the
experiment cannot drift from it.

## Build

```bash
python experiments/run_backtest.py        # only if results/ is missing or stale
python experiments/make_report_tables.py  # regenerate tables, macros, README blocks
cd reports/LaTeX && latexmk -pdf main.tex
```

Needs `texlive-latex-recommended`, `texlive-latex-extra`, `texlive-science` and
`latexmk`. The `paper` job in [`.github/workflows/tests.yml`](../.github/workflows/tests.yml)
builds it on every push and uploads the PDF as an artifact, so the committed
sources are known to compile from a clean checkout.

CI additionally fails if regenerating the tables changes any committed file —
that is, if the prose and `results/` have fallen out of step.

## Guardrails

`tests/test_report.py` enforces, on every test run:

- every `\macro{}` the paper uses is defined by the generator (a stale one would
  otherwise render blank);
- every generated macro is actually cited somewhere (so a deleted result cannot
  vanish silently);
- the generated macros still agree with `results/` to the digit;
- every `\input`, `\includegraphics`, `\ref` and `\cite` resolves;
- the README and postmortem results blocks are present and non-empty.

## The previous version

The March 2025 report is preserved at
[`legacy/PortfolioOptimization_v1_SUPERSEDED.pdf`](../legacy/PortfolioOptimization_v1_SUPERSEDED.pdf).
Its abstract states a result that is invalid; see
[`docs/postmortem.md`](../docs/postmortem.md).
