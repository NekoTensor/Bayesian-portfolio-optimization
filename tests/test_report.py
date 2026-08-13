"""The paper must not contain numbers that ``results/`` cannot justify.

The original version of this project drifted from its own data because every
headline figure was typed by hand into the README and the LaTeX source. These
tests make that failure mode mechanical rather than a matter of discipline:

* every ``\\macro{}`` the paper uses must be defined by the generator, so a
  renamed or deleted result breaks the build instead of silently rendering blank;
* every generated file the paper ``\\input``s must exist;
* the generated macros must agree with ``results/`` to the digit;
* the README and postmortem results blocks must be present and delimited.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
LATEX = ROOT / "reports" / "LaTeX"
GENERATED = LATEX / "generated"
RESULTS = ROOT / "results"

# LaTeX control sequences that legitimately take an empty argument.
BUILTIN_EMPTY_ARG = {"today", "textbf", "emph", "texttt", "hfill", "quad", "qquad"}

pytestmark = pytest.mark.skipif(
    not (RESULTS / "metrics.csv").exists(),
    reason="results/ not populated; run experiments/run_backtest.py",
)


def tex_sources() -> list[Path]:
    return [LATEX / "main.tex"] + sorted((LATEX / "sections").glob("*.tex"))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def defined_macros() -> set[str]:
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", read(GENERATED / "macros.tex")))


def used_macros() -> set[str]:
    """Macros invoked in the paper's ``\\macro{}`` house style.

    Bare invocations (``\\maxWeight`` inside math mode) are legal too, so this set
    is only used to find *undefined* macros. Use :func:`is_referenced` to ask
    whether a given macro appears at all.
    """
    used: set[str] = set()
    for source in tex_sources():
        used |= set(re.findall(r"\\([a-zA-Z]+)\{\}", read(source)))
    return used - BUILTIN_EMPTY_ARG


def is_referenced(name: str) -> bool:
    """True if ``\\name`` appears anywhere, with or without a trailing ``{}``."""
    pattern = re.compile(r"\\" + re.escape(name) + r"(?![a-zA-Z])")
    return any(pattern.search(read(source)) for source in tex_sources())


def test_every_macro_used_by_the_paper_is_defined() -> None:
    missing = sorted(used_macros() - defined_macros())
    assert not missing, (
        f"main.tex/sections use undefined macros: {missing}. "
        "They will render blank. Add them in experiments/make_report_tables.py."
    )


def test_no_orphaned_macros() -> None:
    """A defined-but-unused macro means a result silently vanished from the paper."""
    orphaned = sorted(m for m in defined_macros() if not is_referenced(m))
    assert not orphaned, (
        f"macros generated but never cited: {orphaned}. Either use them or drop "
        "them, so the generator stays an honest description of the paper."
    )


def test_inputs_resolve() -> None:
    for source in tex_sources():
        for target in re.findall(r"\\input\{([^}]+)\}", read(source)):
            path = LATEX / target
            if not path.suffix:
                path = path.with_suffix(".tex")
            assert path.exists(), f"{source.name} inputs missing file: {target}"


def bibitem_keys() -> set[str]:
    """Keys defined in the bibliography, allowing natbib's ``\\bibitem[label]{key}``."""
    return set(
        re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", read(LATEX / "main.tex"))
    )


def cited_keys() -> set[str]:
    """Keys cited anywhere, including multi-key ``\\citep{a,b}`` groups."""
    cited: set[str] = set()
    for source in tex_sources():
        for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", read(source)):
            cited |= {key.strip() for key in group.split(",")}
    return cited


def test_every_citation_has_a_bibitem() -> None:
    """An undefined citation key silently renders as ``[?]`` in the built PDF."""
    missing = sorted(cited_keys() - bibitem_keys())
    assert not missing, f"cited but absent from the bibliography: {missing}"


def test_no_unreferenced_bibitems() -> None:
    unused = sorted(bibitem_keys() - cited_keys())
    assert not unused, f"bibliography entries never cited: {unused}"


def test_labels_and_refs_agree() -> None:
    labels: set[str] = set()
    refs: set[str] = set()
    for source in tex_sources():
        text = read(source)
        labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
        refs |= set(re.findall(r"\\ref\{([^}]+)\}", text))

    dangling = sorted(refs - labels)
    assert not dangling, f"\\ref to undefined labels (renders as ??): {dangling}"


def test_included_graphics_resolve() -> None:
    search = [LATEX / "figures", RESULTS / "figures"]
    for source in tex_sources():
        for graphic in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", read(source)):
            assert any((d / graphic).exists() for d in search), (
                f"{source.name} references a figure that is not on the graphicspath: {graphic}"
            )


def test_macros_agree_with_results() -> None:
    """Spot-check the generated macros against results/ so drift cannot survive."""
    macros = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", read(GENERATED / "macros.tex"))
    )
    metrics = pd.read_csv(RESULTS / "metrics.csv", index_col=0)
    inference = pd.read_csv(RESULTS / "significance.csv", index_col=0)
    config = json.loads(read(RESULTS / "config.json"))

    assert macros["benchSharpe"] == f"{metrics.loc['equal_weight', 'sharpe']:.3f}"
    assert macros["bestSharpe"] == f"{metrics['sharpe'].max():.3f}"
    assert macros["oosPeriods"] == str(config["oos_periods"])
    assert macros["nAssets"] == str(len(config["panel"]["assets"]))
    assert macros["nStrategies"] == str(len(metrics) - 1)

    n_beating = int((inference["p_ledoit_wolf"].dropna() < 0.05).sum())
    assert macros["nBeatingBenchmark"] == str(n_beating)


@pytest.mark.parametrize(
    "doc", [ROOT / "README.md", ROOT / "docs" / "postmortem.md"]
)
def test_markdown_results_blocks_are_delimited(doc: Path) -> None:
    text = read(doc)
    assert text.count("<!-- RESULTS:START -->") == 1, f"{doc.name} lost its results marker"
    assert text.count("<!-- RESULTS:END -->") == 1, f"{doc.name} lost its results marker"
    body = text.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    assert len(body.strip()) > 200, f"{doc.name} results block is empty -- run the generator"


def test_generator_is_idempotent(tmp_path: Path) -> None:
    """Re-running the generator must not change tracked files.

    If this fails, a committed document is out of date with respect to results/.
    """
    before = {p: p.read_bytes() for p in [ROOT / "README.md", ROOT / "docs" / "postmortem.md"]}
    before |= {p: p.read_bytes() for p in GENERATED.glob("*.tex")}

    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "make_report_tables.py")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, f"generator failed:\n{proc.stdout}\n{proc.stderr}"

    changed = [p.name for p, content in before.items() if p.read_bytes() != content]
    assert not changed, (
        f"regenerating changed {changed} -- committed docs are stale. "
        "Run: python experiments/make_report_tables.py"
    )
