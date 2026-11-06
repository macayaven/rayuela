# Contributing to Project Rayuela

Start with the [interactive analysis](https://carloscrespomacaya.com/rayuela/)
and the README's findings and limitations. A clear report about one confusing
figure or setup step is useful; you do not need to run a large model to help.

## Useful first contributions

- Fix a broken public link or clarify one installation step.
- Improve a figure label or explanation while preserving what the measurement
  establishes and what it leaves open.
- Report a reproduction difference with the command, software versions and
  expected versus observed result.
- Propose a small public-domain or openly licensed example before adding a new
  corpus or analysis path.

Use [GitHub Issues](https://github.com/macayaven/rayuela/issues/new) for a report or
proposal. Include the relevant file or figure and one concrete example. Keep
credentials, private paths and copyrighted corpus text out of reports and patches.
Never attach a novel, raw chapter, private dataset or model credential.

## Edit and verify

Keep changes small. The README describes the full GPU-backed environment.
For documentation-export and publishing changes, use the development tools:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/test_md_to_html.py tests/test_prepare_ghpages.py --no-cov
python3 -m pre_commit run --all-files
```

The pre-commit configuration pins its formatter and defines the checked files.
Keep that scope rather than reformatting unrelated analysis scripts.

Changes to analysis code also need the relevant tests, the full coverage gate
(`python3 -m pytest`) and the checks in `.github/workflows/`. Model-dependent
checks need an explicit local setup; describe any check you did not run.

Preserve the article sources and the generated-site workflow in the README.
Avoid replacing an interpretive boundary with a stronger claim about authorial
intent, literary value or model comprehension.

## Rights and attribution

Contribute original Python code under Apache-2.0, as described in
[LICENSE](LICENSE) and [NOTICE](NOTICE), and preserve existing attribution.
Documentation and figure changes retain their applicable existing rights.
Discuss any new content license with the maintainer before including it.
