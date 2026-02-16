<!-- This disables the "First line in file should be a top level heading" rule -->
<!-- markdownlint-disable MD041 -->
<a href="https://github.com/alexandrainst/multi_ifeval">
<img
 src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/alexandra/alexandra-logo.jpeg"
 width="239"
 height="175"
 align="right"
 alt="Alexandra Institute Logo"
/>
</a>

# Multi Ifeval

An automatically generated multilingual version of IFEval, for 300+ languages.

______________________________________________________________________
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/multi_ifeval/tree/main/tests)
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/multi_ifeval)
[![License](https://img.shields.io/github/license/alexandrainst/multi_ifeval)](https://github.com/alexandrainst/multi_ifeval/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/multi_ifeval)](https://github.com/alexandrainst/multi_ifeval/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg)](https://github.com/alexandrainst/multi_ifeval/blob/main/CODE_OF_CONDUCT.md)

Developer:

- Dan Saattrup Smart (<dan.smart@alexandra.dk>)

## Setup

### Installation

1. Run `make install`, which sets up a virtual environment and all Python dependencies
   therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Quickstart

Run `uv run src/scripts/translate_ifeval.py` to translate the IFEval dataset to
different languages. By default this uses the Gemini 3 Pro Preview model, which requires
you to have set the `GEMINI_API_KEY` environment variable. You can also specify a
different model with the `--model` flag.
