# Working on Xenon

First-party C++, CUDA, Python, shell and CMake sources use tabs displayed at four columns, with a 100-column limit where supported. `.editorconfig`, `.clang-format`, `ruff.toml` and `.cmake-format.py` keep editors and formatters consistent. Clangd uses the repository's ClangFormat configuration.

Install the formatting tools in an isolated environment:

```sh
sudo apt install python3-venv ripgrep
python3 -m venv build/format-tools
build/format-tools/bin/pip install clang-format==23.1.3 ruff==0.17.0 cmakelang==0.6.13 shfmt-py==4.2.0
export PATH="$PWD/build/format-tools/bin:$PATH"
scripts/format.sh          # apply
scripts/format.sh --check  # verify
```

The command covers first-party source, tests and scene generators. Vendored code, generated lookup-table data, downloaded scenes and build outputs retain their original format. Python formatting preserves executable structure and data strings; docstring indentation follows the surrounding code.

Use `./xenon.sh` to build and render incrementally. [Getting started](GETTING_STARTED.md) describes dependencies and CMake options; [qualification](QUALIFICATION.md) records the test configurations and rendering checks. For an import change, run the affected Python/Blender regression and inspect a small render before increasing samples.
