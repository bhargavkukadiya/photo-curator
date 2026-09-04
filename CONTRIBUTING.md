# Contributing to Photo Curator

Thank you for your interest in contributing to **Photo Curator**! 🎉

We welcome contributions of all kinds: bug reports, documentation improvements, feature requests, and pull requests. Please take a moment to review these guidelines before getting started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Guidelines](#coding-guidelines)
- [Running Tests](#running-tests)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please treat all contributors and users with respect and courtesy.

---

## Getting Started

1. **Fork the repository** on GitHub: [https://github.com/bhargavkukadiya/photo-curator](https://github.com/bhargavkukadiya/photo-curator)
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/photo-curator.git
   cd photo-curator
   ```
3. **Set the upstream remote**:
   ```bash
   git remote add upstream https://github.com/bhargavkukadiya/photo-curator.git
   ```

---

## Development Setup

We recommend using a virtual environment:

```bash
# 1. Create a virtual environment with Python 3.9+
python3 -m venv .venv

# 2. Activate the virtual environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install in editable development mode with test dependencies
pip install -e ".[dev]"
```

Optional dependencies can also be installed:
```bash
# Optional: Facial emotion scoring (DeepFace)
pip install -e ".[emotion]"

# Optional: HEIC/HEIF photo support
pip install -e ".[heic]"

# Or install everything:
pip install -e ".[all,dev]"
```

---

## Coding Guidelines

- **Python Version:** Support Python 3.9, 3.10, 3.11, and 3.12.
- **Type Annotations:** Use PEP 484 and PEP 604 type annotations (`int | None`, `list[str]`, etc.). Include `from __future__ import annotations` at the top of all new modules.
- **Docstrings:** Document all public functions, classes, and CLI parameters using Google or PEP 257 style docstrings. Explain the *why* and operational constraints, not just the *what*.
- **Error Handling:** Avoid bare `except:` clauses. Catch specific exceptions and log helpful context.
- **Memory Safety:** When processing images, avoid keeping uncompressed full-resolution bitmaps in long-lived variables or batch lists.
- **Filesystem Operations:** Always adhere to the non-destructive two-phase commit paradigm: write to staging, validate, atomic commit, and rollback on error. Never overwrite untracked files without `--overwrite`.

---

## Running Tests

All unit tests run fast in-memory using mocked neural network models with zero network overhead:

```bash
# Run the complete test suite
pytest -v

# Run a specific test class or method
pytest test_album_selector.py -k "TestFilterNearDuplicates" -v

# Check for syntax errors across all files
python3 -m py_compile album_selector.py test_album_selector.py
```

Ensure all tests pass cleanly with **zero failures** and **zero new warnings** before opening a pull request.

---

## Pull Request Process

1. **Create a topic branch** from `main`:
   ```bash
   git checkout -b feature/my-awesome-improvement
   ```
2. **Make your changes** with clear, focused commits following [Conventional Commits](https://www.conventionalcommits.org/) (e.g. `feat:`, `fix:`, `docs:`, `test:`, `perf:`).
3. **Add or update unit tests** covering your new functionality or bug fix.
4. **Run the test suite** to ensure no regressions:
   ```bash
   pytest -v
   ```
5. **Push your branch** to your fork:
   ```bash
   git push origin feature/my-awesome-improvement
   ```
6. **Open a Pull Request** against the `main` branch of `bhargavkukadiya/photo-curator`.
7. Fill in the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) completely, describing the rationale, changes made, and verification steps.

---

## Reporting Issues

- **Bug Reports:** Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include your OS, Python version, exact command line arguments, and terminal traceback.
- **Feature Requests:** Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md). Clearly describe the problem you are trying to solve and your proposed solution.
- **Security Vulnerabilities:** Please review our [Security Policy](SECURITY.md) for instructions on confidential disclosure.
