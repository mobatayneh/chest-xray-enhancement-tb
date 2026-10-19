# Colab Pip Bootstrap Repair

The user's first Colab setup failed before training because the existing isolated Python interpreter had no pip module. Setup had checked the interpreter file but had not checked pip availability. Repeating the old setup cell could also collide with its existing exclusive log filename.

The updated notebook checks pip in new and existing environments, tries offline `ensurepip`, falls back to the notebook host pip's `--python TARGET install pip`, and verifies the target pip. New environments are created explicitly without pip before this shared bootstrap path. Each setup attempt receives a separate log directory. Model code, numerical dependencies, datasets, splits, experiment settings, and checkpoints are unchanged.

## Executed regression checks

- Twelve notebook-focused tests passed, including the six existing notebook tests and six additional bootstrap/retry checks.
- Two real pipless local virtual environments were repaired without package-index access. One used offline ensurepip; the other used a real host pip targeting a second pipless environment while ensurepip failure was simulated.
- Existing-pip reuse, bootstrap failures, propagation of failure, target-only installation arguments, and preservation of earlier setup logs were tested.
- Full-suite results are saved separately in `validation/pip_recovery_pytest.log` and `validation/pip_recovery_tests.json`; the original delivery evidence is retained.
- Full-project Ruff validation passed, including the regenerated notebook.

These checks validate local bootstrap behavior. They do not establish successful CUDA wheel installation, Google Drive operation, or real-data training in the user's Colab session. The user should rerun setup and preserve the actual output.

The fallback is documented by [pip's official interpreter-management guide](https://pip.pypa.io/en/stable/topics/python-option/). The temporary repair cell for the initial notebook is included in [COLAB_GUIDE.md](COLAB_GUIDE.md).
