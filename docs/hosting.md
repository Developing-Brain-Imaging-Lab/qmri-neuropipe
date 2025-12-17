# Hosting on Read the Docs

This project is configured for extensive documentation hosting on [Read the Docs](https://readthedocs.org/) (RTD).

## Configuration

The build process is controlled by `.readthedocs.yaml` in the repository root.

### Key Settings
*   **OS**: Ubuntu 22.04
*   **Python**: 3.10
*   **Dependencies**:
    *   Installs project dependencies via `pip install .[dev]` (or similar).
    *   Installs documentation dependencies from `docs/requirements.txt`.

## Connecting to Read the Docs

1.  **Log in** to your Read the Docs account (e.g., via GitHub).
2.  **Import a Project**:
    *   Go to "My Projects" -> "Import a Project".
    *   Select the `qmri-neuropipe` repository.
3.  **Advanced Settings** (Usually auto-detected, but good to verify):
    *   **Default branch**: `main` (or your default branch).
    *   **Documentation type**: Sphinx Html.
    *   **Config file**: `.readthedocs.yaml` (RTD should find this automatically).

## Triggering Builds

*   **Automatic**: RTD uses webhooks to trigger a build whenever you push to the configured branch.
*   **Manual**: You can trigger a build manually from the project dashboard.

## Troubleshooting

If the build fails:
1.  Check the **Build Logs** on the RTD dashboard.
2.  Verify `docs/requirements.txt` contains all Sphinx extensions used in `conf.py`.
3.  Ensure `qmri-neuropipe` installs correctly (check `pyproject.toml` dependencies).
