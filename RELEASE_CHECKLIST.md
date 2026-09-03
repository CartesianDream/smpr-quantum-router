# Public release checklist

- [ ] Replace the anonymous author and copyright-holder placeholders.
- [ ] Have the copyright holder approve a public software license.
- [ ] Replace `LICENSE-REVIEW-REQUIRED.txt` and update `pyproject.toml`.
- [ ] Insert the public repository URL and archival DOI in `CITATION.cff`.
- [ ] Run `python scripts/verify_public_release.py` with zero failures.
- [ ] Run all unit tests in the frozen Python/Qiskit environment.
- [ ] Run the five-case V5 smoke test and obtain 182 selected SWAP.
- [ ] Verify V5 exact and V6 primary reference equality.
- [ ] Freeze the V7 manifest before inspecting V7 routing results.
- [ ] Publish the historical archive separately; do not merge its source tree
      into this public-core package.

