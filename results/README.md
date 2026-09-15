# Results

Results are stored as tidy CSV or JSON files. They are included so deterministic tables and figures can be rebuilt without rerunning the computationally expensive HMC models.

- `validation/`: manual-reference classification metrics and inter-annotator agreement.
- `baseline/`: controls-only, separate-share, multivariable, and event-context regressions.
- `descriptive/taliban/`: pre/post-takeover summaries and monthly series.
- `calibration/uniform_joint/`: direct calibration, BCA, BCM, validation bootstrap, and benchmark joint-model output.
- `calibration/joint_beta/`: preferred beta latent-share joint model and numerical diagnostics.
- `calibration/sigma_prior_sensitivity/`: residual-scale prior sensitivity.
- `robustness/`: article-threshold, aggregation, timing, additional-label, and VIIRS results.

Run `python reproduce.py full` to replace the stored stochastic output with a fresh run.
