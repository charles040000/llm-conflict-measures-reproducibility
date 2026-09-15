# Analysis scripts

## Main workflow

- `run_validation_analysis.py`: primary and full manual-validation metrics plus inter-annotator agreement.
- `run_baseline_models.py`: controls-only, three separate raw-share models, the multivariable binary model, and the event-context model.
- `run_calibration_models.py`: pooled validation rates, bounded share calibration, BCA, BCM, validation bootstrap, and the uniform latent-share benchmark.
- `run_joint_beta_model.py`: preferred beta latent-share HMC and integrated-MLE models.
- `build_estimator_comparison.py`: combines two-step uncertainty and preferred joint-model draws in one result file.
- `run_timing_diagnostic.py`: backward, contemporaneous, and forward outcome windows for physical violence.
- `run_robustness_models.py`: article thresholds, temporal aggregation, HAC bandwidth, additional LLM shares, and satellite outcomes.
- `generate_figures.py`: all non-Taliban descriptive, baseline, calibration, and robustness figures.

## Supporting code

- `model_utils.py`: OLS, actor-specific Newey-West covariance, diagnostics, FDR, direct calibration, BCA, and BCM.
- `joint_model.py`: NumPyro measurement/outcome models, HMC wrappers, quadrature, and integrated MLE.
- `build_actor_month_panel.py`: article/UCDP actor-month construction used by the full data pipeline.
- `temporal_aggregation_utils.py` and `run_temporal_robustness.py`: one-, two-, and four-week and calendar-month panels.
- `run_taliban_descriptive.py`: pre/post-takeover descriptive comparison.
- `build_viirs_*` and `run_viirs_*`: Earth Engine extraction and satellite-outcome regressions.
- `run_joint_prior_sensitivity.py`: residual-standard-deviation prior comparison.

All command-line scripts support `--help`. Defaults are repository-relative.
