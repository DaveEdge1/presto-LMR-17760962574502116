import cfr
import yaml
import os
import math
import numpy as np

# Maximum ensemble members per sequential run.
# Above this, nens is capped here and recon_seeds is expanded so that
# total ensemble count (nens × n_seeds) stays the same.
# At the current prior regrid (42×63), nens=100 uses ~2 GB on top of the
# ~5 GB prior download, comfortably within the 7 GB free-tier runner.
NENS_BATCH = 100

# Minimum observation error variance (floor on PSMmse).
# Prevents kdenom = varye + ob_err → 0 when a proxy has near-zero MSE
# (e.g. constant-value records, or perfectly-fitted OLS with few points).
MIN_R = 0.01

job_cfg = cfr.ReconJob()

# Load base config (all static settings baked into the image)
with open('lmr_configs.yml') as f:
    base_config = yaml.safe_load(f) or {}

# Merge user overrides if present (mounted from workflow as /app/user_config.yml)
user_config_path = 'user_config.yml'
if os.path.exists(user_config_path):
    with open(user_config_path) as f:
        user_overrides = yaml.safe_load(f) or {}
    base_config.update(user_overrides)
    print(f'Loaded user overrides: {list(user_overrides.keys())}')

# ── Auto-batch large ensemble sizes ──────────────────────────────────────────
nens  = base_config.get('nens', NENS_BATCH)
seeds = list(base_config.get('recon_seeds', [1]))

if nens > NENS_BATCH:
    n_batches = math.ceil(nens / NENS_BATCH)
    max_seed  = max(seeds)
    extra_seeds = [s + b * max_seed for b in range(1, n_batches) for s in seeds]
    base_config['nens']         = NENS_BATCH
    base_config['recon_seeds']  = seeds + extra_seeds
    print(f'Auto-batching: nens={nens} > {NENS_BATCH}; '
          f'running {n_batches} batches of {NENS_BATCH} '
          f'({len(base_config["recon_seeds"])} total seeds, '
          f'{NENS_BATCH * len(base_config["recon_seeds"])} total ensemble members)')
else:
    print(f'nens={nens} <= {NENS_BATCH}; running {len(seeds)} seed(s) as configured')

# Write merged config
with open('/tmp/merged_config.yml', 'w') as f:
    yaml.dump(base_config, f)

# ── Phase 1: prep (load data, calibrate PSMs) ────────────────────────────────
job_cfg.prep_da_cfg('/tmp/merged_config.yml', verbose=True)

# ── Phase 2: enforce minimum R floor ─────────────────────────────────────────
# PSMmse=0 → ob_err=0, combined with varye=0 (flat PSM slope) → kdenom=0
# → Kalman gain blows up. Apply MIN_R to all calibrated records.
n_floor = 0
for pid, pobj in job_cfg.proxydb.records.items():
    r_val = getattr(pobj, 'R', None)
    if r_val is not None and np.isfinite(r_val) and r_val < MIN_R:
        pobj.R = MIN_R
        n_floor += 1
if n_floor:
    print(f'R floor: raised {n_floor} record(s) from PSMmse < {MIN_R} to {MIN_R}')

# ── Phase 3: run DA ───────────────────────────────────────────────────────────
cfg = job_cfg.configs
job_cfg.run_da_mc(
    recon_period=cfg['recon_period'],
    recon_loc_rad=cfg['recon_loc_rad'],
    recon_timescale=cfg.get('recon_timescale', 1),
    recon_seeds=cfg.get('recon_seeds', [0]),
    assim_frac=cfg.get('assim_frac', 0.75),
    compress_params=cfg.get('compress_params', {'zlib': True}),
    output_full_ens=cfg.get('output_full_ens', False),
    output_indices=cfg.get('output_indices', None),
    verbose=True,
)
