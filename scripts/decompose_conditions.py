#!/usr/bin/env python3
"""Read-only decomposition of committed RACC_SCALING_6000_V1 predictions."""
import csv
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1] / 'inputs/racc_artifacts'
SOURCE = BASE / 'scaling_6000_v1_results'
CACHE = BASE / 'scaling_6000_v1_eval1000_cache'
OUT = Path(__file__).resolve().parents[1] / 'results_local/condition_decomposition'
CONDITIONS = ('clean', 'focal_keep30', 'focal_keep20', 'focal_keep10',
              'context_keep30', 'context_keep20', 'context_keep10')
SIZES = (700, 1500, 3000, 5000)
SEEDS = (2026, 2027, 2028, 2029, 2030)
KEYS = {'QCNet': 'QCNet', 'Pairwise': 'Pairwise', 'DU': 'DirectUtility'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(name, data):
    path = OUT / name
    if path.exists():
        assert path.read_bytes() == data, f'Existing artifact differs: {path}'
        return
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_json(name, data):
    save(name, (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def save_csv(name, rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    save(name, f.getvalue().encode())


def intervention(modes, baseline, values, baseline_values):
    changed = modes != baseline
    gain = baseline_values - values
    good = changed & (gain > 1e-6)
    bad = changed & (gain < -1e-6)
    count = int(changed.sum())
    return dict(intervention_rate=float(changed.mean()),
                beneficial_rate_all=float(good.mean()), harmful_rate_all=float(bad.mean()),
                beneficial_rate_among_interventions=float(good.sum() / count) if count else 0.,
                harmful_rate_among_interventions=float(bad.sum() / count) if count else 0.,
                mean_gain=float(gain[good].mean()) if good.any() else 0.,
                mean_loss=float(-gain[bad].mean()) if bad.any() else 0.,
                mean_net_gain_among_interventions=float(gain[changed].mean()) if count else 0.)


def main():
    assert np.__version__ == '1.26.4' and torch.__version__ == '2.4.1+cu124'
    torch.set_num_threads(1)
    original_path = SOURCE / 'primary_scaling_results.json'
    original = json.loads(original_path.read_text())
    lock_path = BASE / 'scaling_6000_v1_primary/training_lock.json'
    freeze_path = CACHE / 'eval1000_cache_freeze_manifest.json'
    selection_path = BASE / 'scaling_6000_v1_selection/selection_manifest.json'
    lock_sha, freeze_sha = sha(lock_path), sha(freeze_path)
    freeze = json.loads(freeze_path.read_text())
    assert original['status'] == 'PRIMARY_EVAL1000_COMPLETE'
    assert original['conditions'] == list(CONDITIONS) and original['seeds'] == list(SEEDS)
    assert original['training_lock_sha256'] == lock_sha == freeze['training_lock_sha256']
    assert original['eval_freeze_sha256'] == freeze_sha
    assert freeze['status'] == 'EVAL1000_FINAL_AUDIT_PASS' and freeze['records'] == 7000
    assert sha(selection_path) == original['selection_sha256']
    ids = json.loads(selection_path.read_text())['ordered_6000_ids'][5000:6000]
    assert len(ids) == len(set(ids)) == 1000
    provenance = {'source_results_sha256': sha(original_path), 'training_lock_sha256': lock_sha,
                  'eval_freeze_sha256': freeze_sha, 'selection_sha256': sha(selection_path),
                  'script_sha256': sha(Path(__file__)), 'cache_chunks': [], 'prediction_commits': []}
    # Read only mode errors from the original audited eval tars; no model imports/inference.
    errors = {}
    for chunk in range(50):
        cp = CACHE / f'chunk_{chunk:04d}.commit.json'
        commit = json.loads(cp.read_text())
        frozen_chunk = next(x for x in freeze['chunks'] if x['chunk_index'] == chunk)
        assert sha(cp) == frozen_chunk['commit_sha256']
        assert commit['scenario_ids'] == ids[chunk * 20:chunk * 20 + 20]
        tar_path = CACHE / commit['tar_file']
        assert tar_path.parent == CACHE
        assert sha(tar_path) == commit['tar_sha256'] == frozen_chunk['tar_sha256']
        provenance['cache_chunks'].append({'chunk': chunk, 'commit_sha256': sha(cp),
                                          'tar_sha256': commit['tar_sha256']})
        with tarfile.open(tar_path, 'r:*') as archive:
            members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
            for entry in commit['records']:
                raw = archive.extractfile(members[Path(entry['file']).name]).read()
                assert digest(raw) == entry['sha256']
                record = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                key = (str(record['scenario_id']), str(record['condition']))
                assert key[0] in commit['scenario_ids'] and key[1] == entry['condition']
                assert record['split'] == 'scaling_eval1000' and key not in errors
                errors[key] = (record['ade_by_mode'].float().numpy(), record['fde_by_mode'].float().numpy())
        if (chunk + 1) % 25 == 0:
            print(f'CACHE_READ {chunk + 1}/50', flush=True)
    assert set(errors) == {(sid, c) for sid in ids for c in CONDITIONS}
    ade = np.stack([errors[(sid, c)][0] for sid in ids for c in CONDITIONS]).reshape(1000, 7, 6)
    fde = np.stack([errors[(sid, c)][1] for sid in ids for c in CONDITIONS]).reshape(1000, 7, 6)
    assert np.isfinite(ade).all() and np.isfinite(fde).all()
    draws = np.random.default_rng(2026).integers(0, 1000, size=(10000, 1000))

    def boot(scene):
        assert scene.shape == (1000,) and np.isfinite(scene).all()
        lo, hi = np.quantile(scene[draws].mean(axis=1), (.025, .975))
        return dict(difference=float(scene.mean()), ci95_lower=float(lo), ci95_upper=float(hi))

    metrics, seed_metrics, intervals, interventions, tails = [], [], [], [], []
    differences = {}
    baseline_ref = None
    original_methods = {'QCNet': 'QCNet', 'Pairwise': 'Original Pairwise', 'DU': 'Direct Utility'}
    for n in SIZES:
        chosen = {m: [] for m in KEYS}
        for seed in SEEDS:
            folder = SOURCE / f'N{n}/seed{seed}'
            cp, pp = folder / 'commit.json', folder / 'predictions.npz'
            commit = json.loads(cp.read_text())
            assert commit['n'] == n and commit['seed'] == seed and commit['records'] == 7000
            assert commit['training_lock_sha256'] == lock_sha and commit['eval_freeze_sha256'] == freeze_sha
            assert commit['script_sha256'] == original['script_sha256']
            assert commit['predictions_sha256'] == sha(pp)
            provenance['prediction_commits'].append({'n': n, 'seed': seed, 'commit_sha256': sha(cp),
                                                      'predictions_sha256': commit['predictions_sha256']})
            with np.load(pp, allow_pickle=False) as saved:
                for m, k in KEYS.items():
                    modes = saved[k]
                    assert modes.shape == (7000,) and np.all((modes >= 0) & (modes < 6))
                    chosen[m].append(modes.reshape(1000, 7).astype(np.int64))
        chosen = {m: np.stack(v) for m, v in chosen.items()}
        if baseline_ref is None:
            baseline_ref = chosen['QCNet'][0]
        assert np.all(chosen['QCNet'] == baseline_ref)
        values = {m: np.take_along_axis(np.broadcast_to(ade, (5, 1000, 7, 6)), v[..., None], -1)[..., 0]
                  for m, v in chosen.items()}
        fvalues = {m: np.take_along_axis(np.broadcast_to(fde, (5, 1000, 7, 6)), v[..., None], -1)[..., 0]
                   for m, v in chosen.items()}
        differences[n] = (values['Pairwise'] - values['DU']).mean(axis=0)
        for ci, condition in enumerate(CONDITIONS):
            b = boot(differences[n][:, ci])
            intervals.append(dict(N=n, condition=condition, **b,
                                  DU_interval_supported=b['ci95_lower'] > 0,
                                  Pairwise_interval_supported=b['ci95_upper'] < 0))
            for m in KEYS:
                a, f = values[m][:, :, ci], fvalues[m][:, :, ci]
                expected = next(x for x in original['curves'] if x['n'] == n and x['method'] == original_methods[m])
                assert np.isclose(float(a.mean()), expected['condition_top1ADE'][condition], atol=1e-7)
                assert np.isclose(float(f.mean()), expected['condition_top1FDE'][condition], atol=1e-7)
                metrics.append(dict(N=n, condition=condition, method=m, top1ADE=float(a.mean()), top1FDE=float(f.mean())))
                diag = intervention(chosen[m][:, :, ci], chosen['QCNet'][:, :, ci], a, values['QCNet'][:, :, ci])
                interventions.append(dict(N=n, condition=condition, method=m, **diag))
                p90, p95 = np.quantile(a.mean(axis=0), (.9, .95))
                tails.append(dict(N=n, condition=condition, method=m, P90_ADE=float(p90), P95_ADE=float(p95)))
                for j, seed in enumerate(SEEDS):
                    sd = intervention(chosen[m][j, :, ci], chosen['QCNet'][j, :, ci], a[j], values['QCNet'][j, :, ci])
                    q90, q95 = np.quantile(a[j], (.9, .95))
                    seed_metrics.append(dict(N=n, condition=condition, seed=seed, method=m,
                                             top1ADE=float(a[j].mean()), top1FDE=float(f[j].mean()),
                                             Pairwise_minus_DU=float((values['Pairwise'][j, :, ci] - values['DU'][j, :, ci]).mean()),
                                             P90_ADE=float(q90), P95_ADE=float(q95), **sd))
        aggregate = next(x for x in original['comparisons'] if x['n'] == n and
                         x['difference'] == 'ADE_Original Pairwise - ADE_Direct Utility')
        assert np.isclose(differences[n].mean(), aggregate['mean_scene_difference'], atol=1e-7)
        check = boot(differences[n].mean(axis=1))
        assert np.allclose([check['ci95_lower'], check['ci95_upper']],
                           [aggregate['ci95_lower'], aggregate['ci95_upper']], atol=1e-7)

    d = differences[5000]
    total = float(d.mean())
    contributions = sorted([dict(condition=c, difference=float(d[:, i].mean()),
                                  contribution_to_7condition_mean=float(d[:, i].mean() / 7),
                                  share_of_net_advantage=float(d[:, i].mean() / (7 * total)) if total else None)
                            for i, c in enumerate(CONDITIONS)], key=lambda r: r['difference'], reverse=True)
    group_contrasts = {'corruption_minus_clean_N5000': boot(d[:, 1:].mean(1) - d[:, 0]),
                       'focal_minus_context_N5000': boot(d[:, 1:4].mean(1) - d[:, 4:7].mean(1))}
    scaling = differences[5000] - differences[700]
    group_contrasts['corruption_minus_clean_scaling_change'] = boot(scaling[:, 1:].mean(1) - scaling[:, 0])
    severity = {}
    for group, idx in [('focal', (1, 2, 3)), ('context', (4, 5, 6))]:
        severity[group] = {}
        for n in SIZES:
            means = differences[n][:, idx].mean(0)
            severity[group][str(n)] = {'keep30_20_10_differences': means.tolist(),
                                      'monotonic_numerical_increase': bool(np.all(np.diff(means) > 0)),
                                      'keep10_minus_keep30': boot(differences[n][:, idx[2]] - differences[n][:, idx[0]])}
        severity[group]['severity_scaling_amplification'] = boot(scaling[:, idx[2]] - scaling[:, idx[0]])
    changes = sorted([dict(condition=c, **boot(scaling[:, i]),
                           N700_difference=float(differences[700][:, i].mean()),
                           N5000_difference=float(differences[5000][:, i].mean()))
                      for i, c in enumerate(CONDITIONS)], key=lambda x: x['difference'], reverse=True)
    findings = {
        'largest_N5000_contribution': contributions[0]['condition'],
        'clean_vs_corruption': group_contrasts['corruption_minus_clean_N5000'],
        'focal_vs_context': group_contrasts['focal_minus_context_N5000'],
        'strongest_scaling_shift': changes[0]['condition'],
        'corruption_scaling_amplification_interval_supported':
            group_contrasts['corruption_minus_clean_scaling_change']['ci95_lower'] > 0,
        'severity_amplification_interval_supported': {
            g: severity[g]['severity_scaling_amplification']['ci95_lower'] > 0
            for g in ('focal', 'context')},
        'diversity_causal_explanation': 'unresolved; observational associations only',
    }
    summary = dict(study='RACC_CONDITION_DECOMPOSITION_V1', status='PASS', conditions=list(CONDITIONS),
                   sizes=list(SIZES), seeds=list(SEEDS), scenes=1000,
                   bootstrap={'replicates': 10000, 'seed': 2026, 'CI': '95% percentile',
                              'unit': 'scene, five seeds averaged within condition',
                              'multiplicity': 'exploratory unadjusted intervals'},
                   N5000_contributions=contributions, group_contrasts=group_contrasts,
                   severity=severity, scaling_changes=changes,
                   findings=findings,
                   interpretation='Condition-level associations cannot establish a causal effect of training-condition diversity; causal explanation remains unresolved.',
                   tail_definition='P90/P95 of the 1000 scene ADE values after averaging five seeds; per-seed quantiles also provided',
                   intervention_definition='beneficial/harmful threshold +/-1e-6 ADE; rates use all records and interventions separately; mean gain/loss condition on beneficial/harmful interventions',
                   fresh500_opened=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [('condition_metrics.csv', metrics), ('condition_seed_metrics.csv', seed_metrics),
                       ('condition_bootstrap.csv', intervals), ('condition_intervention_metrics.csv', interventions),
                       ('condition_tail_metrics.csv', tails)]:
        save_csv(name, rows)
    save_json('condition_trend_summary.json', summary)
    save_json('source_manifest.json', provenance)
    save('decompose_conditions.py', Path(__file__).read_bytes())
    lines = ['# QCNet seven-condition decomposition', '', 'PASS. Read-only analysis of original committed predictions and cached mode errors.',
             '', '| Condition | N5000 Pairwise−DU | 95% CI |', '|---|---:|---|']
    for r in intervals:
        if r['N'] == 5000:
            lines.append(f"| {r['condition']} | {r['difference']:+.6f} | [{r['ci95_lower']:+.6f}, {r['ci95_upper']:+.6f}] |")
    lines += ['', '## Contribution and severity', '',
              f"Largest N5000 contribution: {findings['largest_N5000_contribution']}. Strongest N700→5000 shift: {findings['strongest_scaling_shift']}.",
              '',
              'Each condition has equal weight: its contribution to the pooled advantage is its difference divided by seven. Shares are signed and can exceed 100% when other conditions offset gains.',
              '', '```json', json.dumps({'contributions': contributions, 'group_contrasts': group_contrasts,
                                       'severity': severity, 'scaling_changes': changes}, indent=2), '```', '',
              summary['interpretation'], '', 'All intervals are exploratory and unadjusted for multiple comparisons.',
              summary['tail_definition'], summary['intervention_definition'], '',
              'No training, model inference, checkpoint selection, or fresh500 access. Historical files were read only.',
              '', 'Resume with the pinned environment and mounted Drive:',
              '`python scripts/decompose_conditions.py`', '']
    save('summary.md', '\n'.join(lines).encode())
    save_json('output_manifest.json', {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'output_manifest.json'})
    print('CONDITION_DECOMPOSITION_PASS', flush=True)
    for r in intervals:
        if r['N'] == 5000:
            print(r['condition'], *(round(r[k], 6) for k in ('difference', 'ci95_lower', 'ci95_upper')), flush=True)


if __name__ == '__main__':
    main()
