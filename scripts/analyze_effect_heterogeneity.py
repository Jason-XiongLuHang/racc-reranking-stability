"""Read frozen choices and saved ADE only; no model imports or inference."""
import collections
import csv
import hashlib
import io
import json
import pickle
import tarfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_local/effect_heterogeneity'
DOWNLOADS = ROOT / 'inputs'
CONDITIONS = ['clean', 'focal_keep30', 'focal_keep20', 'focal_keep10',
              'context_keep30', 'context_keep20', 'context_keep10']
SEEDS = list(range(2026, 2031))
INPUTS = {
    'development_results': 'development_results.zip',
    'development_cache': 'development_cache.zip',
    'fresh_results': 'fresh500_results.zip',
    'confirmation_results': 'confirmation_results.zip',
    'confirmation_cache': 'confirmation_cache.zip',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor_reference(storage, offset, shape, stride, *unused):
    return storage, offset, tuple(shape), tuple(stride)


class MetadataReader(pickle.Unpickler):
    """Restricted reader for existing torch ZIP metadata; materialize ADE only.

    No arbitrary globals and no Torch runtime. Unsupported serialization stops.
    """
    def find_class(self, module, name):
        if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
            return tensor_reference
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        dtypes = {'FloatStorage': '<f4', 'DoubleStorage': '<f8', 'BoolStorage': '?',
                  'LongStorage': '<i8', 'IntStorage': '<i4', 'ByteStorage': 'u1'}
        if module == 'torch' and name in dtypes:
            return dtypes[name]
        raise ValueError(f'Unsupported pickle global: {module}.{name}')

    def persistent_load(self, pid):
        kind, dtype, key, device, numel = pid
        assert kind == 'storage' and device == 'cpu' and isinstance(numel, int)
        return str(key), dtype, numel


def ade_record(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        pk = next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix = pk[:-len('data.pkl')]
        if prefix + 'byteorder' in z.namelist():
            assert z.read(prefix + 'byteorder') == b'little'
        meta = MetadataReader(io.BytesIO(z.read(pk))).load()
        (key, dtype, numel), offset, shape, stride = meta['ade_by_mode']
        assert shape == (6,) and stride == (1,) and dtype in ('<f4', '<f8')
        values = np.frombuffer(z.read(prefix + 'data/' + key), dtype=dtype)
        assert len(values) == numel and 0 <= offset <= numel - 6
        ade = values[offset:offset + 6].astype(np.float64)
        assert np.isfinite(ade).all() and (ade >= 0).all()
        return str(meta['scenario_id']), meta['condition'], ade


def cache(z):
    freeze_name = next(n for n in z.namelist() if n.endswith('/eval1000_cache_freeze_manifest.json'))
    prefix = freeze_name.rsplit('/', 1)[0] + '/'
    freeze_raw = z.read(freeze_name)
    freeze = json.loads(freeze_raw)
    assert freeze['scenes'] == 1000 and freeze['records'] == 7000
    assert freeze['conditions'] == CONDITIONS and len(freeze['chunks']) == 50
    ids, values = [], {}
    for expected in sorted(freeze['chunks'], key=lambda x: x['chunk_index']):
        i = expected['chunk_index']
        commit_raw = z.read(prefix + f'chunk_{i:04d}.commit.json')
        assert sha(commit_raw) == expected['commit_sha256']
        commit = json.loads(commit_raw)
        assert commit['chunk_index'] == i and len(commit['scenario_ids']) == 20
        ids.extend(commit['scenario_ids'])
        tar_raw = z.read(prefix + commit['tar_file'])
        assert sha(tar_raw) == expected['tar_sha256'] == commit['tar_sha256']
        with tarfile.open(fileobj=io.BytesIO(tar_raw)) as archive:
            members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
            assert len(commit['records']) == 140
            for entry in commit['records']:
                raw = archive.extractfile(members[entry['file']]).read()
                assert sha(raw) == entry['sha256']
                sid, condition, ade = ade_record(raw)
                key = sid, condition
                assert sid in commit['scenario_ids'] and condition == entry['condition'] and key not in values
                values[key] = ade
    assert len(ids) == len(set(ids)) == 1000
    assert set(values) == {(sid, c) for sid in ids for c in CONDITIONS}
    return ids, np.stack([values[(sid, c)] for sid in ids for c in CONDITIONS]).reshape(1000, 7, 6), sha(freeze_raw)


def predictions(z, ade, freeze_hash, sizes):
    prefix = next(n for n in z.namelist() if n.endswith('/primary_scaling_results.json')).rsplit('/', 1)[0] + '/'
    result_raw = z.read(prefix + 'primary_scaling_results.json')
    result = json.loads(result_raw)
    assert result['eval_freeze_sha256'] == freeze_hash
    assert result['conditions'] == CONDITIONS and result['seeds'] == SEEDS and result['scenes'] == 1000
    differences, commits = {}, {}
    for n in sizes:
        selected = {'Pairwise': [], 'DirectUtility': []}
        commits[n] = []
        for seed in SEEDS:
            cp = prefix + f'N{n}/seed{seed}/commit.json'
            commit = json.loads(z.read(cp))
            raw = z.read(cp.replace('commit.json', 'predictions.npz'))
            assert sha(raw) == commit['predictions_sha256']
            assert commit['n'] == n and commit['seed'] == seed and commit['eval_freeze_sha256'] == freeze_hash
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                for method in selected:
                    choice = data[method].reshape(1000, 7)
                    assert choice.dtype.kind in 'iu' and (choice >= 0).all() and (choice < 6).all()
                    selected[method].append(np.take_along_axis(ade, choice[..., None], -1)[..., 0])
            commits[n].append(commit)
        selected = {m: np.stack(v) for m, v in selected.items()}
        for method, name in [('Pairwise', 'Original Pairwise'), ('DirectUtility', 'Direct Utility')]:
            target = next(c for c in result['curves'] if c['n'] == n and c['method'] == name)
            assert abs(selected[method].mean() - target['top1ADE']) < 1e-6
            assert np.allclose(selected[method].mean((1, 2)), target['seed_top1ADE'], atol=1e-6, rtol=0)
        differences[n] = (selected['Pairwise'] - selected['DirectUtility']).mean(0)
    return differences, commits, sha(result_raw)


def paired_bootstrap(x):
    rng = np.random.default_rng(2026)
    return np.concatenate([x[rng.integers(0, len(x), (1000, len(x)))].mean(1) for _ in range(10)])


def independent_bootstrap(x, y):
    rng = np.random.default_rng(2026)
    reps = []
    for _ in range(10):
        # Distinct independent draws even when the two set sizes are equal.
        bx = x[rng.integers(0, len(x), (1000, len(x)))].mean(1)
        by = y[rng.integers(0, len(y), (1000, len(y)))].mean(1)
        reps.append(bx - by)
    return np.concatenate(reps)


def row(name, tier, x, y=None):
    reps = paired_bootstrap(x) if y is None else independent_bootstrap(x, y)
    low, high = np.quantile(reps, [0.025, 0.975])
    mean = x.mean() if y is None else x.mean() - y.mean()
    return {'contrast': name, 'role': tier, 'estimate_m': float(mean),
            'ci95_lower_m': float(low), 'ci95_upper_m': float(high),
            'interval_supported': bool(low > 0 or high < 0),
            'design': 'paired_scene' if y is None else 'independent_scene_sets',
            'scenes_a': len(x), 'scenes_b': '' if y is None else len(y),
            'fixed_seeds': 5, 'bootstrap_replicates': 10000, 'rng_seed': 2026}, reps


def write_csv(name, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    write(name, stream.getvalue().encode())


def write(name, raw):
    p = OUT / name
    if p.exists():
        assert p.read_bytes() == raw, f'Differing existing output: {p}'
    else:
        temporary = p.with_suffix(p.suffix + '.pending')
        temporary.write_bytes(raw)
        temporary.replace(p)


def main():
    assert np.allclose(paired_bootstrap(np.full(7, .2)), .2)
    assert np.allclose(independent_bootstrap(np.full(7, .3), np.full(5, .1)), .2)
    # Pairing must preserve a constant within-scene effect, despite varying scene difficulty.
    a = np.array([1., 4., 8.])
    assert np.allclose(paired_bootstrap((a + .2) - a), .2)
    OUT.mkdir(parents=True, exist_ok=True)
    archives = {k: zipfile.ZipFile(DOWNLOADS / v) for k, v in INPUTS.items()}
    dev_ids, dev_ade, dev_freeze = cache(archives['development_cache'])
    dev, dev_commits, dev_result_hash = predictions(archives['development_results'], dev_ade, dev_freeze, [700, 1500, 3000, 5000])
    confirm_ids, confirm_ade, confirm_freeze = cache(archives['confirmation_cache'])
    confirm, confirm_commits, confirm_result_hash = predictions(archives['confirmation_results'], confirm_ade, confirm_freeze, [5000])
    f = archives['fresh_results']
    hashes = json.loads(f.read('artifact_hashes.json'))
    for name in ['scene_level_paired_results.csv', 'primary_metrics.json', 'selection_manifest.json', 'execution_lock.json']:
        assert sha(f.read(name)) == hashes[name]
    fresh_primary = json.loads(f.read('primary_metrics.json'))
    lock = json.loads(f.read('execution_lock.json'))
    assert fresh_primary['execution_lock_sha256'] == sha(f.read('execution_lock.json'))
    fresh_selection = json.loads(f.read('selection_manifest.json'))
    assert lock['selection_sha256'] == sha(f.read('selection_manifest.json'))
    fresh_rows = list(csv.DictReader(io.StringIO(f.read('scene_level_paired_results.csv').decode(), newline='')))
    fresh_ids = [r['scenario_id'] for r in fresh_rows]
    fresh = np.array([float(r['Pairwise_minus_DU']) for r in fresh_rows])
    assert len(fresh_ids) == len(set(fresh_ids)) == 500 and fresh_ids == fresh_selection['fresh500_ids']
    assert abs(fresh.mean() - fresh_primary['primary_bootstrap']['difference']) < 1e-7
    for a_ids, b_ids in [(dev_ids, fresh_ids), (dev_ids, confirm_ids), (confirm_ids, fresh_ids)]:
        assert not set(a_ids) & set(b_ids), 'Independent sets overlap'
    for i, seed in enumerate(SEEDS):
        for method, key in [('pairwise', 'pairwise_checkpoint_sha256'), ('direct_utility', 'direct_utility_checkpoint_sha256')]:
            expected = dev_commits[5000][i][key]
            assert confirm_commits[5000][i][key] == expected
            item = next(c for c in lock['checkpoints'] if c['method'] == method and c['seed'] == seed)
            assert item['checkpoint_sha256'] == expected
    scene_d = {n: d.mean(1) for n, d in dev.items()}
    scale_rows, reps = [], {}
    for high, low, tier in [(5000, 700, 'primary'), (1500, 700, 'secondary'), (3000, 700, 'secondary'), (5000, 1500, 'secondary')]:
        r, rep = row(f'd_{high}-d_{low}', tier, scene_d[high] - scene_d[low])
        scale_rows.append(r)
        reps[r['contrast']] = rep
    condition, rep = row('d_corruption-d_clean', 'secondary', dev[5000][:, 1:].mean(1) - dev[5000][:, 0])
    condition.update({'d_clean_m': float(dev[5000][:, 0].mean()), 'd_corruption_m': float(dev[5000][:, 1:].mean())})
    reps['condition'] = rep
    cross_rows = []
    for name, tier, values in [('d_dev-d_fresh', 'requested_cross_set', fresh), ('d_dev-d_confirm_same_N5000', 'secondary', confirm[5000].mean(1))]:
        r, rep = row(name, tier, scene_d[5000], values)
        r.update({'d_dev_m': float(scene_d[5000].mean()), 'd_other_m': float(values.mean()), 'same_checkpoint_hashes': True})
        cross_rows.append(r)
        reps[name] = rep
    write_csv('scale_interaction.csv', scale_rows)
    write_csv('condition_interaction.csv', [condition])
    write_csv('cross_set_effect_difference.csv', cross_rows)
    scene_rows = [{'dataset': 'development', 'scenario_id': sid, **{f'd_{n}_m': float(scene_d[n][i]) for n in scene_d},
                   'd_clean_N5000_m': float(dev[5000][i, 0]), 'd_corruption_N5000_m': float(dev[5000][i, 1:].mean())} for i, sid in enumerate(dev_ids)]
    write_csv('development_scene_effects.csv', scene_rows)
    write_csv('other_set_scene_effects.csv', [{'dataset': dataset, 'scenario_id': sid, 'd_5000_m': float(values[i])}
              for dataset, ids, values in [('fresh500', fresh_ids, fresh), ('confirm1000', confirm_ids, confirm[5000].mean(1))] for i, sid in enumerate(ids)])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **reps)
    write('bootstrap_replicates.npz', buffer.getvalue())
    audit = {'status': 'PASS', 'original_results_unchanged': True, 'training_runs': 0, 'model_inference_runs': 0,
             'new_data_opened': False, 'new_holdout_opened': False, 'second_dataset_expansion': 'closed',
             'numpy_version': np.__version__, 'script_sha256': sha(Path(__file__).read_bytes()),
             'archives': {k: {'path': str(DOWNLOADS / name), 'sha256': sha((DOWNLOADS / name).read_bytes())} for k, name in INPUTS.items()},
             'development_result_sha256': dev_result_hash, 'confirmation_result_sha256': confirm_result_hash,
             'cache_freeze_hashes': {'development': dev_freeze, 'confirmation': confirm_freeze},
             'fresh_scene_csv_sha256': sha(f.read('scene_level_paired_results.csv')),
             'record_tar_commit_prediction_hashes_verified': True, 'N5000_checkpoint_identity_verified_across_sets': True,
             'scene_counts': {'development': 1000, 'fresh': 500, 'confirmation': 1000}, 'cross_set_scene_overlap': 0,
             'conditions': CONDITIONS, 'seeds': SEEDS,
             'aggregation': 'five fixed seeds equally averaged within each scene-condition; condition means within scene',
             'rng': 'NumPy default_rng(2026), 10 batches of 1000 replicates; independent contrasts draw A then B in each batch',
             'storage_reader': 'restricted torch ZIP metadata; only existing ADE storage materialized; no future coordinates loaded'}
    write('input_audit.json', (json.dumps(audit, indent=2) + '\n').encode())
    all_rows = [*scale_rows, condition, *cross_rows]
    lines = ['# RACC effect heterogeneity', '', '## Design and evidence status', '',
             'These are formal retrospective contrasts of already consumed frozen results. The designated primary contrast is N5000 minus N700 on the original fixed evaluation set. It was requested after the historical outcomes were known and is not a newly preregistered confirmatory test.', '',
             'The unit is scene. Five fixed seeds are equally averaged within each scene-condition. Seven conditions are equally weighted for overall effects; the corruption subgroup averages six conditions equally. Scale and condition contrasts resample the same scene jointly. Cross-set contrasts independently resample the two scene sets and subtract their means within each replicate. All analyses use 10,000 replicates, RNG seed2026 and nominal 95% percentile intervals. No condition-level CIs are averaged. No seeds are independently resampled.', '',
             '## Results', '', '| Contrast | Estimate (m) | 95% CI (m) | Interval support | Role |', '|---|---:|---|---|---|']
    lines.extend(f"| {r['contrast']} | {r['estimate_m']:+.6f} | [{r['ci95_lower_m']:+.6f}, {r['ci95_upper_m']:+.6f}] | {r['interval_supported']} | {r['role']} |" for r in all_rows)
    lines += ['', f"N5000 development effects: clean {condition['d_clean_m']:+.6f} m; six-corruption mean {condition['d_corruption_m']:+.6f} m.", '',
              '## Interpretation boundaries', '',
              'A positive interval-supported scale contrast supports a change in the Pairwise-versus-DU development ordering/effect between the nominated training scales on this fixed QCNet evaluation regime. It does not establish a universal scaling law, monotonicity or saturation.', '',
              'An interval-supported condition contrast supports an association with the evaluation regime at N5000. The same trained models are evaluated under clean and corrupted histories. It cannot identify a causal effect of training condition diversity.', '',
              'An interval-supported cross-set contrast supports a difference in the relative effect across these evaluation sets, conditional on the five frozen model pairs. It does not establish distribution shift, explain its mechanism, or prove generalization degradation. A zero-crossing interval means the difference is not interval-supported, even if one set individually has a positive interval and another does not.', '',
              'Secondary intervals are nominal and unadjusted for multiple comparisons. No subgroup discovery or model/threshold/condition selection was performed. Fresh500 remains consumed; its original NOT CONFIRMED result is unchanged. The same-N5000 confirmation comparison is distinct from the historical N10000 comparison.', '',
              '## Verification', '', 'All cached commit/tar/record hashes, prediction hashes and relevant fresh artifact hashes passed. Scene sets were disjoint. Same-N5000 Pairwise/DU checkpoint hashes matched across development, fresh500 and confirmation. Reconstructed development/confirmation per-seed ADE matched saved aggregate results within 1e-6 m. Float64 aggregation uses saved float32 ADE values; tiny rounding differences from historical summaries are expected.', '',
              'Only existing ADE storage and saved choices/scene effects were used. No model, trajectory, target or prediction was recomputed. Second-dataset expansion is closed. No new holdout was opened. STOP.', '']
    write('effect_heterogeneity_summary.md', '\n'.join(lines).encode())
    print(json.dumps(all_rows, indent=2))


if __name__ == '__main__':
    main()
