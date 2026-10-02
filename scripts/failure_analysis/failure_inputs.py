"""Load only the two already-opened, frozen RACC evaluation datasets."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / 'inputs/racc_artifacts'
O = ROOT / 'results_local/failure_analysis'
CONDITIONS = ('clean', 'focal_keep30', 'focal_keep20', 'focal_keep10',
              'context_keep30', 'context_keep20', 'context_keep10')
SEEDS = (2026, 2027, 2028, 2029, 2030)


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load():
    selection = json.loads((B / 'scaling_6000_v1_selection/selection_manifest.json').read_text())
    fresh_selection = json.loads((B / 'fresh500_final_confirm_v1/selection_manifest.json').read_text())
    source_result = json.loads((B / 'scaling_6000_v1_results/primary_scaling_results.json').read_text())
    fresh_primary = json.loads((B / 'fresh500_final_confirm_v1/primary_metrics.json').read_text())
    datasets = {}
    audit = {'historical_files_modified': 0, 'new_holdouts_opened': 0,
             'primary_fresh_result_sha256': sha(B / 'fresh500_final_confirm_v1/primary_metrics.json'),
             'source_result_sha256': sha(B / 'scaling_6000_v1_results/primary_scaling_results.json'),
             'datasets': {}}
    settings = [
        ('dev', selection['ordered_6000_ids'][5000:6000], B / 'scaling_6000_v1_eval1000_cache', B / 'scaling_6000_v1_results', 50),
        ('fresh', fresh_selection['fresh500_ids'], B / 'fresh500_final_confirm_v1/cache', B / 'fresh500_final_confirm_v1/predictions', 25)]
    for label, ids, cache, predictions, chunks in settings:
        freeze_path = cache / 'eval1000_cache_freeze_manifest.json'
        freeze = json.loads(freeze_path.read_text())
        assert freeze['scenes'] == len(ids) and freeze['records'] == len(ids) * 7
        by_key = {}
        for chunk in range(chunks):
            cp = cache / f'chunk_{chunk:04d}.commit.json'
            commit = json.loads(cp.read_text())
            expected = next(x for x in freeze['chunks'] if x['chunk_index'] == chunk)
            assert sha(cp) == expected['commit_sha256']
            assert commit['scenario_ids'] == ids[chunk * 20:chunk * 20 + 20]
            tp = cache / commit['tar_file']
            assert sha(tp) == commit['tar_sha256'] == expected['tar_sha256']
            with tarfile.open(tp, 'r:*') as archive:
                members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
                for entry in commit['records']:
                    raw = archive.extractfile(members[entry['file']]).read()
                    assert hashlib.sha256(raw).hexdigest() == entry['sha256']
                    r = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
                    key = (str(r['scenario_id']), r['condition'])
                    assert key not in by_key and key[0] in commit['scenario_ids'] and key[1] == entry['condition']
                    by_key[key] = {k: r[k].float().numpy() for k in ('ade_by_mode', 'fde_by_mode', 'features', 'original_logits')}
        assert set(by_key) == {(sid, c) for sid in ids for c in CONDITIONS}
        ordered = [by_key[(sid, c)] for sid in ids for c in CONDITIONS]
        data = {'ids': ids, **{k: np.stack([r[k] for r in ordered]).reshape(len(ids), 7, *ordered[0][k].shape)
                               for k in ordered[0]}}
        choices = {k: [] for k in ('QCNet', 'Pairwise', 'DirectUtility')}
        prediction_audit = []
        for seed in SEEDS:
            cp = predictions / f'N5000/seed{seed}/commit.json'
            pp = cp.with_name('predictions.npz')
            commit = json.loads(cp.read_text())
            assert sha(pp) == commit['predictions_sha256']
            assert commit['eval_freeze_sha256'] == sha(freeze_path)
            assert commit['n'] == 5000 and commit['seed'] == seed
            with np.load(pp, allow_pickle=False) as saved:
                for k in choices:
                    assert saved[k].shape == (len(ids) * 7,) and np.all(saved[k] < 6)
                    choices[k].append(saved[k].reshape(len(ids), 7).astype(np.int64))
            prediction_audit.append({'seed': seed, 'commit_sha256': sha(cp), 'predictions_sha256': sha(pp),
                                     'pairwise_checkpoint_sha256': commit['pairwise_checkpoint_sha256'],
                                     'direct_utility_checkpoint_sha256': commit['direct_utility_checkpoint_sha256']})
        data['choices'] = {k: np.stack(v) for k, v in choices.items()}
        assert np.all(data['choices']['QCNet'] == data['original_logits'].argmax(-1))
        data['selected_ADE'] = {k: np.take_along_axis(np.broadcast_to(data['ade_by_mode'], (5, len(ids), 7, 6)), v[..., None], -1)[..., 0]
                                for k, v in data['choices'].items()}
        for m, values in data['selected_ADE'].items():
            if label == 'fresh':
                target = fresh_primary['overall_ADE'][{'DirectUtility': 'DU'}.get(m, m)]
            else:
                target = next(x['top1ADE'] for x in source_result['curves'] if x['n'] == 5000 and
                              x['method'] == {'Pairwise': 'Original Pairwise', 'DirectUtility': 'Direct Utility'}.get(m, m))
            assert abs(float(values.mean()) - target) < 1e-6
        datasets[label] = data
        audit['datasets'][label] = {'scenes': len(ids), 'cache_freeze_sha256': sha(freeze_path), 'predictions': prediction_audit}
        print(f'INPUT_AUDIT_PASS {label} scenes={len(ids)}', flush=True)
    for dev, fresh in zip(audit['datasets']['dev']['predictions'], audit['datasets']['fresh']['predictions']):
        assert dev['pairwise_checkpoint_sha256'] == fresh['pairwise_checkpoint_sha256']
        assert dev['direct_utility_checkpoint_sha256'] == fresh['direct_utility_checkpoint_sha256']
    return datasets, audit
