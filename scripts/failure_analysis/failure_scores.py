"""Authorized frozen reranker score replay; never run QCNet or fit anything."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from failure_inputs import B, O, SEEDS, sha


def compute(datasets):
    root = Path(__file__).resolve().parents[2] / 'inputs/external/racc_mf'
    sys.path[:0] = [str(root), str(root / 'experiments'), str(root.parent / 'QCNet')]
    import cache1000_stronger_baselines as helper
    import route_a_oof_support as support
    from racc_mf import v2
    from racc_mf.features import FEATURE_NAMES
    admission = json.loads((B / 'fresh500_admission_audit_v1/admission_audit.json').read_text())
    for info in admission['locked_sources'].values():
        relative = Path(info['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Use relative source identities in a local-only admission metadata copy')
        assert sha(root / relative) == info['expected']
    torch.set_num_threads(1)
    identity = []
    for label, data in datasets.items():
        n = len(data['ids'])
        rows = []
        for i in range(n):
            for c in range(7):
                raw = torch.from_numpy(data['features'][i, c])
                mode, scene = v2.extract_inputs(raw, FEATURE_NAMES)
                rows.append({'mode_features': mode, 'scene_features': scene,
                    'original_logits': torch.from_numpy(data['original_logits'][i, c]),
                    'ade': torch.from_numpy(data['ade_by_mode'][i, c]),
                    'fde': torch.from_numpy(data['fde_by_mode'][i, c]),
                    'scenario_id': data['ids'][i], 'condition': str(c), 'split': 'diagnostic'})
        pw_scores, du_scores = [], []
        for j, seed in enumerate(SEEDS):
            saves = {}
            for method in ('pairwise', 'direct_utility'):
                p = B / f'scaling_6000_v1_primary/{method}/N5000/seed{seed}/final_model.pt'
                expected = next(x['checkpoint_sha256'] for x in admission['checkpoints'] if x['method'] == method and x['seed'] == seed)
                assert sha(p) == expected
                saves[method] = torch.load(p, map_location='cpu', weights_only=False)
                identity.append({'dataset': label, 'method': method, 'seed': seed, 'checkpoint_sha256': expected})
            pair = saves['pairwise']
            model = helper.make_model('pairwise_ranknet', pair['v11_config'])
            model.load_state_dict(pair['state_dict'], strict=True)
            model.eval()
            batch = v2.tensor_batch(rows, pair['normalizer'], 'cpu')
            with torch.inference_mode():
                score = helper.forward(model, 'pairwise_ranknet', batch)['corrected_logits']
            choice = score.argmax(-1).numpy().reshape(n, 7)
            assert np.array_equal(choice, data['choices']['Pairwise'][j]), 'Pairwise replay differs'
            pw_scores.append(score.numpy().reshape(n, 7, 6))
            saved = saves['direct_utility']
            utility_model = support.DirectUtility()
            utility_model.load_state_dict(saved['state_dict'], strict=True)
            utility_model.eval()
            tensors = support.tensors(rows)
            raw, baseline, candidates = support.raw_utility(tensors['mode'], tensors['logits'])
            with torch.inference_mode():
                pred = utility_model((raw - saved['normalizer']['mean']) / saved['normalizer']['std'])
                choice = support.utility_select(pred, baseline, candidates)
            assert np.array_equal(choice.numpy().reshape(n, 7), data['choices']['DirectUtility'][j]), 'DU replay differs'
            full = np.zeros((n * 7, 6), dtype=np.float32)
            np.put_along_axis(full, candidates.numpy(), pred.numpy(), axis=-1)
            du_scores.append(full.reshape(n, 7, 6))
        data['scores'] = {'Pairwise': np.stack(pw_scores), 'DirectUtility': np.stack(du_scores)}
        data['feature_names'] = list(FEATURE_NAMES)
        pp = O / f'frozen_scores_{label}.npz'
        if pp.exists():
            with np.load(pp, allow_pickle=False) as existing:
                assert all(np.array_equal(existing[k], v) for k, v in data['scores'].items())
        else:
            temp = pp.with_name(pp.name + '.pending')
            with temp.open('wb') as f:
                np.savez_compressed(f, **data['scores'])
            temp.replace(pp)
        print(f'FROZEN_SCORE_REPLAY_PASS {label} all 5 seeds exact choices', flush=True)
    return identity
