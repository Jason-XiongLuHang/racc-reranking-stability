#!/usr/bin/env python3
"""Exploratory failure analysis, using frozen predictions and authorized scores."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from failure_inputs import B, O, CONDITIONS, SEEDS, load, sha
from failure_scores import compute


def save(name, raw):
    p = O / name
    if p.exists():
        assert p.read_bytes() == raw, f'Existing diagnostic differs: {p}'
        return
    t = p.with_name(p.name + '.pending')
    with t.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(t, p)


def jsave(name, data):
    save(name, (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())


def csvsave(name, rows):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    save(name, f.getvalue().encode())


def corr(x, y, rank=False):
    if rank:
        x, y = ranks(x), ranks(y)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = ends - (counts - 1) / 2
    return average[inverse].astype(float)


def auc(x, positive):
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return None
    return float((ranks(x)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def dist(x):
    q = np.quantile(x, (.1, .5, .9, .95))
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()), P10=float(q[0]),
                P50=float(q[1]), P90=float(q[2]), P95=float(q[3]))


def switches(changed, true):
    good, bad = changed & (true > 1e-6), changed & (true < -1e-6)
    count = int(changed.sum())
    return {'records_with_seed_repeats': len(true), 'interventions': count,
            'intervention_rate': float(changed.mean()), 'beneficial_rate_all': float(good.mean()),
            'harmful_rate_all': float(bad.mean()),
            'beneficial_rate_among_switches': float(good.sum() / count) if count else 0.,
            'harmful_rate_among_switches': float(bad.sum() / count) if count else 0.,
            'beneficial_mean_gain': float(true[good].mean()) if good.any() else 0.,
            'harmful_mean_loss': float(-true[bad].mean()) if bad.any() else 0.,
            'beneficial_gain_mass_per_record': float(np.where(good, true, 0).mean()),
            'harmful_loss_mass_per_record': float(np.where(bad, -true, 0).mean()),
            'mean_true_utility_all': float(true.mean())}


def bins(x, y, count=10):
    assert len(x) == len(y) and len(x) > 0
    edges = np.unique(np.quantile(x, np.linspace(0, 1, count + 1)))
    if len(edges) < 2:
        edges = np.array([float(x[0]), float(x[0])])
    labels = np.searchsorted(edges[1:-1], x, side='right')
    for i in np.unique(labels):
        mask = labels == i
        yield {'bin': int(i + 1), 'bin_lower': float(edges[i]), 'bin_upper': float(edges[min(i + 1, len(edges) - 1)]),
               'count': int(mask.sum()), 'mean_predictor': float(x[mask].mean()),
               'mean_true_utility': float(y[mask].mean()), 'positive_gain_rate': float((y[mask] > 1e-6).mean()),
               'harmful_rate': float((y[mask] < -1e-6).mean())}


def main():
    assert auc(np.array([0., 1., 2., 3.]), np.array([False, False, True, True])) == 1.
    assert auc(np.ones(4), np.array([False, False, True, True])) == .5
    O.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    datasets, audit = load()
    replay = compute(datasets)
    audit.update({'score_replay_authorized_by_user': True, 'QCNet_inference_runs': 0,
                  'training_runs': 0, 'tuning_runs': 0, 'original_choices_exactly_reproduced': True,
                  'frozen_checkpoints': replay,
                  'scripts': {name: sha(Path(__file__).with_name(name)) for name in
                              ('analyze_fresh_failure.py', 'failure_inputs.py', 'failure_scores.py')}})
    switch_rows, calibration_rows, margin_rows, shift_rows, proxy_rows = [], [], [], [], []
    attribution = {}
    calibration_summary, margin_summary = {}, {}
    proxy_top = {}
    for label, data in datasets.items():
        n = len(data['ids'])
        a = data['selected_ADE']
        baseline = a['QCNet']
        groups = [('all', np.arange(7)), ('clean', np.array([0])), ('corruption', np.arange(1, 7))]
        for ci, c in enumerate(CONDITIONS):
            groups.append((c, np.array([ci]))) if c != 'clean' else None
        data['true'] = {m: baseline - a[m] for m in ('Pairwise', 'DirectUtility')}
        data['changed'] = {m: data['choices'][m] != data['choices']['QCNet'] for m in data['true']}
        data['margin'] = {}
        for method, s in data['scores'].items():
            sorted_score = np.sort(s, axis=-1)
            data['margin'][method] = sorted_score[..., -1] - sorted_score[..., -2]
        du_scores = data['scores']['DirectUtility']
        data['predicted_selected_utility'] = np.take_along_axis(du_scores, data['choices']['DirectUtility'][..., None], -1)[..., 0]
        scores_no_baseline = du_scores.copy()
        np.put_along_axis(scores_no_baseline, data['choices']['QCNet'][..., None], -np.inf, axis=-1)
        ranked = np.sort(scores_no_baseline, axis=-1)
        data['nonbaseline_margin'] = ranked[..., -1] - ranked[..., -2]
        ade_sorted = np.sort(data['ade_by_mode'], axis=-1)
        data['true_candidate_gap'] = ade_sorted[..., 1] - ade_sorted[..., 0]
        data['oracle_opportunity'] = baseline[0] - ade_sorted[..., 0]
        data['utility_ensemble_std'] = np.take_along_axis(np.broadcast_to(du_scores.std(0), du_scores.shape), data['choices']['DirectUtility'][..., None], -1)[..., 0]
        agreement = np.stack([(data['choices']['DirectUtility'] == k).mean(0) for k in range(6)]).max(0)
        data['mode_agreement'] = np.broadcast_to(agreement, (5, n, 7))
        for group, idx in groups:
            for method in ('Pairwise', 'DirectUtility'):
                true = data['true'][method][:, :, idx].ravel()
                changed = data['changed'][method][:, :, idx].ravel()
                row = {'dataset': label, 'group': group, 'method': method, **switches(changed, true)}
                assert abs(row['mean_true_utility_all'] - (row['beneficial_gain_mass_per_record'] - row['harmful_loss_mass_per_record'])) < 1e-5
                switch_rows.append(row)
            pw, du = switch_rows[-2:]
            harm = du['harmful_loss_mass_per_record'] - pw['harmful_loss_mass_per_record']
            benefit = pw['beneficial_gain_mass_per_record'] - du['beneficial_gain_mass_per_record']
            du_minus_pw = float((a['DirectUtility'][:, :, idx] - a['Pairwise'][:, :, idx]).mean())
            assert abs(harm + benefit - du_minus_pw) < 1e-5
            attribution[label + '/' + group] = {'DU_minus_Pairwise_ADE': du_minus_pw,
                'excess_DU_harmful_loss_mass': harm, 'missing_DU_beneficial_gain_mass': benefit}
            if group not in ('all', 'clean', 'corruption'):
                continue
            changed = data['changed']['DirectUtility'][:, :, idx].ravel()
            true = data['true']['DirectUtility'][:, :, idx].ravel()
            pred = data['predicted_selected_utility'][:, :, idx].ravel()
            for r in bins(pred[changed], true[changed]):
                calibration_rows.append({'dataset': label, 'group': group, 'population': 'DU switches only',
                    **r, 'mean_predicted_minus_true': r['mean_predictor'] - r['mean_true_utility']})
            top_cut = np.quantile(pred[changed], .9)
            top = changed & (pred >= top_cut)
            calibration_summary[label + '/' + group] = {
                'mean_predicted_utility_switches': float(pred[changed].mean()),
                'mean_true_utility_switches': float(true[changed].mean()),
                'prediction_bias_switches': float((pred[changed] - true[changed]).mean()),
                'prediction_true_pearson_switches': corr(pred[changed], true[changed]),
                'prediction_true_spearman_switches': corr(pred[changed], true[changed], True),
                'top_predicted_decile_mean_true_gain': float(true[top].mean()),
                'top_predicted_decile_harmful_rate': float((true[top] < -1e-6).mean()),
                'overall_switch_harmful_rate': float((true[changed] < -1e-6).mean()),
                'top_predicted_decile_positive_gain_rate': float((true[top] > 1e-6).mean())}
            for method in ('Pairwise', 'DirectUtility'):
                change = data['changed'][method][:, :, idx].ravel()
                utility = data['true'][method][:, :, idx].ravel()
                margin = data['margin'][method][:, :, idx].ravel()
                for r in bins(margin[change], utility[change], 5):
                    margin_rows.append({'dataset': label, 'group': group, 'method': method,
                        'margin_definition': 'top1 minus runner-up among 6 decisions; DU includes baseline utility 0', **r})
                low_cut = np.quantile(margin[change], .2)
                low = change & (margin <= low_cut)
                bad = change & (utility < -1e-6)
                bh = change & (np.abs(utility) > 1e-6)
                margin_summary[label + '/' + group + '/' + method] = {
                    'margin_true_gain_spearman': corr(margin[change], utility[change], True),
                    'higher_margin_harmful_AUC': auc(margin[bh], utility[bh] < -1e-6),
                    'lowest_margin_quintile_share_of_switches': float(low.sum() / change.sum()),
                    'lowest_margin_quintile_share_of_harmful_switches': float((low & bad).sum() / bad.sum()) if bad.any() else 0.,
                    'lowest_margin_quintile_harmful_rate': float(bad[low].mean()),
                    'overall_harmful_rate_among_switches': float(bad[change].mean())}
            measurements = {
                'baseline_top1_ADE': baseline[0][:, idx],
                'oracle_ADE': ade_sorted[:, idx, 0],
                'oracle_opportunity_baseline_minus_best': data['oracle_opportunity'][:, idx],
                'true_candidate_best_second_ADE_gap': data['true_candidate_gap'][:, idx],
                'DU_true_utility_all': data['true']['DirectUtility'][:, :, idx],
                'DU_true_utility_switches': true[changed],
                'DU_predicted_utility_all': pred,
                'DU_predicted_utility_switches': pred[changed],
                'DU_margin_switches': data['margin']['DirectUtility'][:, :, idx].ravel()[changed],
                'Pairwise_true_utility_all': data['true']['Pairwise'][:, :, idx]}
            for k, values in measurements.items():
                shift_rows.append({'dataset': label, 'group': group, 'variable': k, **dist(values.ravel())})
            # No classifier: only empirical rank AUC of frozen, existing features.
            feat = np.broadcast_to(data['features'], (5, n, 7, 6, 15))
            selected = np.take_along_axis(feat, data['choices']['DirectUtility'][..., None, None], axis=3)[..., 0, :]
            basefeat = np.take_along_axis(feat, data['choices']['QCNet'][..., None, None], axis=3)[..., 0, :]
            variables = {}
            for j, name in enumerate(data['feature_names']):
                for prefix, features in [('selected', selected), ('baseline', basefeat), ('selected_minus_baseline', selected - basefeat)]:
                    variables[prefix + '/' + name] = (features[:, :, idx, j].ravel(), True)
            for name, v in [('DU_predicted_selected_utility', data['predicted_selected_utility']),
                            ('DU_decision_margin', data['margin']['DirectUtility']),
                            ('DU_nonbaseline_candidate_margin', data['nonbaseline_margin']),
                            ('DU_utility_ensemble_std', data['utility_ensemble_std']),
                            ('DU_mode_agreement_across_frozen_seeds', data['mode_agreement'])]:
                variables[name] = (v[:, :, idx].ravel(), True)
            variables['true_candidate_ADE_gap_DIAGNOSTIC_ONLY'] = (np.broadcast_to(data['true_candidate_gap'], (5, n, 7))[:, :, idx].ravel(), False)
            variables['baseline_ADE_DIAGNOSTIC_ONLY'] = (baseline[:, :, idx].ravel(), False)
            bh = changed & (np.abs(true) > 1e-6)
            for name, (x, available) in variables.items():
                score = auc(x[bh], true[bh] < -1e-6)
                proxy_rows.append({'dataset': label, 'group': group, 'variable': name,
                    'inference_available': available, 'switch_count_beneficial_or_harmful': int(bh.sum()),
                    'AUC_harmful_higher_value': score, 'descriptive_separation_strength': max(score, 1 - score) if score is not None else None,
                    'harmful_mean': float(x[changed & (true < -1e-6)].mean()),
                    'beneficial_mean': float(x[changed & (true > 1e-6)].mean())})
        proxy_top[label] = sorted([r for r in proxy_rows if r['dataset'] == label and r['group'] == 'all' and
            r['inference_available'] and r['descriptive_separation_strength'] is not None],
            key=lambda r: r['descriptive_separation_strength'], reverse=True)[:8]
    fresh = datasets['fresh']
    n = len(fresh['ids'])
    loss = fresh['selected_ADE']['DirectUtility'] - fresh['selected_ADE']['Pairwise']
    scene_loss = loss.mean(axis=(0, 2))
    order = np.argsort(-scene_loss, kind='stable')
    worst = np.zeros(n, dtype=bool)
    worst[order[:n // 10]] = True
    worst_rows = []
    for sid_index in order[:n // 10]:
        for ci, condition in enumerate(CONDITIONS):
            change = fresh['changed']['DirectUtility'][:, sid_index, ci]
            truth = fresh['true']['DirectUtility'][:, sid_index, ci]
            worst_rows.append({'scenario_id': fresh['ids'][sid_index], 'condition': condition,
                'scene_DU_minus_Pairwise_ADE': float(scene_loss[sid_index]),
                'condition_DU_minus_Pairwise_ADE': float(loss[:, sid_index, ci].mean()),
                'DU_switch_rate_across_seeds': float(change.mean()),
                'DU_true_utility': float(truth.mean()),
                'DU_predicted_selected_utility': float(fresh['predicted_selected_utility'][:, sid_index, ci].mean()),
                'DU_decision_margin': float(fresh['margin']['DirectUtility'][:, sid_index, ci].mean()),
                'Pairwise_margin': float(fresh['margin']['Pairwise'][:, sid_index, ci].mean()),
                'DU_harmful_rate_across_seeds': float((change & (truth < -1e-6)).mean()),
                'true_candidate_ADE_gap': float(fresh['true_candidate_gap'][sid_index, ci]),
                'baseline_top1_ADE': float(fresh['selected_ADE']['QCNet'][0, sid_index, ci]),
                'oracle_opportunity': float(fresh['oracle_opportunity'][sid_index, ci])})
    worst_summary = {}
    pred_cut = np.quantile(fresh['predicted_selected_utility'][fresh['changed']['DirectUtility']], .9)
    margin_cut = np.quantile(fresh['margin']['DirectUtility'][fresh['changed']['DirectUtility']], .2)
    for label, scene_mask in [('worst10pct', worst), ('remaining90pct', ~worst)]:
        change = fresh['changed']['DirectUtility'][:, scene_mask, :]
        truth = fresh['true']['DirectUtility'][:, scene_mask, :]
        pred = fresh['predicted_selected_utility'][:, scene_mask, :]
        margin = fresh['margin']['DirectUtility'][:, scene_mask, :]
        bad = change & (truth < -1e-6)
        worst_summary[label] = {'scenes': int(scene_mask.sum()), 'mean_DU_minus_Pairwise_ADE': float(scene_loss[scene_mask].mean()),
            'DU_harmful_rate_among_switches': float(bad.sum() / change.sum()),
            'high_predicted_utility_but_harmful_rate_among_switches': float((bad & (pred >= pred_cut)).sum() / change.sum()),
            'mean_DU_margin_switches': float(margin[change].mean()),
            'low_margin_share_of_harmful_switches': float((bad & (margin <= margin_cut)).sum() / bad.sum()),
            'mean_true_candidate_ADE_gap': float(fresh['true_candidate_gap'][scene_mask].mean()),
            'condition_mean_DU_minus_Pairwise_ADE': {c: float(loss[:, scene_mask, i].mean()) for i, c in enumerate(CONDITIONS)}}
    transition = {k: attribution['fresh/all'][k] - attribution['dev/all'][k]
                  for k in ('DU_minus_Pairwise_ADE', 'excess_DU_harmful_loss_mass', 'missing_DU_beneficial_gain_mass')}
    shift_comparisons = []
    indexed_shift = {(r['dataset'], r['group'], r['variable']): r for r in shift_rows}
    for key, dev in indexed_shift.items():
        if key[0] != 'dev':
            continue
        fresh_row = indexed_shift[('fresh', key[1], key[2])]
        delta = fresh_row['mean'] - dev['mean']
        pooled_sd = np.sqrt((dev['std'] ** 2 + fresh_row['std'] ** 2) / 2)
        shift_comparisons.append({'group': key[1], 'variable': key[2], 'dev_mean': dev['mean'],
            'fresh_mean': fresh_row['mean'], 'mean_change': delta,
            'standardized_mean_change_descriptive': float(delta / pooled_sd) if pooled_sd else 0.})
    shift_comparisons.sort(key=lambda r: abs(r['standardized_mean_change_descriptive']), reverse=True)
    summary = {'status': 'FAILURE_ANALYSIS_COMPLETE', 'scope': 'exploratory; no new holdout or fitted classifier',
        'switch_attribution': attribution, 'dev_to_fresh_reversal_decomposition': transition,
        'calibration_summary': calibration_summary, 'margin_summary': margin_summary,
        'dev_vs_fresh_shift_comparisons': shift_comparisons,
        'worst_scene_summary': worst_summary, 'strongest_inference_available_proxies': proxy_top,
        'composition': 'both datasets have identical 1/7 clean and 6/7 corruption weights; composition change cannot explain the difference',
        'limitations': ['seed-scene-condition rows are repeated measurements, not independent samples',
            'AUCs are descriptive and unadjusted; proxy rankings are exploratory',
            'true utility, candidate ADE-gap and baseline ADE require future ground truth and cannot be inference inputs',
            'fresh500 is now consumed and can guide hypotheses but cannot independently validate RACC-U',
            'no operating threshold, new normalizer, classifier or reranker has been fit'],
        'audit': audit}
    for name, rows in [('switch_decomposition.csv', switch_rows), ('utility_calibration_bins.csv', calibration_rows),
        ('margin_analysis.csv', margin_rows), ('dev_vs_fresh_shift.csv', shift_rows),
        ('worst_scene_analysis.csv', worst_rows), ('uncertainty_proxy_auc.csv', proxy_rows)]:
        csvsave(name, rows)
    jsave('failure_analysis_results.json', summary)
    jsave('source_audit.json', audit)
    fa = attribution['fresh/all']
    ca = calibration_summary['fresh/all']
    ma = margin_summary['fresh/all/DirectUtility']
    lines = ['# RACC fresh500 failure analysis', '', 'Frozen-model diagnostic analysis; no QCNet inference, training or tuning.',
        '', '## Findings', '',
        f"Fresh DU−Pairwise ADE = {fa['DU_minus_Pairwise_ADE']:+.6f} m = excess DU harmful loss mass {fa['excess_DU_harmful_loss_mass']:+.6f} + missing DU beneficial gain mass {fa['missing_DU_beneficial_gain_mass']:+.6f}.",
        f"DU switch utility: predicted mean {ca['mean_predicted_utility_switches']:.6f} m versus true mean {ca['mean_true_utility_switches']:.6f} m. Highest predicted decile harmful rate {ca['top_predicted_decile_harmful_rate']:.3%}.",
        f"DU margin: harmful-label AUC for increasing margin {ma['higher_margin_harmful_AUC']}; lowest margin quintile contains {ma['lowest_margin_quintile_share_of_harmful_switches']:.3%} of harmful switches.",
        'Existing inference-available proxy rankings below are descriptive design evidence, not validation of a future conservative policy. Oracle/true ADE quantities cannot be deployment inputs.',
        'These diagnostics can motivate RACC-U hypotheses; a new method still needs independent evaluation outside the consumed fresh500.',
        '', '## Harmful versus beneficial switch decomposition', '', json.dumps(attribution, indent=2),
        '', '## Development-to-fresh reversal', '', json.dumps(transition, indent=2),
        '', '## Utility calibration', '', json.dumps(calibration_summary, indent=2),
        '', '## Margin as a proxy', '', json.dumps(margin_summary, indent=2),
        '', '## Worst 10% of fresh scenes', '', json.dumps(worst_summary, indent=2),
        '', '## Existing inference-available proxy variables', '', json.dumps(proxy_top, indent=2),
        '', '## Development versus fresh distribution summaries', '', json.dumps(shift_comparisons, indent=2),
        '', summary['composition'], '', *summary['limitations'], '',
        'The earlier one-shot fresh500 conclusion remains NOT CONFIRMED. This exploratory analysis cannot revise it.', '']
    save('failure_analysis_summary.md', '\n'.join(lines).encode())
    jsave('artifact_hashes.json', {p.name: sha(p) for p in sorted(O.iterdir()) if p.is_file() and
        p.name not in ('artifact_hashes.json', 'execution.log') and not p.name.endswith('.pending')})
    print('FAILURE_ANALYSIS_COMPLETE', flush=True)
    print('ATTRIBUTION', attribution['dev/all'], attribution['fresh/all'], flush=True)
    print('REVERSAL', transition, flush=True)
    print('CALIBRATION', calibration_summary['dev/all'], calibration_summary['fresh/all'], flush=True)
    print('MARGIN', margin_summary['fresh/all/DirectUtility'], flush=True)
    print('TOP_PROXIES', [(r['variable'], r['AUC_harmful_higher_value']) for r in proxy_top['fresh']], flush=True)


if __name__ == '__main__':
    main()
