"""Synthetic software checks only: no dataset, training, inference or holdout."""
import json
from pathlib import Path
import numpy as np
from analyze_effect_heterogeneity import paired_bootstrap, independent_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def main():
    difficulty = np.array([1., 4., 8.])
    assert np.allclose(paired_bootstrap((difficulty + .2) - difficulty), .2)
    assert np.allclose(independent_bootstrap(np.full(7, .3), np.full(5, .1)), .2)
    assert np.array_equal(paired_bootstrap(np.arange(7.)), paired_bootstrap(np.arange(7.)))
    training = json.loads((ROOT / 'configs/frozen_training.json').read_text())
    evaluation = json.loads((ROOT / 'configs/frozen_evaluation.json').read_text())
    assert training['seeds'] == list(range(2026, 2031))
    assert training['du']['actual_seeds'] == [5*s for s in training['seeds']]
    assert len(training['conditions']) == 7
    assert training['steps']['5000'] == {'pairwise':2857,'du':3571,'c1_per_head':11429}
    assert evaluation['bootstrap']['unit'] == 'scene'
    assert evaluation['bootstrap']['condition_CIs_averaged'] is False
    assert evaluation['bootstrap']['replicates'] == 10000
    assert evaluation['final_success_rule'] == {'mean_min_m':.01,'positive_seeds_min':4,'every_LOSO_mean_positive':True,'CI_lower_positive':True}
    print('SYNTHETIC_SELF_CHECK_PASS: paired/independent bootstrap, reproducible RNG and frozen config invariants')


if __name__ == '__main__':
    main()
