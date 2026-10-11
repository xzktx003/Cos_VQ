import json

import pytest

from experiments.nowag_d6.v19_e2e_selection import summarize, selected_learning_rate


def fixture(tmp_path, name, nll, completed=True):
    checkpoint = tmp_path / f'{name}_checkpoint.pt'
    checkpoint.write_bytes(b'fixture')
    progress = tmp_path / f'{name}_progress.json'
    progress.write_text(json.dumps({'completed': completed, 'best_validation_nll': nll,
        'selected_epoch': 0, 'arguments': {'output_dir': str(tmp_path), 'run_stamp': name,
        'learning_rate': 0.0001, 'train_sequences': 4, 'validation_sequences': 2,
        'context_length': 16, 'epochs': 3, 'batch_size': 4, 'seed': 0,
        'token_cache': 'same.pt'}, 'split_audit': {'unique_validation': 2}}))
    return progress


def test_selects_completed_development_nll_and_preserves_anchor_status(tmp_path):
    first = fixture(tmp_path, 'first', 2.)
    second = fixture(tmp_path, 'second', 1.)
    result = summarize([first, second], ['lr1', 'lr2'])
    assert result['selected']['label'] == 'lr2'
    assert result['completed'] is True
    assert result['uses_test_benchmarks_for_selection'] is False


def test_rejects_incomplete_training_and_mismatched_protocol(tmp_path):
    first = fixture(tmp_path, 'first', 2.)
    second = fixture(tmp_path, 'second', 1., completed=False)
    with pytest.raises(ValueError, match='completed'):
        summarize([first, second], ['lr1', 'lr2'])
    data = json.loads(second.read_text())
    data['completed'] = True
    data['arguments']['seed'] = 9
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='protocol'):
        summarize([first, second], ['lr1', 'lr2'])


def test_seed_replication_uses_selected_rate_for_same_source_only(tmp_path):
    report = summarize([fixture(tmp_path, 'first', 1.)], ['lr1'])
    report['protocol']['checkpoint'] = str(tmp_path / 'original.pt')
    path = tmp_path / 'selection.json'
    path.write_text(json.dumps(report))
    assert selected_learning_rate(path, tmp_path / 'original.pt') == .0001
    with pytest.raises(ValueError, match='source'):
        selected_learning_rate(path, tmp_path / 'unrelated.pt')
