import json
import os
import sys

import pytest

from experiments.nowag_d6.v19_campaign_runner import run_campaign, resolve_selection


def manifest(tmp_path, jobs, protected=()):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'jobs': jobs, 'protected_paths': list(map(str, protected))}))
    return path


def job(tmp_path, name, code=None, depends=()):
    root = tmp_path / name
    output = root / 'result.txt'
    return {'id': name, 'argv': [sys.executable, '-c', code or
            "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(exist_ok=True); p.write_text('ok'); print('out'); print('err',file=sys.stderr)", str(output)],
            'cwd': str(tmp_path), 'env': {}, 'expected_outputs': [str(output)],
            'output_paths': [str(root)], 'depends_on': list(depends)}


def test_success_restart_missing_output_and_separate_logs(tmp_path):
    spec = job(tmp_path, 'a')
    path = manifest(tmp_path, [spec])
    state = tmp_path / 'state.json'
    first = run_campaign(path, state)
    assert first['jobs']['a']['status'] == 'succeeded'
    second = run_campaign(path, state)
    assert second['jobs']['a']['attempts'] == 1
    (tmp_path / 'a/result.txt').unlink()
    third = run_campaign(path, state)
    assert third['jobs']['a']['status'] == 'needs_fresh_outputs'
    assert third['jobs']['a']['attempts'] == 1
    assert 'out' in open(third['jobs']['a']['stdout_log']).read()
    assert 'err' in open(third['jobs']['a']['stderr_log']).read()


def test_failed_dependency_blocks_but_independent_runs(tmp_path):
    jobs = [job(tmp_path, 'dependent', depends=['failed']),
            job(tmp_path, 'failed', 'import sys; sys.exit(2)'), job(tmp_path, 'independent')]
    result = run_campaign(manifest(tmp_path, jobs), tmp_path / 'state.json')
    assert result['jobs']['failed']['status'] == 'failed'
    assert result['jobs']['dependent']['status'] == 'blocked'
    assert result['jobs']['independent']['status'] == 'succeeded'
    assert not (tmp_path / 'dependent').exists()


def test_existing_output_without_runner_ownership_rejected(tmp_path):
    spec = job(tmp_path, 'a')
    (tmp_path / 'a').mkdir()
    with pytest.raises(FileExistsError):
        run_campaign(manifest(tmp_path, [spec]), tmp_path / 'state.json')


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink', 'parent'])
def test_protected_checkpoint_cannot_be_output(tmp_path, alias):
    source = tmp_path / 'source.pt'
    source.write_text('original')
    spec = job(tmp_path, 'a')
    path = tmp_path / 'alias.pt'
    if alias == 'symlink':
        path.symlink_to(source)
    elif alias == 'hardlink':
        path.hardlink_to(source)
    elif alias == 'same':
        path = source
    else:
        path = tmp_path
    spec.update(output_paths=[str(path)], expected_outputs=[str(path)])
    with pytest.raises(ValueError):
        run_campaign(manifest(tmp_path, [spec], [source]), tmp_path / 'state.json')
    assert source.read_text() == 'original'


def test_selection_resolution_requires_completed_and_checkpoint(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_text('model')
    score = tmp_path / 'score.json'
    item = {'label': 'v19', 'checkpoint': str(checkpoint), 'nll': 1.}
    score.write_text(json.dumps({'completed': True, 'scores': [item], 'selected': item}))
    assert resolve_selection('${selection:' + str(score) + ':selected}') == str(checkpoint)
    assert resolve_selection('${selection:' + str(score) + ':v19}') == str(checkpoint)
    score.write_text(json.dumps({'completed': False, 'scores': [item], 'selected': item}))
    with pytest.raises(ValueError):
        resolve_selection('${selection:' + str(score) + ':selected}')


def test_cross_lane_dependencies_rejected_and_lane_selection(tmp_path):
    a, b = job(tmp_path, 'a'), job(tmp_path, 'b')
    a['lane'], b['lane'] = 'one', 'two'
    b['depends_on'] = ['a']
    with pytest.raises(ValueError):
        run_campaign(manifest(tmp_path, [a, b]), tmp_path / 'state.json', lane='one')
    b['depends_on'] = []
    result = run_campaign(manifest(tmp_path, [a, b]), tmp_path / 'state.json', lane='one')
    assert result['jobs']['a']['status'] == 'succeeded'
    assert not (tmp_path / 'b').exists()


def test_missing_expected_output_fails_even_with_zero_exit(tmp_path):
    result = run_campaign(manifest(tmp_path, [job(tmp_path, 'a', 'pass')]), tmp_path/'state.json')
    assert result['jobs']['a']['returncode'] == 0
    assert result['jobs']['a']['status'] == 'failed'
    assert result['jobs']['a']['missing_outputs']


def test_live_child_recovery_rejected_then_dead_handle_recovers(tmp_path):
    from experiments.nowag_d6.v19_campaign_runner import _identity
    path, state = manifest(tmp_path, [job(tmp_path, 'a')]), tmp_path/'state.json'
    saved = run_campaign(path, state)
    saved['jobs']['a'].update(status='running', pid=os.getpid(), process_identity=_identity(os.getpid()))
    state.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match='still running'):
        run_campaign(path, state)
    saved['jobs']['a'].update(pid=999999999, process_identity='unavailable')
    state.write_text(json.dumps(saved))
    result = run_campaign(path, state)
    assert result['jobs']['a']['status'] == 'needs_fresh_outputs'
    assert result['jobs']['a']['attempts'] == 1


def test_overlapping_job_outputs_and_changed_definition_rejected(tmp_path):
    first, second = job(tmp_path, 'a'), job(tmp_path, 'b')
    second.update(output_paths=first['output_paths'], expected_outputs=first['expected_outputs'])
    with pytest.raises(ValueError, match='overlapping'):
        run_campaign(manifest(tmp_path, [first, second]), tmp_path/'state.json')
    path = manifest(tmp_path, [first])
    run_campaign(path, tmp_path/'state.json')
    first['env']['DIFFERENT'] = '1'
    with pytest.raises(ValueError, match='definition changed'):
        run_campaign(manifest(tmp_path, [first]), tmp_path/'state.json')


def test_literal_argv_is_not_shell_interpreted(tmp_path):
    spec = job(tmp_path, 'a', "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(); p.write_text(sys.argv[2])")
    literal = 'echo hello; $(echo unsafe) & other'
    spec['argv'].append(literal)
    run_campaign(manifest(tmp_path, [spec]), tmp_path/'state.json')
    assert (tmp_path/'a/result.txt').read_text() == literal


def test_dangling_output_symlink_is_not_fresh(tmp_path):
    spec = job(tmp_path, 'a')
    (tmp_path/'a').symlink_to(tmp_path/'missing')
    with pytest.raises(FileExistsError):
        run_campaign(manifest(tmp_path, [spec]), tmp_path/'state.json')


@pytest.mark.parametrize('completed', [False, True, 1, 'true'])
def test_completed_json_requires_literal_true(tmp_path, completed):
    code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(); p.write_text(sys.argv[2])"
    spec = job(tmp_path, 'a', code)
    spec['argv'].append(json.dumps({'completed': completed}))
    spec['completed_json'] = spec['expected_outputs']
    path = manifest(tmp_path, [spec])
    result = run_campaign(path, tmp_path/'state.json')
    assert result['jobs']['a']['status'] == ('succeeded' if completed is True else 'failed')
    if completed is not True:
        content = (tmp_path/'a/result.txt').read_text()
        again = run_campaign(path, tmp_path/'state.json')
        assert again['jobs']['a']['status'] == 'needs_fresh_outputs'
        assert (tmp_path/'a/result.txt').read_text() == content


def test_completed_json_revalidated_on_restart(tmp_path):
    code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(); p.write_text('{\"completed\": true}')"
    spec = job(tmp_path, 'a', code)
    spec['completed_json'] = spec['expected_outputs']
    path = manifest(tmp_path, [spec])
    run_campaign(path, tmp_path/'state.json')
    (tmp_path/'a/result.txt').write_text('{"completed": false}')
    again = run_campaign(path, tmp_path/'state.json')
    assert again['jobs']['a']['status'] == 'needs_fresh_outputs'


@pytest.mark.parametrize('payload', ['{"completed": false}', 'not-json', '[]'])
def test_incomplete_json_blocks_dependencies(tmp_path, payload):
    code = "from pathlib import Path; import sys; p=Path(sys.argv[1]); p.parent.mkdir(); p.write_text(sys.argv[2])"
    producer = job(tmp_path, 'a', code)
    producer['argv'].append(payload)
    producer['completed_json'] = producer['expected_outputs']
    result = run_campaign(manifest(tmp_path, [producer, job(tmp_path, 'b', depends=['a'])]), tmp_path/'state.json')
    assert result['jobs']['a']['status'] == 'failed'
    assert result['jobs']['b']['status'] == 'blocked'
    assert not (tmp_path/'b').exists()


def test_topological_order_prefers_first_newly_ready_manifest_job(tmp_path):
    from experiments.nowag_d6.v19_campaign_runner import _jobs
    jobs = [job(tmp_path, 'a'), job(tmp_path, 'a_score', depends=['a']), job(tmp_path, 'b')]
    assert [j['id'] for j in _jobs({'jobs': jobs}, tmp_path)] == ['a', 'a_score', 'b']
