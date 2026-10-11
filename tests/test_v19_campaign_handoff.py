import pytest
import copy
import json
from unittest.mock import patch

from experiments.nowag_d6.v19_campaign_handoff import _verify_process, _execute


def test_process_identity_rejects_reused_pid_or_wrong_command():
    expected = {'pid': 10, 'starttime': '100'}
    with pytest.raises(ValueError):
        _verify_process(expected, {'starttime': '101', 'argv': ['python'], 'state': 'S'}, ['python'])
    with pytest.raises(ValueError):
        _verify_process(expected, {'starttime': '100', 'argv': ['other'], 'state': 'S'}, ['python'])


def test_handoff_order_never_signals_training_child():
    events = []

    class MockOps:
        def stop(self, name): events.append(('stop', name))
        def verify_paused_job(self): events.append(('verify_paused', 'gpu0'))
        def start_lane(self, lane):
            events.append(('start', lane))
            return lane
        def wait_adopted(self, name): events.append(('wait_adopted', name))
        def record_child_outputs(self): events.append(('outputs', 'child'))
        def kill_paused(self, name): events.append(('kill', name))
        def wait_owned(self, lane): events.append(('wait_owned', lane))
        def verify_lane(self, lane):
            events.append(('verify_lane', lane))
            return True

    assert _execute(MockOps()) is True
    assert events.index(('wait_adopted', 'child')) < events.index(('kill', 'gpu0'))
    assert events.index(('outputs', 'child')) < events.index(('start', 'gpu0'))
    assert events.index(('start', 'gpu3')) < events.index(('wait_adopted', 'child'))
    for lane in ('gpu0', 'gpu3', 'gpu6'):
        assert events.index(('verify_lane', lane)) < events.index(('start', 'large32_phase2'))
    assert not any(event in ('stop', 'kill') and name in ('child', 'gpu6') for event, name in events)


def test_fail_closed_before_starting_gpu3_when_paused_state_changed():
    class MockOps:
        def stop(self, name): pass
        def verify_paused_job(self): raise ValueError('runner advanced')
        def start_lane(self, lane): pytest.fail('must not start a lane')
    with pytest.raises(ValueError, match='advanced'):
        _execute(MockOps())


@pytest.mark.parametrize('invalid', ['argv', 'started', 'protected', 'other_model'])
def test_manifest_only_allows_unstarted_llama3_lane_migration(invalid):
    from experiments.nowag_d6.v19_campaign_handoff import _manifest_change
    name = 'llama3_1_8b_instruct_train'
    old = {'protected_paths': ['source.pt'], 'jobs': [{'id': name, 'lane': 'gpu0',
            'env': {'CUDA_VISIBLE_DEVICES': '0'}, 'argv': ['python', 'train.py']}]}
    new = copy.deepcopy(old)
    new['jobs'][0].update(lane='gpu3', env={'CUDA_VISIBLE_DEVICES': '3'})
    states = {'gpu0': {'jobs': {}}}
    _manifest_change(old, new, states)
    if invalid == 'argv':
        new['jobs'][0]['argv'].append('--different')
    elif invalid == 'started':
        states['gpu0']['jobs'][name] = {'started_at': 1}
    elif invalid == 'protected':
        new['protected_paths'] = []
    else:
        old['jobs'][0]['id'] = new['jobs'][0]['id'] = 'llama2_train'
    with pytest.raises(ValueError):
        _manifest_change(old, new, states)


def test_kill_requires_paused_verified_scheduler_and_never_child():
    from experiments.nowag_d6.v19_campaign_handoff import _Operations
    ops = object.__new__(_Operations)
    ops.records = {'gpu0': {'pid': 10, 'starttime': '100', 'argv': ['expected']}}
    ops.fds = {'gpu0': 33}
    with patch('signal.pidfd_send_signal', create=True) as send:
        with pytest.raises(ValueError):
            ops.kill_paused('child')
        with patch('experiments.nowag_d6.v19_campaign_handoff._read_process',
                   return_value={'starttime': '100', 'argv': ['expected'], 'state': 'S'}):
            with pytest.raises(ValueError, match='not stopped'):
                ops.kill_paused('gpu0')
        send.assert_not_called()


def test_only_model_lane_metadata_change_allowed():
    from experiments.nowag_d6.v19_campaign_handoff import _manifest_change
    old = {'models': {'llama3_1_8b_instruct': {'lane': 'gpu0', 'model': 'unchanged'}},
           'jobs': [{'id': 'llama3_1_8b_instruct_train', 'lane': 'gpu0', 'env': {'CUDA_VISIBLE_DEVICES': '0'}}]}
    new = copy.deepcopy(old)
    new['jobs'][0].update(lane='gpu3', env={'CUDA_VISIBLE_DEVICES': '3'})
    new['models']['llama3_1_8b_instruct']['lane'] = 'gpu3'
    _manifest_change(old, new, {})
    new['models']['llama3_1_8b_instruct']['model'] = 'changed'
    with pytest.raises(ValueError, match='metadata'):
        _manifest_change(old, new, {})


def test_gpu3_spawn_failure_never_starts_gpu0_or_phase_two():
    events = []

    class MockOps:
        def stop(self, name): events.append(('stop', name))
        def verify_paused_job(self): pass
        def start_lane(self, lane):
            events.append(('start', lane))
            raise OSError('mock gpu3 spawn failure')
        def wait_adopted(self, name): pytest.fail('No follow-on operations after spawn failure')

    with pytest.raises(OSError):
        _execute(MockOps())
    assert events == [('stop', 'coordinator'), ('stop', 'gpu0'), ('start', 'gpu3')]


def test_paused_runner_actual_state_advance_is_rejected(tmp_path):
    from experiments.nowag_d6.v19_campaign_handoff import _Operations
    ops = object.__new__(_Operations)
    path = tmp_path/'lane_state.json'
    path.write_text(json.dumps({'jobs': {'next_job': {'status': 'running', 'pid': 11}}}))
    ops.state_paths = {'gpu0': path}
    ops.spec = {'current_job_id': 'expected_job'}
    ops.check = lambda *args, **kwargs: None
    with pytest.raises(ValueError, match='advanced'):
        ops.verify_paused_job()


def test_stop_allows_signal_delivery_before_one_state_verification(monkeypatch):
    from experiments.nowag_d6 import v19_campaign_handoff as module
    ops = object.__new__(module._Operations)
    ops.records = {'gpu0': {'pid': 10}}
    ops.fds = {'gpu0': 33}
    ops.state = {'adopted_processes': {'gpu0': {}}}
    events = []
    ops.event = lambda *args, **kwargs: None
    ops.save = lambda: None
    def check(name, paused=False):
        if paused:
            assert events == ['signal', 'delivery_wait']
        events.append('verified' if paused else 'identity')
    ops.check = check
    def send(*args):
        assert events == ['identity']
        events.clear()
        events.append('signal')
    monkeypatch.setattr(module.signal, 'pidfd_send_signal', send, raising=False)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: events.append('delivery_wait'))
    ops.stop('gpu0')
    assert events == ['signal', 'delivery_wait', 'verified']


def test_parallel32_starts_early_exactly_once():
    events = []
    class Ops:
        spec = {'parallel_large32': True}
        def stop(self, name): pass
        def verify_paused_job(self): pass
        def start_lane(self, name): events.append(('start', name))
        def wait_adopted(self, name): events.append(('adopt', name))
        def record_child_outputs(self): pass
        def kill_paused(self, name): pass
        def wait_owned(self, name): events.append(('wait', name))
        def verify_lane(self, name): return True
    assert _execute(Ops())
    assert events.count(('start', 'large32_phase2')) == 1
    assert events.index(('start', 'gpu3')) < events.index(('start', 'large32_phase2')) < events.index(('adopt', 'child'))


@pytest.mark.parametrize('bad', [None, 'disabled', 'argv', 'started', 'overlap'])
def test_parallel_manifest_strict_changes(bad):
    from experiments.nowag_d6.v19_campaign_handoff import _manifest_change, PARALLEL_LAUNCH_POLICY
    old = {'launch_policy': 'old', 'jobs': [
        {'id': 'llama3_1_8b_instruct_train', 'lane': 'gpu0', 'argv': [], 'env': {'CUDA_VISIBLE_DEVICES': '0'}},
        {'id': 'qwen3_32b_train', 'lane': 'large32_phase2', 'argv': [], 'env': {'CUDA_VISIBLE_DEVICES': '0'}},
        {'id': 'qwen3_32b_eval', 'lane': 'large32_phase2', 'argv': ['--split-gpus'], 'env': {'CUDA_VISIBLE_DEVICES': '0,6'}}]}
    new = copy.deepcopy(old)
    new['launch_policy'] = PARALLEL_LAUNCH_POLICY
    new['jobs'][0].update(lane='gpu3', env={'CUDA_VISIBLE_DEVICES': '3'})
    new['jobs'][1]['env']['CUDA_VISIBLE_DEVICES'] = '1'
    new['jobs'][2]['env']['CUDA_VISIBLE_DEVICES'] = '1,2'
    states = {}
    if bad == 'argv':
        new['jobs'][1]['argv'].append('--changed')
    if bad == 'started':
        states = {'large32_phase2': {'jobs': {'qwen3_32b_train': {'started_at': 1}}}}
    if bad == 'overlap':
        new['jobs'][0]['env']['CUDA_VISIBLE_DEVICES'] = '1'
    if bad is None:
        _manifest_change(old, new, states, parallel_large32=True)
    else:
        with pytest.raises(ValueError):
            _manifest_change(old, new, states, parallel_large32=bad != 'disabled')
