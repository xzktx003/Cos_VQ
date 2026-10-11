import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.nowag_d6.v19_campaign_coordinator import coordinate


def test_phase_two_waits_for_every_phase_one_lane_even_after_failure(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    events = []

    class Child:
        def __init__(self, argv, **kwargs):
            self.lane = argv[-1]
            self.pid = len(events) + 100
            assert kwargs['shell'] is False
            if self.lane == 'large32_phase2':
                assert {lane for event, lane in events if event == 'wait'} == {'gpu0', 'gpu3', 'gpu6'}
            events.append(('start', self.lane))

        def wait(self):
            events.append(('wait', self.lane))
            return 1 if self.lane == 'gpu3' else 0

    with patch('subprocess.Popen', Child):
        result = coordinate(manifest, tmp_path / 'run')
    assert result['finished'] is True and result['completed'] is False
    assert events[:3] == [('start', 'gpu0'), ('start', 'gpu3'), ('start', 'gpu6')]
    assert result['lanes']['gpu3']['returncode'] == 1
    assert result['lanes']['large32_phase2']['returncode'] == 0
    assert json.loads((tmp_path/'run/coordinator_state.json').read_text()) == result


def test_success_and_no_output_directory_reuse(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')

    class Child:
        pid = 100

        def __init__(self, argv, **kwargs):
            assert argv[:3] == [__import__('sys').executable, '-m', 'experiments.nowag_d6.v19_campaign_runner']
            assert Path(argv[argv.index('--manifest')+1]) == manifest

        def wait(self):
            return 0

    with patch('subprocess.Popen', Child) as spawn:
        assert coordinate(manifest, tmp_path/'run')['completed'] is True
        with pytest.raises(FileExistsError):
            coordinate(manifest, tmp_path/'run')
    assert spawn is Child


def test_spawn_failure_waits_existing_child_without_starting_phase_two(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    events = []

    class Child:
        pid = 100

        def __init__(self, argv, **kwargs):
            lane = argv[-1]
            assert kwargs['pass_fds']
            events.append(('start', lane))
            if lane == 'gpu3':
                raise OSError('mock spawn failure')

        def wait(self):
            events.append(('wait', 'gpu0'))
            return 0

    with patch('subprocess.Popen', Child), pytest.raises(OSError):
        coordinate(manifest, tmp_path/'run')
    assert events == [('start', 'gpu0'), ('start', 'gpu3'), ('wait', 'gpu0')]
    result = json.loads((tmp_path/'run/coordinator_state.json').read_text())
    assert result['finished'] and not result['completed']
    assert result['phases']['phase1']['aborted']


def test_concurrent_fresh_directory_cannot_duplicate_same_manifest(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')

    class Child:
        pid = 100

        def __init__(self, argv, **kwargs):
            if argv[-1] == 'gpu0':
                with pytest.raises(RuntimeError, match='still active'):
                    coordinate(manifest, tmp_path/'duplicate')

        def wait(self):
            return 0

    with patch('subprocess.Popen', Child):
        coordinate(manifest, tmp_path/'run')
    assert not (tmp_path/'duplicate').exists()
