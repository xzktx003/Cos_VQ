"""Fresh-directory-only two-phase coordinator for the frozen campaign snapshot.

No automatic recovery: inspect recorded child states before creating new jobs.
A manifest lock is inherited by runners, so a surviving runner prevents a second
coordinator from launching the same campaign after coordinator interruption.
"""

import argparse
from contextlib import ExitStack
import fcntl
from pathlib import Path
import subprocess
import sys
import time

from experiments.nowag_d6.v19_campaign_runner import _atomic


def coordinate(manifest_path, output_dir):
    manifest = Path(manifest_path).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Coordinator is fresh-dir-only; inspect old jobs before a new launch')
    lock_path = manifest.with_suffix(manifest.suffix + '.coordinator.lock')
    with lock_path.open('a') as lock, ExitStack() as files:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Campaign coordinator or inherited runner is still active') from error
        output.mkdir(parents=True, exist_ok=False)
        state_path = output / 'coordinator_state.json'
        state = {'manifest': str(manifest), 'started_at': time.time(), 'completed': False,
                 'finished': False, 'recovery': 'fresh-dir-only; never restart unresolved live jobs',
                 'phases': {}, 'lanes': {}}
        _atomic(state_path, state)
        children = {}
        waited = set()

        def start(lane):
            stdout_path, stderr_path = output/f'{lane}.stdout.log', output/f'{lane}.stderr.log'
            stdout = files.enter_context(stdout_path.open('x'))
            stderr = files.enter_context(stderr_path.open('x'))
            entry = {'started_at': time.time(), 'state_path': str(output/f'{lane}_state.json'),
                     'stdout_log': str(stdout_path), 'stderr_log': str(stderr_path), 'status': 'starting'}
            state['lanes'][lane] = entry
            _atomic(state_path, state)
            child = subprocess.Popen(
                [sys.executable, '-m', 'experiments.nowag_d6.v19_campaign_runner',
                 '--manifest', str(manifest), '--state', entry['state_path'], '--lane', lane],
                cwd=str(Path(__file__).resolve().parents[2]), shell=False,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, pass_fds=(lock.fileno(),))
            children[lane] = child
            entry.update(pid=child.pid, status='running')
            _atomic(state_path, state)

        def finish(lane):
            code = children[lane].wait()
            waited.add(lane)
            state['lanes'][lane].update(returncode=code, finished_at=time.time(), status='exited')
            _atomic(state_path, state)

        try:
            state['phases']['phase1'] = {'started_at': time.time()}
            _atomic(state_path, state)
            for lane in ('gpu0', 'gpu3', 'gpu6'):
                start(lane)
            for lane in ('gpu0', 'gpu3', 'gpu6'):
                finish(lane)
            state['phases']['phase1']['finished_at'] = time.time()
            state['phases']['phase2'] = {'started_at': time.time()}
            _atomic(state_path, state)
            start('large32_phase2')
            finish('large32_phase2')
            state['phases']['phase2']['finished_at'] = time.time()
            state['completed'] = all(entry['returncode'] == 0 for entry in state['lanes'].values())
        except BaseException as error:
            state['error'] = f'{type(error).__name__}: {error}'
            _atomic(state_path, state)
            # Never advance to phase2 after an interrupted/ambiguous launch.
            # Existing runners finish normally; don't kill their experiments.
            for lane in children:
                if lane not in waited:
                    finish(lane)
            for phase in state['phases'].values():
                if 'finished_at' not in phase:
                    phase.update(finished_at=time.time(), aborted=True)
            state.update(finished=True, finished_at=time.time())
            _atomic(state_path, state)
            raise
        state.update(finished=True, finished_at=time.time())
        _atomic(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    return int(not coordinate(args.manifest, args.output_dir)['completed'])


if __name__ == '__main__':
    raise SystemExit(main())
