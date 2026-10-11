"""Explicit, fail-closed scheduler handoff; never signals experiment children.

Linux only. Spec process records additionally require the exact expected argv.
This module does not resume failed handoffs: stopped schedulers and any launched
runner remain recorded for manual inspection. No training process is killed.
"""

import argparse
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time

from experiments.nowag_d6.v19_campaign_runner import _atomic, _jobs, _output_errors, resolve_selection

PARALLEL_LAUNCH_POLICY = 'parallel-on-dedicated-gpus: large32_phase2=1,2; gpu0/gpu3/gpu6=0,3,6'


def _verify_process(expected, actual, argv):
    if str(expected['starttime']) != actual['starttime'] or actual['argv'] != argv:
        raise ValueError(f'Process identity or command mismatch: {expected["pid"]}')
    if actual['state'] in ('Z', 'X'):
        raise ValueError('Process already exited')


def _read_process(pid):
    root = Path(f'/proc/{pid}')
    fields = (root/'stat').read_text().rsplit(')', 1)[1].split()
    return {'starttime': fields[19], 'state': fields[0], 'ppid': int(fields[1]),
            'argv': (root/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
            'cwd': str((root/'cwd').resolve(strict=True))}


def _execute(ops):
    parallel = getattr(ops, 'spec', {}).get('parallel_large32', False)
    ops.stop('coordinator')
    ops.stop('gpu0')
    ops.verify_paused_job()
    ops.start_lane('gpu3')
    if parallel:
        ops.start_lane('large32_phase2')
    ops.wait_adopted('child')
    ops.record_child_outputs()
    ops.kill_paused('gpu0')
    ops.start_lane('gpu0')
    ops.wait_owned('gpu3')
    ops.wait_owned('gpu0')
    ops.wait_adopted('gpu6')
    phase1 = [ops.verify_lane(lane) for lane in ('gpu0', 'gpu3', 'gpu6')]
    ops.kill_paused('coordinator')
    if not parallel:
        ops.start_lane('large32_phase2')
    ops.wait_owned('large32_phase2')
    phase2 = ops.verify_lane('large32_phase2')
    return all(phase1) and phase2


def _manifest_change(old, new, states, *, parallel_large32=False):
    if not isinstance(parallel_large32, bool):
        raise ValueError('parallel_large32 must be boolean')
    old_metadata = {key: value for key, value in old.items() if key != 'jobs'}
    new_metadata = {key: value for key, value in new.items() if key != 'jobs'}
    allowed_metadata = copy.deepcopy(old_metadata)
    model = allowed_metadata.get('models', {}).get('llama3_1_8b_instruct')
    if model is not None:
        model['lane'] = 'gpu3'
    if parallel_large32:
        allowed_metadata['launch_policy'] = PARALLEL_LAUNCH_POLICY
        if new_metadata.get('launch_policy') != PARALLEL_LAUNCH_POLICY:
            raise ValueError('Parallel policy must explicitly document dedicated GPUs')
    if new_metadata not in (old_metadata, allowed_metadata):
        raise ValueError('Only Llama3 model lane metadata may change')
    old_jobs, new_jobs = {j['id']: j for j in old['jobs']}, {j['id']: j for j in new['jobs']}
    if old_jobs.keys() != new_jobs.keys() or old.get('protected_paths') != new.get('protected_paths'):
        raise ValueError('Handoff must preserve job IDs and protected sources')
    changed = []
    large_count = 0
    for name, before in old_jobs.items():
        after = new_jobs[name]
        if parallel_large32 and before['lane'] == 'large32_phase2':
            large_count += 1
            expected = copy.deepcopy(before)
            expected['env']['CUDA_VISIBLE_DEVICES'] = '1,2' if '--split-gpus' in before['argv'] else '1'
            if after != expected:
                raise ValueError('Parallel32 permits only dedicated GPU env changes')
            if any(state.get('jobs', {}).get(name, {}).get('started_at') for state in states.values()):
                raise ValueError('Parallel32 job already started')
            continue
        if parallel_large32:
            gpu = after.get('env', {}).get('CUDA_VISIBLE_DEVICES', '')
            if not gpu or not set(gpu.split(',')) <= {'0', '3', '6'}:
                raise ValueError('Other lanes overlap dedicated GPUs or have unknown GPU bindings')
        if before == after:
            continue
        expected = copy.deepcopy(before)
        expected['lane'] = 'gpu3'
        expected['env']['CUDA_VISIBLE_DEVICES'] = '3'
        if not name.startswith('llama3_1_8b_instruct_') or before['lane'] != 'gpu0' or after != expected:
            raise ValueError(f'Only untouched Llama3 lane/env may change: {name}')
        if any(state.get('jobs', {}).get(name, {}).get('started_at') for state in states.values()):
            raise ValueError(f'Cannot migrate a started job: {name}')
        changed.append(name)
    if not changed:
        raise ValueError('No Llama3 jobs migrated')
    if parallel_large32 and not large_count:
        raise ValueError('Parallel32 requested without any large32 jobs')


class _Operations:
    def __init__(self, spec, output, resources, lock_fd):
        self.spec, self.output, self.resources, self.lock_fd = spec, output, resources, lock_fd
        self.manifest = Path(spec['new_manifest']).resolve(strict=True)
        old_path = Path(spec['expected_old_manifest']).resolve(strict=True)
        self.snapshot = Path(spec['frozen_source_snapshot']).resolve(strict=True)
        self.state_paths = {lane: Path(path).resolve(strict=True) for lane, path in spec['old_lane_state_paths'].items()}
        if set(self.state_paths) != {'gpu0', 'gpu3', 'gpu6'}:
            raise ValueError('Exactly three original lane states required')
        old, new = json.loads(old_path.read_text()), json.loads(self.manifest.read_text())
        states = {lane: json.loads(path.read_text()) for lane, path in self.state_paths.items()}
        if spec.get('parallel_large32', False):
            coordinator_argv = spec['old_coordinator']['argv']
            old_output = Path(coordinator_argv[coordinator_argv.index('--output-dir') + 1]).resolve()
            old_large_state = old_output/'large32_phase2_state.json'
            if old_large_state.exists():
                states['large32_phase2'] = json.loads(old_large_state.read_text())
        _manifest_change(old, new, states, parallel_large32=spec.get('parallel_large32', False))
        self.old_jobs = {j['id']: j for j in _jobs(old, old_path.parent)}
        self.new_jobs = {j['id']: j for j in _jobs(new, self.manifest.parent)}
        if spec.get('parallel_large32', False):
            for job in self.new_jobs.values():
                if job['lane'] == 'large32_phase2' and any(path.exists() or path.is_symlink() for path in job['output_paths']):
                    raise ValueError('Parallel32 requires untouched output paths')
        if any(Path(j['cwd']) != self.snapshot for j in self.new_jobs.values()):
            raise ValueError('Every job must use the frozen snapshot')
        self.records = {name: spec[key] for name, key in [('coordinator', 'old_coordinator'),
                       ('gpu0', 'old_gpu0_runner'), ('gpu6', 'old_gpu6_runner'), ('child', 'current_child')]}
        pids = [r['pid'] for r in self.records.values()]
        if len(set(pids)) != 4 or any(not isinstance(pid, int) or pid <= 1 for pid in pids):
            raise ValueError('Distinct explicit process IDs required')
        self.fds, self.children = {}, {}
        for name, record in self.records.items():
            fd = os.pidfd_open(record['pid'])
            resources.callback(os.close, fd)
            self.fds[name] = fd
            actual = _read_process(record['pid'])
            _verify_process(record, actual, record['argv'])
            if actual['cwd'] != str(self.snapshot):
                raise ValueError('Unexpected scheduler/child cwd')
            argv = actual['argv']
            if name != 'child':
                module = 'v19_campaign_coordinator' if name == 'coordinator' else 'v19_campaign_runner'
                if argv[1:3] != ['-m', 'experiments.nowag_d6.' + module]:
                    raise ValueError('Target is not the expected scheduler module')
                if argv.count('--manifest') != 1 or Path(argv[argv.index('--manifest')+1]).resolve() != old_path:
                    raise ValueError('Scheduler has a different manifest')
                if name != 'coordinator':
                    if argv.count('--lane') != 1 or argv[argv.index('--lane')+1] != name:
                        raise ValueError('Wrong runner lane')
                    if argv.count('--state') != 1 or Path(argv[argv.index('--state')+1]).resolve() != self.state_paths[name]:
                        raise ValueError('Wrong runner state path')
                    if actual['ppid'] != spec['old_coordinator']['pid']:
                        raise ValueError('Runner not parented by the specified coordinator')
            elif actual['ppid'] != spec['old_gpu0_runner']['pid']:
                raise ValueError('Child not parented by the specified gpu0 runner')
        self.current_job = self.old_jobs[spec['current_job_id']]
        expected_child = [resolve_selection(arg, base_dir=self.current_job['cwd']) for arg in self.current_job['argv']]
        if self.records['child']['argv'] != expected_child or self.current_job['lane'] != 'gpu0':
            raise ValueError('Current child does not match the old manifest job')
        if not self._lane_success('gpu3', old=True):
            raise ValueError('Original gpu3 lane is not fully successful')
        self.state = {'started_at': time.time(), 'finished': False, 'completed': False,
                      'spec': spec, 'events': [], 'lanes': {},
                      'adopted_processes': {name: dict(record, last_verified_status='live')
                                            for name, record in self.records.items()},
                      'recovery': 'manual; never automatically resume old schedulers or relaunch unresolved jobs'}
        for lane, path in self.state_paths.items():
            (output/f'{lane}_state.json').symlink_to(path)
        self.save()

    def save(self):
        _atomic(self.output/'coordinator_state.json', self.state)

    def event(self, action, name, **fields):
        self.state['events'].append({'action': action, 'name': name, 'time': time.time(), **fields})
        self.save()

    def check(self, name, *, paused=False):
        record = self.records[name]
        actual = _read_process(record['pid'])
        _verify_process(record, actual, record['argv'])
        if paused and actual['state'] not in ('T', 't'):
            raise ValueError('Scheduler is not stopped; refusing kill/start')

    def stop(self, name):
        if name not in ('coordinator', 'gpu0'):
            raise ValueError('Only schedulers may be stopped')
        self.check(name)
        self.event('stop_requested', name, pid=self.records[name]['pid'])
        signal.pidfd_send_signal(self.fds[name], signal.SIGSTOP)
        self.state['adopted_processes'][name]['last_verified_status'] = 'stop_signal_sent'
        self.save()
        # Allow asynchronous signal delivery, then verify once without polling.
        time.sleep(0.1)
        self.check(name, paused=True)
        self.state['adopted_processes'][name]['last_verified_status'] = 'paused'
        self.event('stopped', name, pid=self.records[name]['pid'])

    def verify_paused_job(self):
        self.check('coordinator', paused=True)
        self.check('gpu0', paused=True)
        state = json.loads(self.state_paths['gpu0'].read_text())
        running = {key: value for key, value in state['jobs'].items() if value.get('status') in ('running', 'starting')}
        if set(running) != {self.spec['current_job_id']}:
            raise ValueError('Paused runner advanced to a different job')
        entry = running[self.spec['current_job_id']]
        if entry.get('pid') != self.records['child']['pid'] or str(entry.get('process_identity')) != str(self.records['child']['starttime']):
            raise ValueError('Current child identity differs from lane state')

    def wait_adopted(self, name):
        select.select([self.fds[name]], [], [])
        self.state['adopted_processes'][name]['last_verified_status'] = 'exited'
        self.event('exited', name, returncode=None, evidence='pidfd_readable')

    def record_child_outputs(self):
        state = json.loads(self.state_paths['gpu0'].read_text())
        entry = state['jobs'][self.spec['current_job_id']]
        errors = _output_errors(self.current_job)
        entry.update(status='failed' if errors else 'succeeded', returncode=None,
                     output_errors=errors, finished_at=time.time(), completion_basis='outputs_after_pidfd_exit')
        _atomic(self.state_paths['gpu0'], state)
        self.event('child_outputs_verified', 'child', errors=errors)

    def kill_paused(self, name):
        if name not in ('coordinator', 'gpu0'):
            raise ValueError('Only stopped schedulers may be killed')
        self.check(name, paused=True)
        self.event('kill_paused_requested', name)
        signal.pidfd_send_signal(self.fds[name], signal.SIGKILL)
        select.select([self.fds[name]], [], [])
        self.state['adopted_processes'][name]['last_verified_status'] = 'killed_after_pause'
        self.event('killed_stopped_scheduler', name)

    def start_lane(self, lane):
        path = self.state_paths.get(lane, self.output/f'{lane}_state.json')
        stdout = self.resources.enter_context((self.output/f'{lane}.stdout.log').open('x'))
        stderr = self.resources.enter_context((self.output/f'{lane}.stderr.log').open('x'))
        self.state['lanes'][lane] = {'state_path': str(path), 'started_at': time.time(), 'status': 'starting',
                                     'stdout_log': stdout.name, 'stderr_log': stderr.name}
        self.event('runner_start_requested', lane)
        child = subprocess.Popen([self.records['gpu0']['argv'][0], '-m', 'experiments.nowag_d6.v19_campaign_runner',
                                  '--manifest', str(self.manifest), '--state', str(path), '--lane', lane],
                                 cwd=self.snapshot, shell=False, stdin=subprocess.DEVNULL, stdout=stdout,
                                 stderr=stderr, pass_fds=(self.lock_fd,))
        self.children[lane] = child
        self.state['lanes'][lane].update(pid=child.pid, status='running')
        self.event('runner_started', lane, pid=child.pid)

    def wait_owned(self, lane):
        code = self.children[lane].wait()
        self.state['lanes'][lane].update(returncode=code, finished_at=time.time(), status='exited')
        self.event('runner_exited', lane, returncode=code)

    def _lane_success(self, lane, *, old=False):
        path = self.state_paths.get(lane, self.output/f'{lane}_state.json')
        state = json.loads(path.read_text())
        jobs = self.old_jobs if old else self.new_jobs
        selected = [job for job in jobs.values() if job['lane'] == lane]
        return bool(selected) and all(state['jobs'].get(j['id'], {}).get('status') == 'succeeded'
                                      and not _output_errors(j) for j in selected)

    def verify_lane(self, lane):
        # An abnormal runner exit may leave an experiment alive. Refuse phase2
        # when a lane state still identifies a live child; never kill that child.
        path = self.state_paths.get(lane, self.output/f'{lane}_state.json')
        state = json.loads(path.read_text())
        for entry in state['jobs'].values():
            if entry.get('status') in ('running', 'starting') and entry.get('pid'):
                try:
                    actual = _read_process(entry['pid'])
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if actual['state'] not in ('Z', 'X') and str(entry.get('process_identity')) == actual['starttime']:
                    raise RuntimeError('Runner exited with a live experiment child; manual recovery required')
        success = self._lane_success(lane)
        self.event('lane_verified', lane, success=success)
        return success


def handoff(spec_path, output_dir):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Use a Linux Python with pidfd_open and pidfd_send_signal (e.g. /usr/bin/python3)')
    spec = json.loads(Path(spec_path).read_text())
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Handoff requires a fresh directory; recovery is manual')
    manifest = Path(spec['new_manifest']).resolve(strict=True)
    with ExitStack() as resources:
        lock = resources.enter_context(manifest.with_suffix(manifest.suffix+'.coordinator.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        ops = None
        try:
            ops = _Operations(spec, output, resources, lock.fileno())
            ops.state['completed'] = _execute(ops)
            ops.state.update(finished=True, finished_at=time.time())
            ops.save()
            return ops.state
        except BaseException as error:
            state = ops.state if ops else {'completed': False, 'spec': spec}
            state.update(error=f'{type(error).__name__}: {error}', finished=True, finished_at=time.time())
            _atomic(output/'coordinator_state.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    return int(not handoff(args.spec, args.output_dir)['completed'])


if __name__ == '__main__':
    raise SystemExit(main())
