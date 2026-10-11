"""Persistent sequential experiment jobs; argv is always executed without a shell.

output_paths declares each job's owned output roots (files or directories).
The runner does not precreate those roots. Relative paths use each job's cwd;
cwd and protected_paths resolve relative to the manifest directory. A lane may
depend only on jobs in the same lane. Recovery never resumes a live child twice.
Optional completed_json paths must parse with completed is True. Partial output
roots are never removed or reused; recovery marks needs_fresh_outputs instead.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time


def _atomic(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _output_path(value, base):
    # Preserve dangling symlink identity for ownership checks.
    return Path(os.path.abspath(base / value))


def _protected(output, sources):
    resolved = output.resolve()
    for source in sources:
        if resolved == source or source.is_relative_to(resolved) or resolved.is_relative_to(source):
            raise ValueError(f'Output overlaps protected path: {output}')
        if output.exists() and source.exists() and output.samefile(source):
            raise ValueError(f'Output aliases protected file: {output}')


def resolve_selection(argument, *, base_dir=None):
    """Resolve one whole-argv placeholder from a completed validation result."""
    match = re.fullmatch(r'\$\{selection:(.+):([^:}]+)\}', argument)
    if not match:
        if '${selection:' in argument:
            raise ValueError('Selection placeholder must occupy a complete argv element')
        return argument
    score_path = _path(match[1], Path(base_dir or '.').resolve())
    score = json.loads(score_path.read_text())
    if score.get('completed') is not True:
        raise ValueError('Selection result is not completed')
    items = score.get('scores', [])
    if match[2] == 'selected':
        item = score.get('selected')
        if not isinstance(item, dict) or item not in items:
            raise ValueError('Selected entry is absent from validation scores')
    else:
        matches = [item for item in items if item.get('label') == match[2]]
        if len(matches) != 1:
            raise ValueError('Selection label must match exactly one score')
        item = matches[0]
    checkpoint = _path(item['checkpoint'], score_path.parent)
    if not checkpoint.is_file():
        raise ValueError('Selected checkpoint is missing')
    return str(checkpoint)


def _identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def _output_errors(job):
    errors = [f'Missing output: {path}' for path in job['expected_outputs'] if not path.exists()]
    for path in job['completed_json']:
        try:
            result = json.loads(path.read_text())
            if not isinstance(result, dict) or result.get('completed') is not True:
                errors.append(f'JSON is not completed: {path}')
        except (OSError, ValueError) as error:
            errors.append(f'Invalid completion JSON: {path}: {type(error).__name__}')
    return errors


def _jobs(manifest, base):
    jobs = {}
    for spec in manifest['jobs']:
        name = spec['id']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in jobs:
            raise ValueError('Job IDs must be unique safe identifiers')
        argv = spec['argv']
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or '\0' in arg for arg in argv):
            raise ValueError('argv must be a nonempty list of strings')
        env = spec.get('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k
                                          or not isinstance(v, str) or '\0' in v for k, v in env.items()):
            raise ValueError('env must map valid names to strings')
        cwd = _path(spec['cwd'], base)
        if not cwd.is_dir():
            raise ValueError('Job cwd must exist')
        expected = [_output_path(p, cwd) for p in spec['expected_outputs']]
        completed = [_output_path(p, cwd) for p in spec.get('completed_json', [])]
        if not expected:
            raise ValueError('Each job must declare expected outputs')
        outputs = [_output_path(p, cwd) for p in spec.get('output_paths', sorted({str(p.parent) for p in expected}))]
        if not outputs or any(not any(p == root or p.is_relative_to(root) for root in outputs) for p in expected + completed):
            raise ValueError('Expected outputs must fall under declared output_paths')
        fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        jobs[name] = dict(spec, cwd=str(cwd), expected_outputs=expected, output_paths=outputs,
                          fingerprint=fingerprint, lane=spec.get('lane', 'default'), env=env,
                          depends_on=spec.get('depends_on', []), completed_json=completed)
    pending, order = dict(jobs), []
    roots = []
    for name, job in jobs.items():
        for output in job['output_paths']:
            for other_name, other in roots:
                if name != other_name:
                    try:
                        _protected(output, [other.resolve()])
                    except ValueError as error:
                        raise ValueError(f'Jobs {name} and {other_name} have overlapping output ownership') from error
            roots.append((name, output))
    for name, job in jobs.items():
        for dependency in job['depends_on']:
            if dependency not in jobs or jobs[dependency]['lane'] != job['lane']:
                raise ValueError(f'Unknown or cross-lane dependency for {name}: {dependency}')
    while pending:
        ready = [name for name, job in pending.items() if all(dep not in pending for dep in job['depends_on'])]
        if not ready:
            raise ValueError('Dependency cycle')
        # Recompute after each choice so a newly ready early manifest job wins
        # over later independent jobs (e.g. score model A before training B).
        order.append(pending.pop(ready[0]))
    return order


def run_campaign(manifest_path, state_path, *, lane=None):
    manifest_path, state_path = Path(manifest_path).resolve(), Path(state_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    ordered = _jobs(manifest, manifest_path.parent)
    if lane is not None and lane not in {job['lane'] for job in ordered}:
        raise ValueError('Unknown lane')
    selected = [job for job in ordered if lane is None or job['lane'] == lane]
    protected = [_path(p, manifest_path.parent) for p in manifest.get('protected_paths', [])]
    log_root = state_path.with_suffix('.logs')
    lock_path = state_path.with_suffix(state_path.suffix + '.lock')
    for path in (state_path, log_root, lock_path):
        _protected(path, protected)
    for job in selected:
        for output in job['output_paths'] + job['expected_outputs'] + job['completed_json']:
            _protected(output, protected + [state_path, log_root, lock_path, manifest_path])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Campaign state is already owned by a running runner') from error
        state = json.loads(state_path.read_text()) if state_path.exists() else {'format': 'v19_campaign_v1', 'jobs': {}}
        if state.get('format') != 'v19_campaign_v1':
            raise ValueError('Unrecognized runner state')
        for job in selected:
            previous = state['jobs'].get(job['id'], {})
            if previous and previous.get('fingerprint') != job['fingerprint']:
                raise ValueError('Existing job definition changed; use a new job ID and output roots')
            for output in job['output_paths'] + job['expected_outputs'] + job['completed_json']:
                if (output.exists() or output.is_symlink()) and not previous.get('started_at'):
                    raise FileExistsError(f'Existing output has no runner ownership: {output}')
            if previous.get('status') in ('starting', 'running'):
                if not previous.get('pid'):
                    raise RuntimeError('Interrupted launch without PID evidence; refusing duplicate launch')
                identity = _identity(previous['pid'])
                if identity is not None and previous.get('process_identity') in (None, identity):
                    raise RuntimeError(f'Job child is still running: {job["id"]}')
        if log_root.exists() and not state_path.exists():
            raise FileExistsError('Existing logs have no runner state')
        log_root.mkdir(parents=True, exist_ok=True)
        _atomic(state_path, state)
        for job in selected:
            name = job['id']
            previous = state['jobs'].get(name, {})
            if any(state['jobs'].get(dep, {}).get('status') != 'succeeded' for dep in job['depends_on']):
                state['jobs'][name] = dict(previous, fingerprint=job['fingerprint'], status='blocked',
                                           reason='Dependency did not succeed', lane=job['lane'])
                _atomic(state_path, state)
                continue
            output_errors = _output_errors(job)
            if previous.get('status') == 'succeeded' and not output_errors:
                continue
            existing = [str(path) for path in job['output_paths'] + job['expected_outputs'] + job['completed_json']
                        if path.exists() or path.is_symlink()]
            if previous.get('started_at') and existing:
                state['jobs'][name] = dict(previous, status='needs_fresh_outputs', output_errors=output_errors,
                                           reason='Keep existing artifacts; add a new job ID with fresh output paths',
                                           existing_outputs=existing)
                _atomic(state_path, state)
                continue
            attempt = previous.get('attempts', 0) + 1
            stdout_path = log_root / f'{name}.attempt{attempt}.stdout.log'
            stderr_path = log_root / f'{name}.attempt{attempt}.stderr.log'
            entry = dict(fingerprint=job['fingerprint'], status='starting', started_at=time.time(), attempts=attempt,
                         stdout_log=str(stdout_path), stderr_log=str(stderr_path), lane=job['lane'])
            state['jobs'][name] = entry
            _atomic(state_path, state)
            try:
                argv = [resolve_selection(arg, base_dir=job['cwd']) for arg in job['argv']]
                for output in job['output_paths'] + job['expected_outputs'] + job['completed_json']:
                    _protected(output, protected)
                with stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
                    child = subprocess.Popen(argv, cwd=job['cwd'], env={**os.environ, **job['env']},
                                             shell=False, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
                    entry.update(status='running', pid=child.pid, process_identity=_identity(child.pid))
                    _atomic(state_path, state)
                    returncode = child.wait()
                missing = [str(p) for p in job['expected_outputs'] if not p.exists()]
                output_errors = _output_errors(job)
                entry.update(returncode=returncode, missing_outputs=missing, output_errors=output_errors,
                             status='succeeded' if returncode == 0 and not output_errors else 'failed')
            except (OSError, ValueError, KeyError) as error:
                entry.update(status='failed', error=f'{type(error).__name__}: {error}')
            entry['finished_at'] = time.time()
            _atomic(state_path, state)
        state['finished_at'] = time.time()
        _atomic(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--lane')
    args = parser.parse_args()
    state = run_campaign(args.manifest, args.state, lane=args.lane)
    return int(any(job['status'] != 'succeeded' for job in state['jobs'].values()
                   if args.lane is None or job.get('lane', 'default') == args.lane))


if __name__ == '__main__':
    raise SystemExit(main())
