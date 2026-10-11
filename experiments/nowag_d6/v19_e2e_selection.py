"""Choose completed E2E trials using their saved development NLL only."""
import argparse
import json
import math
from pathlib import Path

from experiments.nowag_d6.v19_model_driver import atomic_json, file_sha256
from experiments.nowag_d6.v19_validation_score import select_candidate


def selected_learning_rate(path, checkpoint):
    result = json.loads(Path(path).read_text())
    if result.get('completed') is not True or result.get('selected') not in result.get('scores', []):
        raise ValueError('Learning-rate selection must be completed and contain its winner')
    if Path(result['protocol']['checkpoint']).resolve() != Path(checkpoint).resolve():
        raise ValueError('Selected learning rate belongs to another source checkpoint')
    rate = float(result['selected']['learning_rate'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Selected learning rate must be finite and positive')
    return rate


def summarize(paths, labels):
    if len(paths) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError('Unique labels must match progress paths')
    scores, protocol = [], None
    fields = ('model', 'checkpoint', 'token_cache', 'train_sequences', 'validation_sequences',
              'context_length', 'epochs', 'batch_size', 'seed', 'tune_group_scales',
              'binary_assignments', 'distillation', 'residual_rank', 'validation_interval')
    for path, label in zip(paths, labels):
        path = Path(path).resolve(strict=True)
        report = json.loads(path.read_text())
        if report.get('completed') is not True:
            raise ValueError(f'Training not completed: {path}')
        args = report['arguments']
        current = {key: args.get(key) for key in fields}
        current['split_audit'] = report['split_audit']
        if protocol is not None and current != protocol:
            raise ValueError('Learning-rate trials must share their protocol')
        protocol = current
        checkpoint = Path(args['output_dir']) / f"{args['run_stamp']}_checkpoint.pt"
        checkpoint = checkpoint.resolve(strict=True)
        scores.append({'label': label, 'nll': report['best_validation_nll'],
                       'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
                       'progress': str(path), 'progress_sha256': file_sha256(path),
                       'learning_rate': args['learning_rate'],
                       'selected_epoch': report['selected_epoch'],
                       'selected_step': report.get('selected_step'),
                       'selected_initial_anchor': report['selected_epoch'] == -1})
    selected = select_candidate(scores)
    return {'completed': True, 'selection': 'lowest saved independent development NLL',
            'uses_test_benchmarks_for_selection': False, 'protocol': protocol,
            'scores': scores, 'selected': selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--progress', nargs='+', required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new timestamped selection output')
    result = summarize(args.progress, args.labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)


if __name__ == '__main__':
    main()
