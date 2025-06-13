"""
This script evaluates an optimizee model that is trained using a Meta-Optimizer (MetaOpt) on a specified task (e.g., CIFAR-10-conv).
Usage:
    python -m src.l2o.l2o_eval --ckpt <model_checkpoint_path> --train_tasks <task_id>

- Loads a trained MetaOpt model or a baseline optimizer (e.g., Adam, SGD).
- Applies it to optimize the target model (optimizee) on the specified task with multiple random seeds.
- Saves per-task accuracy curves to CSVs

"""

from functools import partial
from src.l2o.l2o_utils import *
from src.l2o.l2o_train import MetaOpt, eval_meta_opt, init_config, eval_meta_opt2
import csv
import os
import numpy as np
from datetime import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='l2o evaluation')
    parser.add_argument('--ckpt', type=str, default=None, help='path to the trained l2o checkpoint')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision for the metaopt step')
    args, device = init_config(parser, steps=1000, inner_steps=None)  # during eval, steps should equal inner_steps
    name = args.ckpt if args.ckpt else args.opt + "_" + str(args.lr) + '_' + str(args.wd)
    seed_everything(args.seed)

    train_cfg_ = TEST_TASKS[np.random.choice(args.train_tasks)]
    model, _, _ = init_model(train_cfg_, args)
    layer_layout = get_layout(model)
    print('assuming fixed model', model, 'layer_layout', layer_layout)

    if args.ckpt in [None, 'none']:

        opt_args = {}
        if args.opt == 'adam':
            opt_fn = torch.optim.Adam
        elif args.opt == 'adamw':
            opt_fn = torch.optim.AdamW
        elif args.opt == 'sgd':
            opt_fn = torch.optim.SGD
            opt_args = {'momentum': 0.9}
        else:
            raise NotImplementedError(f'unknown optimizer {args.opt}')
        metaopt = partial(opt_fn, lr=args.lr, weight_decay=args.wd, **opt_args)
        print(f'Using {args.opt}')

    else:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=False)
        if 'metaopt_cfg' in state_dict:
            metaopt_cfg = state_dict['metaopt_cfg']
            metaopt_cfg['layer_layout'] = layer_layout
            print('loaded metaopt_cfg from state_dict', metaopt_cfg)
        else:
            metaopt_cfg = dict(hid=[args.hid] * args.layers,
                               rnn=args.model.lower() == 'lstm',
                               momentum=args.momentum,
                               preprocess=not args.no_preprocess,
                               keep_grads=args.keep_grads,
                               layer_layout=layer_layout,
                               gnn=args.gnn,
                               layers=args.layers,
                               heads=args.heads,
                               wave_pos_embed=args.wave_pos_embed)
            print('init metaopt_cfg from cmd args', metaopt_cfg)
        metaopt = MetaOpt(**metaopt_cfg).to(device).eval()
        metaopt.load_state_dict(state_dict['model_state_dict'])
        #print('MetaOpt with %d params' % sum([p.numel() for p in metaopt.parameters()]),
        #      'loaded from step %d' % state_dict['step'])

    # <<< NEW: Create directory for all results files
    os.makedirs("eval_results", exist_ok=True)

    # <<< NEW: Prepare CSV for summary final accuracy results
    summary_file = os.path.join("eval_results", f"summary_results_{args.train_tasks}_{name}.csv")
    with open(summary_file, mode='w+', newline='') as f_summary:
        writer_summary = csv.writer(f_summary)
        writer_summary.writerow(['task_id', 'mean_acc', 'std_acc', 'final_accs_per_seed'])  # header

        # Loop over each task specified
        for task in args.train_tasks:
            cgf = TEST_TASKS[task]
            print('\nEval MetaOpt, task:', cgf)

            final_accs = []   # <<< NEW: store final accuracy for each seed
            all_traces = []   # <<< NEW: store per-iteration accuracy traces for each seed

            # Loop over seeds for statistical robustness
            for seed in TEST_SEEDS:
                # <<< NEW: eval_meta_opt now returns list of per-step accuracies, not just final accuracy
                acc_trace = eval_meta_opt2(metaopt, cgf, seed, args, device,
                                         print_interval=20, steps=args.steps, amp=args.amp)
                final_accs.append(acc_trace[-1])  # final accuracy for this seed
                all_traces.append(acc_trace)      # full trace for this seed

            # Compute summary stats on final accuracies
            mean_acc = np.mean(final_accs)
            std_acc = np.std(final_accs)
            print(f'test acc for {len(final_accs)} runs: {mean_acc:.2f} +- {std_acc:.2f}')

            # Write summary CSV (final accuracy statistics)
            writer_summary.writerow([task, mean_acc, std_acc, final_accs])

            # <<< NEW: Write per-iteration accuracy traces for this task into a CSV for plotting
            curve_file = os.path.join("eval_results", f"curve_task_{args.train_tasks}_{name}.csv")
            with open(curve_file, mode='w+', newline='') as f_curve:
                writer_curve = csv.writer(f_curve)
                # header: step + one column per seed
                writer_curve.writerow(['step'] + [f'seed_{i}' for i in range(len(TEST_SEEDS))])

                num_points = len(all_traces[0])  # number of accuracy points (per seed)
                for i in range(num_points):
                    # steps assumed spaced by print_interval=20 (change if you used different interval)
                    step_num = i * 20  
                    # gather ith accuracy for each seed
                    row = [step_num] + [trace[i] for trace in all_traces]
                    writer_curve.writerow(row)

    print('done!', datetime.today())
