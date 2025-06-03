"""
export PYTHONPATH=$PYTHONPATH:~/projects/INR  # add INR to PYTHONPATH

Evaluation on CIFAR-10-conv (task 11, see l2o_utils.py) can be run as:
    python experiments/mnist/l2o_eval.py --ckpt results/l2o_fashionmnist17_.../step_xx.pt --train_tasks 11

"""

from functools import partial
from l2o_utils import *
from l2o_train import MetaOpt, eval_meta_opt, init_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='l2o evaluation')
    parser.add_argument('--ckpt', type=str, default=None, help='path to the trained l2o checkpoint')
    parser.add_argument('--amp', action='store_true',
                        help='use automatic mixed precision for the metaopt step')
    args, device = init_config(parser, steps=1000, inner_steps=None)  # during eval, steps should equal inner_steps

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
        state_dict = torch.load(args.ckpt, map_location=device)
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

    for task in args.train_tasks:
        cgf = TEST_TASKS[task]
        print('\nEval MetaOpt, task:', cgf)
        test_acc = []
        for seed in TEST_SEEDS:
            test_acc.append(eval_meta_opt(metaopt, cgf, seed, args, device, print_interval=1, steps=args.steps,
                                          amp=args.amp))
        print('test acc for %d runs: %.2f +- %.2f' % (len(test_acc), np.mean(test_acc), np.std(test_acc)))

    print('done!', datetime.today())
