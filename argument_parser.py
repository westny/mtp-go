from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")
    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    elif value.lower() in false_vals:
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


parser = ArgumentParser(description='Trajectory prediction arguments')
# Trainer args
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size (default: 128)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping (default: 5)')
parser.add_argument('--teacher-forcing', type=float, default=0.2,
                    help='Start probability of teacher forcing (default: 0.2)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='report interval (default: 100)')

# Model specific arguments
parser.add_argument('--u1-lim', type=float, default=10,
                    help='u1 control limit')
parser.add_argument('--u2-lim', type=float, default=10,
                    help='u2 control limit')
parser.add_argument('--ode-solver', type=str, default='rk4',
                    help='choose explicit solver [ef | mp | heun | rk3 | ssprk3 | rk4 | dopri5 | impl_adam]')
parser.add_argument('--n-mixtures', type=int, default=8,
                    help='number of mixtures (default: 8)')
parser.add_argument('--motion-model', type=str, default='2Xnode',
                    help='choice of motion model (default: 2Xnode)')
parser.add_argument('--hidden-size', type=int, default=64,
                    help='RNN hidden layer size (default: 64)')
parser.add_argument('--gnn-layer', type=str, default="natt",
                    help='Which type of GNN layer to use (default: natt)')
parser.add_argument('--n-gnn-layers', type=int, default=1,
                    help='n graph layers (default: 1)')
parser.add_argument('--n-ode-hidden', type=int, default=16,
                    help='n graph layers (default: 16)')
parser.add_argument('--n-ode-layers', type=int, default=1,
                    help='n ode layers (default: 1)')
parser.add_argument('--n-heads', type=int, default=1,
                    help='n attention heads in GAT (default: 1)')
parser.add_argument('--init-static', type=str_to_bool, default=False, const=True,
                    nargs="?", help='If the initial hidden state should be learned using static f')
parser.add_argument('--n-ode-static', type=str_to_bool, default=False, const=True,
                    nargs="?", help='If the static features should be used in the neural ODE model')
parser.add_argument('--use-edge-features', type=str_to_bool, default=True, const=True,
                    nargs="?", help='If GNN-layers should use edge features/weights (default: True)')

# Program arguments
parser.add_argument('--dataset', type=str, default='highD',
                    help='which data set to train on: |highD|rounD|inD|')
parser.add_argument('--sparse', type=str_to_bool, default=False, const=True,
                    nargs="?", help='If the model should use sparse adjacency matrices')
parser.add_argument('--seed', type=int, default=1234,
                    help='random seed')
parser.add_argument('--use-logger', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if logger should be used')
parser.add_argument('--use-cuda', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if cuda exists and should be used')
parser.add_argument('--n-workers', type=int, default=1,
                    help='number of workers in dataloader')
parser.add_argument('--store-data', type=str_to_bool, default=True,
                    const=True, nargs="?", help='if checkpoints should be stored')
parser.add_argument('--overwrite-data', type=str_to_bool, default=False,
                    const=True, nargs="?", help='overwrite if model exists (default: False)')
parser.add_argument('--add-name', type=str, default="",
                    help='additional string to add to save name')
parser.add_argument('--dry-run', type=str_to_bool, default=False,
                    const=True, nargs="?", help='verify the code and the model')
parser.add_argument('--tune-lr', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if the initial learning rate should be calculated')
parser.add_argument('--tune-batch-size', type=str_to_bool, default=False,
                    const=True, nargs="?", help='if the batch size should be calculated')
parser.add_argument('--small-ds', type=str_to_bool, default=False,
                    const=True, nargs="?", help='Use tiny versions of dataset for fast test')

args = parser.parse_args()
