import warnings
import os.path
from argument_parser import args
from base_mdn import *
from datamodule import *
from models.gru_gnn import *
from lightning.pytorch import Trainer, seed_everything

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")


def main(save_name, encoder, decoder):
    path = f"saved_models/{args.dataset}/{save_name}"
    # path = f"plot/{args.dataset}/{save_name}"

    if os.path.exists(path + ".ckpt"):
        ckpt = path + ".ckpt"
    elif os.path.exists(path + "-v1.ckpt"):
        ckpt = path + "-v1.ckpt"
    else:
        raise NameError(f"Could not find model with name: {save_name}")

    use_cuda = args.use_cuda and torch.cuda.is_available()

    devices, accelerator = (-1, "auto") if use_cuda else (1, "cpu")

    model = LitEncoderDecoder.load_from_checkpoint(ckpt, encoder=encoder, decoder=decoder, args=args)
    datamodule = LitDataModule(args)

    trainer = Trainer(accelerator=accelerator, devices=devices)

    results = trainer.test(model, datamodule=datamodule, verbose=True)[0]

    if not args.dry_run:
        from json import dumps
        # Serializing json
        json_object = dumps(results, indent=4)

        # Writing to sample.json
        with open(f"{save_name}.json", "w") as outfile:
            outfile.write(json_object)


if __name__ == "__main__":
    seed_everything(args.seed, workers=True)
    if args.dataset == 'highD':
        input_len = 2
        v_types = 2
    elif args.dataset == 'rounD':
        input_len = 3
        v_types = 7
    else:
        input_len = 3
        v_types = 4

    n_features = 9
    static_f_dim = v_types * int(args.n_ode_static)  # 0 if static not used in N-ODE

    dt = 2e-1
    max_l = int(input_len * (1 / dt)) + 1

    if args.motion_model == 'neuralode':
        m_model = FirstOrderNeuralODE(solver=args.ode_solver,
                                      dt=dt,
                                      mixtures=args.n_mixtures,
                                      static_f_dim=static_f_dim,
                                      n_hidden=args.n_ode_hidden,
                                      n_layers=args.n_ode_layers)
    else:
        m_model = SecondOrderNeuralODE(solver=args.ode_solver,
                                       dt=dt,
                                       mixtures=args.n_mixtures,
                                       static_f_dim=static_f_dim,
                                       n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)
    save_name = type(m_model).__name__

    d_str = args.dataset
    full_save_name = f"{save_name}{args.hidden_size}G{args.n_gnn_layers}{d_str[0].upper() + d_str[1:]}{args.add_name}"
    print(f'----------------------------------------------------')
    print(f'\nGetting ready to TEST model: {full_save_name} \n')
    print(f'----------------------------------------------------')

    encoder = GRUGNNEncoder(input_size=n_features,
                            hidden_size=args.hidden_size,
                            n_mixtures=m_model.mixtures,
                            n_layers=args.n_gnn_layers,
                            gnn_layer=args.gnn_layer,
                            n_heads=args.n_heads,
                            static_f_dim=static_f_dim,
                            init_static=args.init_static,
                            use_edge_features=args.use_edge_features)
    decoder = GRUGNNDecoder(m_model,
                            hidden_size=encoder.hidden_size,
                            max_length=max_l,
                            n_layers=args.n_gnn_layers,
                            n_heads=args.n_heads,
                            static_f_dim=static_f_dim,
                            gnn_layer=args.gnn_layer,
                            init_static=args.init_static)

    main(full_save_name, encoder, decoder)
