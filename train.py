import os.path
import warnings
import time
from argument_parser import args
from base_mdn import *
from datamodule import *
from models.gru_gnn import *
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
warnings.filterwarnings("ignore", ".*Checkpoint directory*")


def main(save_name, encoder, decoder):
    path = f"saved_models/{args.dataset}/{save_name}.ckpt"

    if os.path.exists(path) and not args.overwrite_data:
        ckpt = path
    else:
        ckpt = None

    callback_list = []

    if args.store_data:
        checkpoint_callback = ModelCheckpoint(dirpath=f"saved_models/{args.dataset}/", filename=f"{save_name}")
        checkpoint_callback_NLL = ModelCheckpoint(dirpath=f"saved_models/{args.dataset}/",
                                                  filename=f"{save_name}_NLL", monitor="val_nll", mode="min")
        checkpoint_callback_ADE = ModelCheckpoint(dirpath=f"saved_models/{args.dataset}/",
                                                  filename=f"{save_name}_ADE", monitor="val_ade", mode="min")

        callback_list.append(checkpoint_callback)
        callback_list.append(checkpoint_callback_NLL)
        callback_list.append(checkpoint_callback_ADE)

    strategy = "auto"
    if torch.cuda.is_available() and args.use_cuda:
        devices = -1
        accelerator = "auto"
        if torch.cuda.device_count() > 1:
            strategy = 'ddp'
    else:
        devices = 1
        accelerator = "cpu"

    model = LitEncoderDecoder(encoder, decoder, args)
    datamodule = LitDataModule(args)

    if args.tune_lr:
        trainer = Trainer(accelerator="auto", devices=devices)
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, early_stop_threshold=None)
        new_lr = lr_finder.suggestion()
        print(new_lr)
        model.learning_rate = new_lr
    elif args.tune_batch_size:
        trainer = Trainer(accelerator="auto", devices=devices)
        batch_size_finder = trainer.tuner.scale_batch_size(model, datamodule=datamodule)
        print(batch_size_finder)

    if args.dry_run or not args.use_logger:
        logger = False
    else:
        run_name = f"{save_name}_{time.strftime('%d-%m_%H:%M:%S')}"
        logger = WandbLogger(project="mtp-go", name=run_name)

    trainer = Trainer(max_epochs=args.epochs,
                      accelerator=accelerator,
                      devices=devices,
                      strategy=strategy,
                      deterministic=False,
                      gradient_clip_val=args.clip,
                      enable_checkpointing=args.store_data,
                      fast_dev_run=args.dry_run,
                      log_every_n_steps=args.log_interval,
                      callbacks=callback_list,
                      logger=logger)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt)


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

    n_features = 8 if args.motion_model in ('singletrack', 'unicycle', 'curvature', 'curvilinear') else 9
    static_f_dim = v_types * int(args.n_ode_static)  # 0 if static not used in N-ODE

    dt = 2e-1
    max_l = int(input_len * (1 / dt)) + 1

    # Pure integrators
    if args.motion_model == '1Xint':
        m_model = SingleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == '2Xint':
        m_model = DoubleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == '3Xint':
        m_model = TripleIntegrator(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)

    # Orientation-based
    elif args.motion_model == 'singletrack':
        m_model = KinematicSingleTrack(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
        if args.init_static:
            static_f_dim = 2
    elif args.motion_model == 'unicycle':
        m_model = Unicycle(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvature':
        m_model = Curvature(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures)
    elif args.motion_model == 'curvilinear':
        m_model = CurviLinear(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures, u1_lim=args.u1_lim)

    # Neural ODEs
    elif args.motion_model == 'neuralode':
        m_model = FirstOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                      static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                      n_layers=args.n_ode_layers)
    else:
        m_model = SecondOrderNeuralODE(solver=args.ode_solver, dt=dt, mixtures=args.n_mixtures,
                                       static_f_dim=static_f_dim, n_hidden=args.n_ode_hidden,
                                       n_layers=args.n_ode_layers)
    save_name = type(m_model).__name__

    d_str = args.dataset
    full_save_name = f"{save_name}{args.hidden_size}G{args.n_gnn_layers}{d_str[0].upper() + d_str[1:]}{args.add_name}"
    print(f'----------------------------------------------------')
    print(f'\nGetting ready to train model: {full_save_name} \n')
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
