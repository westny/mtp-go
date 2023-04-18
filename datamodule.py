import torch
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from dataclasses import dataclass


@dataclass
class MetaInfo:
    rec_id: str
    frame: int
    initial_pos: list
    vehicle_ids: list
    vehicle_types: list
    euclidian_dist: list
    maneuver_id: list
    width: list
    length: list


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.small_ds = args.small_ds
        self.n_workers = args.n_workers
        self.sparse = args.sparse
        if args.motion_model in ('singletrack', 'unicycle', 'curvature', 'curvilinear'):
            self.target = 'rotational'
        else:
            self.target = 'planar'

    def train_dataloader(self):
        dataset = TrajectoryPredictionDataset('training',
                                              self.dataset,
                                              small=self.small_ds,
                                              target=self.target,
                                              sparse=self.sparse)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        dataset = TrajectoryPredictionDataset('validation',
                                              self.dataset,
                                              small=self.small_ds,
                                              target=self.target,
                                              sparse=self.sparse)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        dataset = TrajectoryPredictionDataset('testing',
                                              self.dataset,
                                              small=self.small_ds,
                                              target=self.target,
                                              sparse=self.sparse)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=True,
                          persistent_workers=True)


class TrajectoryPredictionDataset(Dataset):
    def __init__(self,
                 train_test: str,
                 data_set_src: str = 'highD',
                 transform=None,
                 pre_transform=None,
                 feature_scaling: bool = False,
                 small: bool = False,
                 target: str = 'planar',
                 sparse: bool = False):
        super().__init__(None, transform, pre_transform)
        self.mode = train_test
        self.feat_scale = feature_scaling
        self.root = data_set_src
        self.target = target  # planar vs. rotational
        self.version = "sparse-gnn" if (sparse and data_set_src == "highD") else "gnn"
        self.ids = torch.load(f'data/{self.root}-{self.version}/{self.mode}/ids.pt')
        self.v_type_onehot = self._create_v_type_onehot(data_set_src)

        if small:
            # Smaller version for dry runs
            self.ids = self.ids[:500]

    @staticmethod
    def _create_v_type_onehot(data_set_src: str):
        if data_set_src == 'highD':
            return {'Car': torch.Tensor([1, 0]),
                    'Truck': torch.Tensor([0, 1])}
        elif data_set_src == 'rounD':
            return {'car': torch.eye(7)[0, :],
                    'van': torch.eye(7)[0, :],
                    'truck': torch.eye(7)[1, :],
                    'bus': torch.eye(7)[2, :],
                    'trailer': torch.eye(7)[3, :],
                    'motorcycle': torch.eye(7)[4, :],
                    'bicycle': torch.eye(7)[5, :],
                    'pedestrian': torch.eye(7)[6, :]}
        else:
            return {'car': torch.eye(4)[0, :],
                    'truck_bus': torch.eye(4)[1, :],
                    'bicycle': torch.eye(4)[2, :],
                    'pedestrian': torch.eye(4)[3, :]}

    def len(self):
        return len(self.ids)

    def get(self, idx):
        """
        graph_input.shape (n_vehicles, inp_seq_len, node_feats)
        graph_inp_ei.len (inp_seq_len), graph_inp_ei[-1].shape (2, n_edges)
        graph_input_ef.len (inp_seq_len), input_edge_feats[-1].shape (2, n_edges)

        graph_target.shape (n_vehicles, tar_seq_len, node_feats)
        graph_target_ei.len (tar_seq_len), graph_target_ei[-1].shape (2, n_edges)
        graph_target_ef.len (tar_seq_len), graph_target_ef[-1].shape (2, n_edges)

        """
        #  Model inputs
        graph_input = torch.load(f'data/{self.root}-{self.version}/{self.mode}/observation/dat{idx}.pt')
        nan_mask = torch.load(f'data/{self.root}-{self.version}/{self.mode}/observation/nan_mask{idx}.pt')
        graph_input[nan_mask] = 0
        graph_inp_ei = torch.load(f'data/{self.root}-{self.version}/{self.mode}/observation/edge_idx{idx}.pt')
        graph_input_ef = torch.load(f'data/{self.root}-{self.version}/{self.mode}/observation/edge_feat{idx}.pt')

        #  Model targets
        graph_target = torch.load(f'data/{self.root}-{self.version}/{self.mode}/target/dat{idx}.pt')
        graph_target_ei = torch.load(f'data/{self.root}-{self.version}/{self.mode}/target/edge_idx{idx}.pt')
        graph_target_ef = torch.load(f'data/{self.root}-{self.version}/{self.mode}/target/edge_feat{idx}.pt')
        real_mask = torch.load(f'data/{self.root}-{self.version}/{self.mode}/target/real_mask{idx}.pt')
        graph_target[torch.logical_not(real_mask)] = 0

        if self.target == 'planar':
            graph_input = graph_input[..., [0, 1, 3, 4, 5, 6, 2, 7, 8]]
            graph_target = graph_target[..., [0, 1, 3, 4, 5, 6]]  # x, y, vx, vy, ax, ay
        else:
            graph_input[..., 3] = torch.linalg.norm(graph_input[..., 3:4 + 1], dim=-1)
            graph_input = graph_input[..., [0, 1, 3, 2, 5, 6, 7, 8]]
            graph_target[..., 3] = torch.linalg.norm(graph_target[..., 3:4 + 1], dim=-1)
            graph_target = graph_target[..., [0, 1, 3, 2]]  # x, y, v, psi

            #  Static features and targets
        meta_info = torch.load(f'data/{self.root}-{self.version}/{self.mode}/meta/dat{idx}.pt')
        cf = torch.tensor(meta_info.maneuver_id).long()
        dim = torch.tensor((meta_info.length, meta_info.width)).permute(1, 0).float()
        v_type = torch.stack([self.v_type_onehot[v_type] for v_type in meta_info.vehicle_types])

        data = Data(x=graph_input, edge_index=graph_inp_ei, edge_features=graph_input_ef,
                    y=graph_target, tar_edge_index=graph_target_ei, tar_edge_features=graph_target_ef,
                    tar_real_mask=real_mask, cf=cf, dim=dim, v_type=v_type)
        return data
