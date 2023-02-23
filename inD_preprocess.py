import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import pickle
import torch
import os
import shutil

SEARCH_PATH = '../data_sets/inD'
INPUT_LENGTH = 3
PRED_HORIZON = 5
N_IN_FEATURES = 9
N_OUT_FEATURES = 7
DOWN_SAMPLE = 5
fz = 25


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


def create_directories(overwrite=True):
    data_folder = 'inD-gnn'
    root = f'data/{data_folder}'
    top_dirs = ['training', 'validation', 'testing']
    sub_dirs = ['observation', 'target', 'meta']
    for d in top_dirs:
        top_dir = f'{root}/{d}'
        if os.path.exists(top_dir):  # and overwrite:
            response = input(f'Directory exists: {top_dir} - Overwrite? yes|no \n').lower()
            if response in ('yes', 'y'):
                shutil.rmtree(top_dir)
            else:
                raise FileExistsError
        for s in sub_dirs:
            sub_dir = f'{top_dir}/{s}'
            os.makedirs(sub_dir)
    return data_folder


def euclidian(x1, y1, x2, y2):
    from math import sqrt
    r = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return r


def maneuver_label(heading_start, heading_end):
    turn_alts = np.array([-np.pi / 2, 0, np.pi / 2, np.pi])
    tmp = heading_end - heading_start
    head_diff = turn_alts - np.radians(tmp)
    wrap_to_pi = np.arctan2(np.sin(head_diff), np.cos(head_diff))
    return np.argmin(np.abs(wrap_to_pi)), tmp


def find_neighboring_nodes(df, frame, id0, x0, y0, upper_limit=10):
    def filter_ids(dist, radius=50):
        return True if dist[0] < radius else False

    df1 = df[(df.frame == frame) & (df.trackId != id0)]
    if df1.empty:
        return []
    dist = list(df1.apply(lambda x: (euclidian(x0, y0, x.xCenter, x.yCenter), x.trackId), axis=1))
    dist = list(filter(filter_ids, dist))
    dist_sorted = sorted(dist)
    del dist_sorted[upper_limit:]
    return dist_sorted


def get_meta_property(tracks_meta, vehicle_ids, prop='class'):
    prp = [tracks_meta[tracks_meta.trackId == v_id][prop].values[0] for v_id in vehicle_ids]
    return prp


def wrap_to_pi(angle, deg2rad=True):
    if deg2rad:
        angle = np.deg2rad(angle)
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_input_features(df, frame_start, frame_end, trackId=-1):
    if trackId != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == trackId)]
    else:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    x = dfx.xCenter.values
    y = dfx.yCenter.values
    psi = wrap_to_pi(dfx.heading.values)
    vx = dfx.xVelocity.values
    vy = dfx.yVelocity.values
    rho = dfx.rho.values
    theta = dfx.theta.values
    ax = dfx.xAcceleration.values
    ay = dfx.yAcceleration.values
    return x, y, psi, vx, vy, ax, ay, rho, theta


def get_adjusted_features(df, frame_start, frame_end, n_features, x0=0., y0=0., trackId=-1):
    return_array = np.empty((frame_end - frame_start + 1, n_features))
    return_array[:] = np.NaN

    if trackId != -1:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end) & (df.trackId == trackId)]
    else:
        dfx = df[(df.frame >= frame_start) & (df.frame <= frame_end)]
    try:
        first_frame = dfx.frame.values[0]
    except IndexError:
        return return_array
    frame_offset = first_frame - frame_start

    x = dfx.xCenter.values - x0
    y = dfx.yCenter.values - y0
    psi = wrap_to_pi(dfx.heading.values)
    vx = dfx.xVelocity.values
    vy = dfx.yVelocity.values
    rho = dfx.rho.values
    theta = dfx.theta.values
    ax = dfx.xAcceleration.values
    ay = dfx.yAcceleration.values

    if n_features == N_IN_FEATURES:
        feat_stack = np.stack((x, y, psi, vx, vy, ax, ay, rho, theta), axis=1)
    else:
        feat_stack = np.stack((x, y, psi, vx, vy, ax, ay), axis=1)
    return_array[frame_offset:frame_offset + feat_stack.shape[0], :] = feat_stack

    return return_array


def get_storage_dict():
    dd = {}
    for t in ['training', 'validation', 'testing']:
        dd[t] = {'id': 0, 'ids': []}
    return dd


def remove_selected_vehicles(tracks, v_ids, rm=False):
    for v_id in v_ids:
        if rm:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id)].index)
        else:
            tracks = tracks.drop(tracks[(tracks.trackId == v_id) &
                                        (tracks.xAcceleration == 0) &
                                        (tracks.yAcceleration == 0)].index)
    return tracks


def remove_parked_vehicles(tracks, tracks_meta):
    parked_vehicles = tracks_meta[(tracks_meta.initialFrame == 0) &
                                  (tracks_meta.finalFrame == tracks_meta.finalFrame.max())]
    a = parked_vehicles.trackId.values
    tracks = tracks[~tracks['trackId'].isin(a)]
    tracks_meta = tracks_meta[~tracks_meta['trackId'].isin(a)]
    return tracks, tracks_meta


def remove_still_vehicle(tracks, tracks_meta, exceptions=()):
    track_ids = pd.unique(tracks.trackId)
    a = []
    for ti in track_ids:
        if ti in exceptions:
            continue
        v_type = t_meta[t_meta.trackId == ti]['class'].iloc[0]
        if v_type in ('car', 'truck_bus'):
            dfx = tracks[tracks.trackId == ti]
            duration = dfx.trackLifetime.max()
            if duration > 2500:
                xc = dfx.xCenter.to_numpy()
                if np.all(xc == xc[0]):
                    a.append(ti)
    tracks = tracks[~tracks['trackId'].isin(a)]
    tracks_meta = tracks_meta[~tracks_meta['trackId'].isin(a)]
    return tracks, tracks_meta


def remove_parts(tracks, v_ids, duration):
    for vi, dur in zip(v_ids, duration):
        tracks = tracks.drop(tracks[(tracks.trackId == vi)
                                    & (tracks.trackLifetime > dur)].index)
    return tracks


def remove_pre_parts(tracks, v_ids, duration):
    for vi, dur in zip(v_ids, duration):
        tracks = tracks.drop(tracks[(tracks.trackId == vi)
                                    & (tracks.trackLifetime < dur)].index)
    return tracks


def add_maneuver_label(tracks, tracks_meta, rec_id):
    tracks_meta['maneuver'] = np.empty(len(tracks_meta))
    t_ids = tracks_meta.trackId.values
    t_class = tracks_meta['class'].values
    maneuver_count = dict()
    for k in range(5):
        maneuver_count[k] = 0

    for i in range(len(t_ids)):
        t_id = t_ids[i]
        # dfx = tracks[tracks.trackId == t_id]
        # if t_class[i] == 'pedestrian':
        #     m_label = 4
        # else:
        #     h = dfx.heading.values
        #     m_label, a1 = maneuver_label(h[0], h[-1])
        #
        #     if m_label == 3:
        #         yc = dfx.yCenter.values
        #         r = euclidian(0, yc[0], 0, yc[-1])
        #         if rec_id == '09':
        #             if t_id == 667:
        #                 m_label = 1
        #         elif r > 15:
        #             m_label = 2
        # maneuver_count[m_label] += 1
        m_label = 0
        tracks_meta.loc[tracks_meta['trackId'] == t_id, 'maneuver'] = m_label
    return tracks_meta


def add_polar_coordinates(x0, y0, tracks):
    def polar_to_center(x1, y1, x2, y2):
        x2 = np.ones_like(x1) * x2
        y2 = np.ones_like(y1) * y2
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a = np.arctan2(y1 - y2, x1 - x2)
        return r, a

    tracks['rho'] = np.empty(len(tracks))
    tracks['theta'] = np.empty(len(tracks))

    t_ids = pd.unique(tracks.trackId)
    for t_id in t_ids:
        y = tracks.loc[tracks.trackId == t_id, 'yCenter'].to_numpy()
        x = tracks.loc[tracks.trackId == t_id, 'xCenter'].to_numpy()
        rho, th = polar_to_center(x, y, x0, y0)
        tracks.loc[tracks['trackId'] == t_id, ['rho']] = rho
        tracks.loc[tracks['trackId'] == t_id, ['theta']] = th
    return tracks


def build_seq_edge_idx(x):
    def build_edge_idx(x):
        num_nodes = x.size(0)
        nan_indices = torch.isnan(x[:, 0]).nonzero()
        max_connective_nodes = (num_nodes - len(nan_indices)) ** 2
        real_n_nodes = num_nodes - len(nan_indices)
        E = torch.zeros((2, max_connective_nodes), dtype=torch.long)
        node_list = []
        for n in range(num_nodes):
            if n not in nan_indices:
                node_list.append(n)
        for i, node in enumerate(node_list):
            for neighbor in range(len(node_list)):
                E[0, i * real_n_nodes + neighbor] = node
            E[1, i:-1:real_n_nodes] = node
        E[1, -1] = node
        return E

    E = []
    seq_len = x.size(1)
    for i in range(seq_len):
        E.append(build_edge_idx(x[:, i]))
    return E


def build_full_seq_edge_idx(x):
    def build_full_edge_idx(num_nodes):
        E = torch.zeros((2, num_nodes * (num_nodes)), dtype=torch.long)
        for node in range(num_nodes):
            for neighbor in range(num_nodes):
                E[0, node * num_nodes + neighbor] = node
            E[1, node:-1:num_nodes] = node
        E[1, -1] = num_nodes - 1
        return E

    num_nodes = x.size(0)
    seq_len = x.size(1)
    E = []
    edge_index = build_full_edge_idx(num_nodes)
    for _ in range(seq_len):
        E.append(edge_index)
    return E


def euclidian_distance(x1, x2):
    # x1.shape (2, )
    # x2.shape (2, )
    return np.sqrt(np.sum((x1 - x2) ** 2))


def euclidian_instance(inp):
    # inp.shape (n_vehicles, n_features)
    n_vehicles = inp.shape[0]
    output = []
    for v_id in range(n_vehicles):
        for v_neighbor in range(n_vehicles):
            d = euclidian_distance(inp[v_id, :2], inp[v_neighbor, :2])
            if not np.isnan(d):
                output.append(d)
    return torch.tensor(output).unsqueeze(1).float()


def euclidian_sequence(inp):
    # inp.shape (n_vehicles, seq_len, n_features)
    seq_len = inp.shape[1]
    output = []
    for i in range(seq_len):
        output.append(euclidian_instance(inp[:, i]))
    return output


def get_frame_split(n_frames):
    all_frames = list(range(1, n_frames + 1))

    var = np.random.uniform(0, 3)
    if var < 1:
        # first variant 80-10-10
        tr = [1, all_frames[int(0.8 * n_frames) - 1]]
        val = [all_frames[int(0.8 * n_frames)], all_frames[int(0.9 * n_frames) - 1]]
        test = [all_frames[int(0.9 * n_frames)], all_frames[-1]]
    elif 1 <= var < 2:
        # scnd variant 10-80-10
        tr = [all_frames[int(0.1 * n_frames)], all_frames[int(0.9 * n_frames) - 1]]
        val = [1, all_frames[int(0.1 * n_frames) - 1]]
        test = [all_frames[int(0.9 * n_frames)], all_frames[-1]]
    else:
        # third variant 10-10-80
        tr = [all_frames[int(0.2 * n_frames)], all_frames[-1]]
        val = [1, all_frames[int(0.1 * n_frames) - 1]]
        test = [all_frames[int(0.1 * n_frames)], all_frames[int(0.2 * n_frames) - 1]]

    combo = np.random.uniform()
    if combo < 0.5:
        return tr, val, test
    else:
        return tr, test, val


def which_set(v_frames, tr, val, test):
    assert v_frames[-1] > v_frames[0]
    for set_frames, curr in zip((tr, val, test), ('training', 'validation', 'testing')):
        if v_frames[0] >= set_frames[0] and v_frames[-1] <= set_frames[-1]:
            return curr
    return None


if __name__ == "__main__":
    root = create_directories()
    # rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(0, 32 + 1)]
    # rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(11, 17 + 1)]

    # rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(7, 10 + 1)]
    #rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(0, 6 + 1)]
    rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(18, 29 + 1)]

    s_dict = get_storage_dict()
    np.random.seed(1234)
    for r_id in rec_ids:
        # only use some data
        # if r_id == '00':
        if r_id in ('00', '01', '02', '03', '04', '05', '06'):
            p0 = (143.255269808385, -57.91170481615564)
        elif r_id in ('07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'):
            p0 = (55.72110867398384, -32.74837088734138)
        elif r_id in ('18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'):
            p0 = (47.4118205383659, -28.8381176470473)
        else:
            p0 = (40.080060675120016, -25.416623842759034)
        print(f'Starting with recording {r_id}')
        meta = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_recordingMeta.csv')
        t_meta = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_tracksMeta.csv')
        tracks = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_tracks.csv', engine='pyarrow')
        # Perform some initial cleanup
        tracks, t_meta = remove_parked_vehicles(tracks, t_meta)
        if r_id == '04':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (141,))
            tracks = remove_parts(tracks, (141,), (120,))
        elif r_id == '06':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (0,))
            tracks = remove_pre_parts(tracks, (0,), (1700,))
        elif r_id == '24':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (217,))
            tracks = remove_parts(tracks, (217,), (650,))
        elif r_id == '26':
            tracks, t_meta = remove_still_vehicle(tracks, t_meta, (31, 99))
            tracks = remove_parts(tracks, (31, 99), (630, 500))
        else:
            tracks, t_meta = remove_still_vehicle(tracks, t_meta)

        t_meta = add_maneuver_label(tracks, t_meta, r_id)
        tracks = add_polar_coordinates(p0[0], p0[1], tracks)

        # Determine tr, val, test split (by frames)
        train_frames, val_frames, test_frames = get_frame_split(t_meta.finalFrame.array[-1])

        # Get data and store
        # car_ids = list(t_meta[t_meta['class'].isin(['car', 'truck', 'van'])].trackId)
        car_ids = list(t_meta[t_meta['class'].isin(['car'])].trackId)
        ii = tqdm(range(0, len(car_ids)))
        for i in ii:
            id0 = car_ids[i]
            df = tracks[tracks.trackId == id0]
            frames = list(df.frame)

            curr_set = which_set(frames, train_frames, val_frames, test_frames)
            if curr_set is None:
                # If a vehicle is within frames which are overlapping the sets
                continue

            if len(frames) < fz * (INPUT_LENGTH + PRED_HORIZON) + 1:
                continue
            for f in frames[0:-1:fz * 2]:
                fp = f + fz * INPUT_LENGTH
                fT = fp + fz * PRED_HORIZON
                if fT not in frames:
                    break
                x, y, psi, vx, vy, ax, ay, rho, theta = get_input_features(df, f, fp - 1)
                neighbors = find_neighboring_nodes(tracks, fp - 1, id0, x[-1], y[-1])
                n_SVs = len(neighbors)
                sv_ids = [int(neighbors[n][1]) for n in range(n_SVs)]
                euc_dist = [int(neighbors[n][0]) for n in range(n_SVs)]
                v_ids = [id0, *sv_ids]
                v_class = get_meta_property(t_meta, v_ids, prop='class')
                v_man = get_meta_property(t_meta, v_ids, prop='maneuver')
                v_width = get_meta_property(t_meta, v_ids, prop='width')
                v_height = get_meta_property(t_meta, v_ids, prop='length')

                x0 = p0[0]  # x[0]
                y0 = p0[1]  # y[0]

                x -= x0
                y -= y0

                meta_info = MetaInfo(r_id, fp - 1, [x0, y0], v_ids, v_class, [0, *euc_dist], v_man, v_width, v_height)
                input_array = np.empty((n_SVs + 1, fz * INPUT_LENGTH, N_IN_FEATURES))
                target_array = np.empty((n_SVs + 1, fz * PRED_HORIZON, N_OUT_FEATURES))

                input_array[0, :, :] = np.stack((x, y, psi, vx, vy, ax, ay, rho, theta), axis=1)
                target_array[0, :, :] = get_adjusted_features(df, fp, fT - 1, N_OUT_FEATURES, x0, y0)
                for j, n in enumerate(range(0, n_SVs)):
                    (dist, sv_id) = neighbors[n]
                    input_array[j + 1, :, :] = get_adjusted_features(tracks, f, fp - 1, N_IN_FEATURES, x0, y0, sv_id)
                    target_array[j + 1, :, :] = get_adjusted_features(tracks, fp, fT - 1, N_OUT_FEATURES, x0, y0, sv_id)

                input_array = input_array[:, -1:0:-DOWN_SAMPLE][:, ::-1, :]
                target_array = target_array[:, -1:0:-DOWN_SAMPLE][:, ::-1, :]

                # Build edge indices
                input_edge_index = build_seq_edge_idx(torch.tensor(input_array))
                target_edge_index = build_seq_edge_idx(torch.tensor(target_array))
                inference_target_edge_index = build_full_seq_edge_idx(torch.tensor(target_array))

                # Build edge features
                input_edge_feat = euclidian_sequence(input_array)
                target_edge_feat = euclidian_sequence(target_array)

                # Convert to torch tensors
                input_array = torch.from_numpy(input_array).float()
                target_array = torch.from_numpy(target_array).float()

                # Compute masks
                input_nan_mask = torch.isnan(input_array)
                target_real_mask = ~torch.isnan(target_array)

                if np.isnan(input_array[:, -1, :]).any():
                    import pdb

                    pdb.set_trace()

                current_id = s_dict[curr_set]['id']

                torch.save(input_array, f'data/{root}/{curr_set}/observation/dat{current_id}.pt')
                torch.save(input_nan_mask, f'data/{root}/{curr_set}/observation/nan_mask{current_id}.pt')
                torch.save(input_edge_index, f'data/{root}/{curr_set}/observation/edge_idx{current_id}.pt')
                torch.save(input_edge_feat, f'data/{root}/{curr_set}/observation/edge_feat{current_id}.pt')

                torch.save(target_array, f'data/{root}/{curr_set}/target/dat{current_id}.pt')
                torch.save(target_real_mask, f'data/{root}/{curr_set}/target/real_mask{current_id}.pt')
                torch.save(target_edge_index, f'data/{root}/{curr_set}/target/edge_idx{current_id}.pt')
                torch.save(target_edge_feat, f'data/{root}/{curr_set}/target/edge_feat{current_id}.pt')
                torch.save(inference_target_edge_index,
                           f'data/{root}/{curr_set}/target/full_edge_idx{current_id}.pt')
                torch.save(meta_info, f'data/{root}/{curr_set}/meta/dat{current_id}.pt')

                s_dict[curr_set]['ids'].append(current_id)
                s_dict[curr_set]['id'] += 1
    torch.save(s_dict['training']['ids'], f'data/{root}/training/ids.pt')
    torch.save(s_dict['validation']['ids'], f'data/{root}/validation/ids.pt')
    torch.save(s_dict['testing']['ids'], f'data/{root}/testing/ids.pt')
