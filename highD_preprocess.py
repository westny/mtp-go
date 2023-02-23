import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import torch
import os
import shutil
import copy

from scipy.interpolate import splev, splrep
from scipy.signal import savgol_filter

import warnings

warnings.filterwarnings("error")

SEARCH_PATH = '../data_sets/highD'  # Folder containing original data
INPUT_LENGTH = 2
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
    save_dir = 'highD-gnn'
    root = f'data/{save_dir}'
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
        os.makedirs(top_dir)
        for s in sub_dirs:
            sub_dir = f'{top_dir}/{s}'
            os.makedirs(sub_dir)
    return save_dir


def align_x_w_front(tracks_meta, tracks):
    """
    The coordinates are given wrt the upper left corner of the bounding box
    this function modifies the dataframe such that the coordinates are align
    with the center of the front of the vehicle.
    """
    ids = tracks_meta.trackId
    dDs = tracks_meta.drivingDirection
    for i, dD in zip(ids, dDs):
        tracks.loc[tracks.trackId == i, 'y'] += tracks.loc[tracks.trackId == i, 'height'] / 2
        if dD == 2:
            tracks.loc[tracks.trackId == i, 'x'] += tracks.loc[tracks.trackId == i, 'width']
    return tracks


def compute_lane_w(rec_meta):
    upper_l = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    lower_l = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]
    upper_l = np.mean(np.diff(upper_l))
    lower_l = np.mean(np.diff(lower_l))
    return np.mean([upper_l, lower_l])


def get_lane_markings(rec_meta):
    ulm_flt = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    llm_flt = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]
    ulm_flt.extend(llm_flt)
    return np.array(ulm_flt)


def compute_road_w(rec_meta):
    upper_l = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    lower_l = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]
    upper_l = upper_l[-1] - upper_l[0]
    lower_l = lower_l[-1] - lower_l[0]
    return upper_l, lower_l


def get_road_edge_markings(rec_meta):
    ulm_flt = [float(l) for l in list(rec_meta['upperLaneMarkings'])[0].split(';')]
    llm_flt = [float(l) for l in list(rec_meta['lowerLaneMarkings'])[0].split(';')]
    return ulm_flt[0], llm_flt[0]


def get_dyl(y, dD, lm, lw):
    dy = 2 * (y - lm) / lw - 1
    if dD == 2:
        dy *= (-1)
    return dy


def get_dy(y, dD, curr_lane_id, lm, lw):
    dy = 2 * (y - lm[curr_lane_id - 2]) / lw - 1
    if dD == 2:
        dy *= (-1)
    return dy


def euclidian(x1, y1, x2, y2):
    from math import sqrt
    r = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return r


def find_neighboring_nodes(df, frame, id0, x0, y0, driving_dir=1, upper_limit=10):
    def filter_ids(dist, radius=300):
        return True if dist[0] < radius else False

    df1 = df[(df.frame == frame) & (df.trackId != id0) & (df.drivingDirection == driving_dir)]
    if df1.empty:
        return []
    dist = list(df1.apply(lambda x: (euclidian(x0, y0, x.x, x.y), x.trackId), axis=1))
    dist = list(filter(filter_ids, dist))
    dist_sorted = sorted(dist)
    del dist_sorted[upper_limit:]
    return dist_sorted


def get_meta_property(tracks_meta, vehicle_ids, prop='class'):
    prp = [tracks_meta[tracks_meta.trackId == v_id][prop].values[0] for v_id in vehicle_ids]
    return prp


def get_maneuver(tracks, f, vehicle_ids, prop='maneuver'):
    prp = [tracks[(tracks.trackId == v_id) & (tracks.frame == f)][prop].values[0] for v_id in vehicle_ids]
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
    x = dfx.x.values
    y = dfx.y.values
    psi = dfx.heading.values
    vx = dfx.xVelocity.values
    vy = dfx.yVelocity.values
    ax = dfx.xAcceleration.values
    ay = dfx.yAcceleration.values
    dy = dfx.laneDisplacement.values
    dyr = dfx.roadDisplacement.values
    return x, y, psi, vx, vy, ax, ay, dy, dyr


def get_adjusted_features(df, frame_start, frame_end, n_features, x0=0., y0=0., driving_dir=1, trackId=-1):
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

    x = dfx.x.values - x0
    y = dfx.y.values - y0
    psi = dfx.heading.values
    vx = dfx.xVelocity.values
    vy = dfx.yVelocity.values
    ax = dfx.xAcceleration.values
    ay = dfx.yAcceleration.values
    dy = dfx.laneDisplacement.values
    dyr = dfx.roadDisplacement.values

    if driving_dir == 1:
        x *= (-1)
        vx *= (-1)
        ax *= (-1)
    else:
        y *= (-1)
        vy *= (-1)
        ay *= (-1)

    if n_features == N_IN_FEATURES:
        feat_stack = np.stack((x, y, psi, vx, vy, ax, ay, dy, dyr), axis=1)
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


def add_driving_direction(tracks_meta, tracks):
    tracks['drivingDirection'] = np.empty(len(tracks))
    t_ids = pd.unique(tracks.trackId)
    for t_id in t_ids:
        driving_direction = tracks_meta[tracks_meta.trackId == t_id].drivingDirection.values[0]
        tracks.loc[tracks['trackId'] == t_id, ['drivingDirection']] = driving_direction
    return tracks


def add_displacement_feat(rec_meta, tracks_meta, tracks):
    tracks['roadDisplacement'] = np.empty(len(tracks))
    tracks['laneDisplacement'] = np.empty(len(tracks))
    ur, lr = get_road_edge_markings(rec_meta)
    ruw, rlw = compute_road_w(rec_meta)

    lm = get_lane_markings(rec_meta)
    lw = compute_lane_w(rec_meta)
    t_ids = pd.unique(tracks.trackId)
    for t_id in t_ids:
        driving_dir = int(tracks_meta[tracks_meta.trackId == t_id].drivingDirection)
        lane_ids = tracks.loc[tracks.trackId == t_id, 'laneId'].to_numpy()
        y = tracks.loc[tracks.trackId == t_id, 'y'].to_numpy()
        d_y = get_dy(y, driving_dir, lane_ids, lm, lw)

        marking, width = (ur, ruw) if driving_dir == 1 else (lr, rlw)
        d_y_r = get_dyl(y, driving_dir, marking, width)

        tracks.loc[tracks['trackId'] == t_id, ['laneDisplacement']] = d_y
        tracks.loc[tracks['trackId'] == t_id, ['roadDisplacement']] = d_y_r
    return tracks


def add_heading_feat(tracks_meta, tracks):
    tracks['heading'] = np.empty(len(tracks))
    t_ids = pd.unique(tracks.trackId)
    for t_id in t_ids:
        driving_dir = int(tracks_meta[tracks_meta.trackId == t_id].drivingDirection)
        y = tracks.loc[tracks.trackId == t_id, 'y'].to_numpy()
        x = tracks.loc[tracks.trackId == t_id, 'x'].to_numpy()
        x0 = x[0]
        y0 = y[0]
        x_corr = x - x0
        y_corr = y - y0
        if driving_dir == 1:
            x_corr *= (-1)
        else:
            y_corr *= (-1)
        try:
            try:
                tck = splrep(x_corr, y_corr, k=3, s=1.2e-3, task=0)
            except RuntimeWarning:
                tck = splrep(x_corr, y_corr, k=3, s=1.2e-2, task=0)
            if np.isnan(tck[1]).any():
                psi = np.arctan2(np.diff(y_corr), np.diff(x_corr))
                psi = np.insert(psi, 0, psi[0])
            else:
                psi = splev(x_corr, tck, der=1)
        except:
            psi = np.zeros_like(x)
        tracks.loc[tracks['trackId'] == t_id, ['heading']] = psi
    return tracks


def add_maneuver(tracks_meta, tracks):
    tracks['maneuver'] = np.empty(len(tracks))
    t_ids = pd.unique(tracks.trackId)
    for t_id in t_ids:
        n_lane_changes = int(tracks_meta[tracks_meta.trackId == t_id].numLaneChanges)
        if n_lane_changes == 0:
            tracks.loc[tracks['trackId'] == t_id, ['maneuver']] = 3
        else:
            driving_dir = int(tracks_meta[tracks_meta.trackId == t_id].drivingDirection)
            visited_lanes = list(pd.unique(tracks[tracks.trackId == t_id].laneId))
            add = []
            for l in visited_lanes:
                frames = tracks[(tracks.trackId == t_id) & (tracks.laneId == l)].frame.to_numpy()
                if np.any(np.diff(frames) != 1):
                    add.append(l)
            visited_lanes.extend(add)
            final_lane = visited_lanes[-1]
            tracks.loc[(tracks['trackId'] == t_id) & (tracks['laneId'] == final_lane), ['maneuver']] = 3
            once = False
            for i, lane in enumerate(visited_lanes):
                if i == len(visited_lanes) - 1:
                    break
                next_lane = visited_lanes[i + 1]
                frames = tracks[(tracks.trackId == t_id) & (tracks.laneId == lane)].frame.to_numpy()
                # Reality check
                if np.any(np.diff(frames) != 1):
                    idx = (np.diff(frames) != 1).nonzero()[0][0]
                    if not once:
                        frames = frames[:idx]
                        once = True
                    else:
                        frames[idx:]
                if driving_dir == 1:
                    if lane < next_lane:
                        lane_changes = [0, 1, 2]
                    else:
                        lane_changes = [4, 5, 6]
                else:
                    if lane < next_lane:
                        lane_changes = [4, 5, 6]
                    else:
                        lane_changes = [0, 1, 2]

                for f_i, frame in enumerate(frames[::-1]):
                    if f_i < fz:
                        tracks.loc[(tracks['trackId'] == t_id) &
                                   (tracks['frame'] == frame), ['maneuver']] = lane_changes[0]
                    elif f_i < 3 * fz:
                        tracks.loc[(tracks['trackId'] == t_id) &
                                   (tracks['frame'] == frame), ['maneuver']] = lane_changes[1]
                    elif f_i < 5 * fz:
                        tracks.loc[(tracks['trackId'] == t_id) &
                                   (tracks['frame'] == frame), ['maneuver']] = lane_changes[2]
                    else:
                        tracks.loc[(tracks['trackId'] == t_id) &
                                   (tracks['frame'] == frame), ['maneuver']] = 3
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
    np.random.seed(1234)
    save_dir = create_directories()
    rec_ids = ['0' + str(f) if len(str(f)) < 2 else str(f) for f in range(1, 60 + 1)]
    s_dict = get_storage_dict()

    for r_id in rec_ids:
        print(f'Starting with recording {r_id}')
        rec_meta = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_recordingMeta.csv')
        t_meta = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_tracksMeta.csv')
        tracks = pd.read_csv(f'{SEARCH_PATH}/data/{r_id}_tracks.csv', engine='pyarrow')
        # Perform some initial cleanup
        if 'trackId' not in t_meta.columns:
            t_meta.rename(columns={'id': 'trackId'}, inplace=True)
            tracks.rename(columns={'id': 'trackId'}, inplace=True)

        tracks = align_x_w_front(t_meta, tracks)
        tracks = add_driving_direction(t_meta, tracks)
        tracks = add_maneuver(t_meta, tracks)
        tracks = add_displacement_feat(rec_meta, t_meta, tracks)
        tracks = add_heading_feat(t_meta, tracks)

        # Determine tr, val, test split (by frames)
        train_frames, val_frames, test_frames = get_frame_split(t_meta.finalFrame.array[-1])

        # Get data and store
        car_ids = list(t_meta[t_meta['class'].isin(['Car', 'Truck'])].trackId)
        np.random.shuffle(car_ids)

        seen_ids = []
        ii = tqdm(range(0, len(car_ids)))
        for i in ii:
            id0 = car_ids[i]
            n_lane_changes = int(t_meta[t_meta.trackId == id0].numLaneChanges)
            if n_lane_changes == 0:
                if id0 in seen_ids:
                    continue
                else:
                    lk_remove_rate_prob = np.random.uniform(0, 1)
                    if lk_remove_rate_prob > 0.005:
                        continue
                skip = fz * 10
            else:
                skip = 12

            driving_dir = int(t_meta[t_meta.trackId == id0].drivingDirection)
            df = tracks[tracks.trackId == id0]
            frames = list(df.frame)
            curr_set = which_set(frames, train_frames, val_frames, test_frames)
            if curr_set is None:
                # If a vehicle is within frames which are overlapping the sets
                continue

            if len(frames) < fz * (INPUT_LENGTH + PRED_HORIZON) + 1:
                continue
            for f in frames[0:-1:skip]:
                fp = f + fz * INPUT_LENGTH
                fT = fp + fz * PRED_HORIZON
                if fT not in frames:
                    break
                x, y, psi, vx, vy, ax, ay, dy, dyr = get_input_features(df, f, fp - 1)
                neighbors = find_neighboring_nodes(tracks, fp - 1, id0, x[-1], y[-1], driving_dir)
                n_SVs = len(neighbors)
                sv_ids = [int(neighbors[n][1]) for n in range(n_SVs)]

                for sv_id in sv_ids:
                    if sv_id not in seen_ids:
                        seen_ids.append(sv_id)

                euc_dist = [int(neighbors[n][0]) for n in range(n_SVs)]
                v_ids = [id0, *sv_ids]
                v_class = get_meta_property(t_meta, v_ids, prop='class')
                v_man = get_maneuver(tracks, fp - 1, v_ids, prop='maneuver')
                v_width = get_meta_property(t_meta, v_ids, prop='height')
                v_length = get_meta_property(t_meta, v_ids, prop='width')

                current_maneuver = v_man[0]
                if current_maneuver == 3 and n_lane_changes > 0:  # remove some intermediate LKs
                    if np.random.uniform(0, 1) > 0.3:
                        continue

                x0 = x[0]
                y0 = y[0]

                x = x - x0
                y = y - y0

                meta_info = MetaInfo(r_id, f, [x0, y0], v_ids, v_class, [0, *euc_dist], v_man, v_width, v_length)
                input_array = np.empty((n_SVs + 1, fz * INPUT_LENGTH, N_IN_FEATURES))
                target_array = np.empty((n_SVs + 1, fz * PRED_HORIZON, N_OUT_FEATURES))

                if driving_dir == 1:
                    x *= (-1)
                    vx *= (-1)
                    ax *= (-1)
                else:
                    y *= (-1)
                    vy *= (-1)
                    ay *= (-1)

                input_array[0, :, :] = np.stack((x, y, psi, vx, vy, ax, ay, dy, dyr), axis=1)
                target_array[0, :, :] = get_adjusted_features(df, fp, fT - 1, N_OUT_FEATURES, x0, y0, driving_dir)
                for j, n in enumerate(range(0, n_SVs)):
                    (dist, sv_id) = neighbors[n]
                    input_array[j + 1, :, :] = get_adjusted_features(tracks, f, fp - 1, N_IN_FEATURES, x0, y0,
                                                                     driving_dir, sv_id)
                    target_array[j + 1, :, :] = get_adjusted_features(tracks, fp, fT - 1, N_OUT_FEATURES, x0, y0,
                                                                      driving_dir, sv_id)

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
                torch.save(input_array, f'data/{save_dir}/{curr_set}/observation/dat{current_id}.pt')
                torch.save(input_nan_mask, f'data/{save_dir}/{curr_set}/observation/nan_mask{current_id}.pt')
                torch.save(input_edge_index, f'data/{save_dir}/{curr_set}/observation/edge_idx{current_id}.pt')
                torch.save(input_edge_feat, f'data/{save_dir}/{curr_set}/observation/edge_feat{current_id}.pt')

                torch.save(target_array, f'data/{save_dir}/{curr_set}/target/dat{current_id}.pt')
                torch.save(target_real_mask, f'data/{save_dir}/{curr_set}/target/real_mask{current_id}.pt')
                torch.save(target_edge_index, f'data/{save_dir}/{curr_set}/target/edge_idx{current_id}.pt')
                torch.save(target_edge_feat, f'data/{save_dir}/{curr_set}/target/edge_feat{current_id}.pt')
                torch.save(inference_target_edge_index,
                           f'data/{save_dir}/{curr_set}/target/full_edge_idx{current_id}.pt')
                torch.save(meta_info, f'data/{save_dir}/{curr_set}/meta/dat{current_id}.pt')

                s_dict[curr_set]['ids'].append(current_id)
                s_dict[curr_set]['id'] += 1
    torch.save(s_dict['training']['ids'], f'data/{save_dir}/training/ids.pt')
    torch.save(s_dict['validation']['ids'], f'data/{save_dir}/validation/ids.pt')
    torch.save(s_dict['testing']['ids'], f'data/{save_dir}/testing/ids.pt')
