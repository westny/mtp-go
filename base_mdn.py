import math
import lightning.pytorch as pl
import torch.linalg
from argparse import ArgumentParser
from models.utils import *
from losses import *

WTA_LOSS = EWTALoss()
NLL_LOSS = NLLMDNLoss()
MODE_LOSS = ModeDist()


class LitEncoderDecoder(pl.LightningModule):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 args: ArgumentParser):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.motion_model = args.motion_model
        self.dataset = args.dataset
        self.max_epochs = args.epochs
        self.learning_rate = args.lr
        self.teacher_force_epochs = self.max_epochs // 2
        self.tf_init_p = args.teacher_forcing
        self.warm_epochs = self.max_epochs // 4
        self.wta_epochs = self.max_epochs // 8
        self.annealing_epochs = int(self.max_epochs * 0.6)
        self.time_update = self.ekf
        self.initial_uncertainty = 1e-5

        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def forward(self, data):
        pass

    def ekf(self, P_t, q_t, next_states, model_input, static_f):
        """Performs the time update of the state covariance estimate
        using standard EKF approach"""
        F_t, F_t_transpose = self.decoder.motion_model.state_transition_matrix(
            next_states, model_input, static_f)
        G_t, G_t_transpose = self.decoder.motion_model.input_transition_matrix(
            next_states, model_input)
        Q_t = G_t @ q_t @ G_t_transpose
        P_t_next = F_t @ P_t @ F_t_transpose + Q_t
        return P_t_next

    def encode_decode(self, data, batch_idx, tf_prob=0.0):
        """
        Run data through encoder and decoder, returns predicted states and Ps
        """
        target_tensor = data.y
        input_tensor = data.x
        tar_edge_index = data.tar_edge_index
        target_length = target_tensor.size(1)
        batch_size = input_tensor.size(0)

        mixtures = self.decoder.mixtures
        n_states = self.decoder.motion_model.n_states
        target = target_tensor[:, :, :n_states]

        P_t = torch.diag_embed(torch.ones(batch_size, mixtures, n_states,
                                          device=input_tensor.device)
                               ) * self.initial_uncertainty  # (batch_size, mixtures, n_states, n_states)

        encoder_output, mixture_coeffs = self.encoder(data)
        decoder_hidden = self.decoder.get_initial_state(encoder_output[:, -1], data)

        decoder_input = torch.zeros(batch_size, mixtures * n_states,
                                    device=input_tensor.device)

        past_state = input_tensor[:, -1:, :n_states]
        past_state = past_state.expand(-1, mixtures, -1)

        real_mask = data.tar_real_mask[..., 0].to(torch.float32)  # 1 where obs exists
        use_teacher_forcing = (tf_prob > 0.) and (torch.FloatTensor(
            1).uniform_(0, 1) < tf_prob)

        dec_static_features = extract_static_features(data, self.motion_model).unsqueeze(1).expand(
            -1, mixtures, -1)  # (B, N_mixtures, 6)

        # Roll out decoding
        pred_states = []
        Ps = []
        for di in range(target_length):
            dec_in = (decoder_input, decoder_hidden, encoder_output,
                      tar_edge_index[di], past_state, dec_static_features)
            next_states, model_input, q_t, decoder_hidden = self.decoder(dec_in)

            #  Update estimated state covariance
            P_t = self.time_update(P_t, q_t, past_state, model_input,
                                   dec_static_features)

            pred_states.append(next_states)
            Ps.append(P_t)

            prediction_det = next_states.detach()
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                ri_mask = real_mask[:, di].view(-1, 1, 1)  # (B,)
                teacher_pred = target[:, di:di + 1, :].expand(
                    -1, mixtures, -1)  # (B, N_mixture, d)

                # Keep prediction when no teacher prediction exists
                past_state = ri_mask * teacher_pred + (1. - ri_mask) * prediction_det
            else:
                past_state = prediction_det

            decoder_input = past_state.view(batch_size, n_states * mixtures)

        # Compute NLL loss
        all_states = torch.stack(pred_states, dim=1)  # (B, N_t, N_mix, d)
        all_Ps = torch.stack(Ps, dim=1)  # (B, N_t, N_mix, d, d)

        return all_states, all_Ps, mixture_coeffs, real_mask, target

    def training_step(self, data, batch_idx):
        # Compute probability of teacher forcing, only > 0
        tf_prob = max(0, self.tf_init_p * ((self.teacher_force_epochs - self.current_epoch)
                                           / self.teacher_force_epochs))

        all_states, all_Ps, mixture_coeffs, dec_mask, target = self.encode_decode(
            data, batch_idx, tf_prob=tf_prob)
        # all_states: (B, N_t, N_mix, d)
        # all_Ps: (B, N_t, N_mix, d, d)
        # mixture_coeffs: (B, N_mix)
        # target: (B, N_t, d)

        if self.current_epoch < self.wta_epochs and self.wta_epochs > 1:
            # Only EWTA loss
            wta_weight = (self.wta_epochs - self.current_epoch) / self.wta_epochs

            #  The number of winners used for the WTA loss decreases with increasing epoch
            n_mixtures = mixture_coeffs.shape[1]
            n_winners = max(min(n_mixtures, int(wta_weight * n_mixtures)), 1)
            loss = WTA_LOSS(all_states, target, dec_mask, n_winners)
        else:
            # Compute NLL loss
            nll_loss = NLL_LOSS(all_states, all_Ps, mixture_coeffs, target, dec_mask)

            if self.current_epoch < self.warm_epochs and self.warm_epochs > 1:
                # WTA + NLL loss
                #  Update the weight by which the warm-up criterion should be multiplied
                warm_weight = (self.warm_epochs - self.current_epoch) / (
                        self.warm_epochs - self.wta_epochs)

                # Compute WTA loss
                wta_loss = WTA_LOSS(all_states, target, dec_mask)
                # Combine losses
                loss = warm_weight * wta_loss + (1. - warm_weight) * nll_loss
            else:
                loss = nll_loss

        batch_size = target.shape[0]
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, data, batch_idx):

        all_states, all_Ps, mixture_coeffs, dec_mask, target = self.encode_decode(
            data, batch_idx)

        # See training step for shapes
        batch_size = target.shape[0]
        ptr = data.ptr[:-1]

        # Use only position part of states during validation
        all_states = all_states[..., :2]
        all_Ps = all_Ps[..., :2, :2]
        target = target[..., :2]

        # Compute NLL loss
        nll_loss = NLL_LOSS(all_states, all_Ps, mixture_coeffs, target, dec_mask)

        # Compute MSE losses
        most_likely_component = torch.argmax(mixture_coeffs, dim=-1)  # (B,)
        most_likely_trajs = all_states[torch.arange(batch_size), :, most_likely_component, :2]  # (B, d)

        n_pred = torch.sum(dec_mask, dim=0)
        norm = torch.linalg.norm(most_likely_trajs - target, dim=-1)

        sv_norm = (norm * dec_mask).sum(0) / n_pred
        ade = sv_norm.mean()
        fde = sv_norm[-1]

        tv_norm = norm[ptr]
        tv_ade = tv_norm.mean()
        tv_fde = tv_norm[:, -1].mean()

        # Monitor spread of mixture components
        mode_loss = MODE_LOSS(most_likely_trajs, all_states[..., :2], dec_mask)

        self.log_dict({
            "val_ade": ade,
            "val_fde": fde,
            "val_nll": nll_loss,

            "val_tv_ade": tv_ade,
            "val_tv_fde": tv_fde,
            "mode_dist": mode_loss,
        }, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return ade

    def test_step(self, data, batch_idx):
        all_states, all_Ps, mixture_coeffs, dec_mask, target = self.encode_decode(
            data, batch_idx)
        # See training step for shapes
        ptr = data.ptr[:-1]
        batch_size = target.shape[0]
        target_length = target.shape[1]

        # Use only position part of states during validation
        all_states = all_states[..., :2]
        all_Ps = all_Ps[..., :2, :2]
        target = target[..., :2]

        # Compute NLL loss
        nll_loss = NLL_LOSS(all_states, all_Ps, mixture_coeffs, target, dec_mask)
        fnll_loss = NLL_LOSS(all_states[:, -1:], all_Ps[:, -1:], mixture_coeffs, target[:, -1:],
                             dec_mask[:, -1:])

        # Compute MSE losses
        most_likely_component = torch.argmax(mixture_coeffs, dim=-1)  # (B,)
        most_likely_trajs = all_states[torch.arange(batch_size), :, most_likely_component, :2]  # (B, d)

        n_pred = torch.sum(dec_mask, dim=0)
        norm = torch.linalg.norm(most_likely_trajs - target, dim=-1)

        masked_norm = norm * dec_mask
        sv_norm = masked_norm.sum(0) / n_pred
        ade = sv_norm.mean()
        fde = sv_norm[-1]

        indexed_errors = torch.cdist(most_likely_trajs, target, p=2)
        path_dist, _ = indexed_errors.min(dim=-1)
        masked_path = path_dist * dec_mask
        path_norm = masked_path.sum(0) / n_pred
        ape = path_norm.mean()

        # SV miss rate
        mr = 0
        nn = 0
        for bi in range(batch_size):
            current_norm = masked_norm[bi]
            idx = torch.nonzero(current_norm == 0)
            i = idx[0] - 1 if idx.nelement() else -1
            if i < 20:  # less than 4s into the future
                # if i < 0:
                # Skip vehicles without target
                continue
            else:
                curr_fde = masked_norm[bi, i]
                if curr_fde > 2.0:
                    mr += 1
            nn += 1
        mr /= nn

        # Compute TV losses
        tv_nll = NLL_LOSS(all_states[ptr], all_Ps[ptr], mixture_coeffs[ptr], target[ptr], dec_mask[ptr])
        tv_fnll = NLL_LOSS(all_states[ptr, -1:], all_Ps[ptr, -1:], mixture_coeffs[ptr],
                           target[ptr, -1:], dec_mask[ptr, -1:])

        tv_norm = norm[ptr]
        tv_ade = tv_norm.mean()
        tv_fde = tv_norm[:, -1].mean()
        tv_mr = torch.count_nonzero(tv_norm[:, -1] >= 2.0) / (len(ptr) - 1)

        tv_ape = path_dist[ptr].mean()

        results = {
            "test_tv_ade": tv_ade,
            "test_tv_fde": tv_fde,
            "test_tv_anll": tv_nll / target_length,
            "test_tv_fnll": tv_fnll,
            "test_tv_apde": tv_ape,
            "tv_mr": tv_mr,

            "test_ade": ade,
            "test_fde": fde,
            "test_anll": nll_loss / target_length,
            "test_fnll": fnll_loss,
            "test_apde": ape,
            "mr": mr
        }

        self.log_dict(results, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5,
                                                      total_iters=self.annealing_epochs)
        return [optimizer], [scheduler]
