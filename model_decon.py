import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import FCNet, gaussianKL, gaussianNLL, recons_loss

class ModelDecon(nn.Module):
    def __init__(self, opts):
        super(ModelDecon, self).__init__()
        self.opts = opts
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define networks
        self.encoder = FCNet(opts['x_dim'], opts['qzgx_net_layers'], opts['qzgx_net_outlayers'])
        self.xarencoder = FCNet(opts['qzgx_net_outlayers'][-1][0]+opts['qzga_net_outlayers'][-1][0]+ opts['qzgr_net_outlayers'][-1][0], opts['qzgxar_net_layers'], opts['qzgxar_net_outlayers'])
        self.q_a_net = FCNet(opts['a_dim'], opts['qzga_net_layers'], opts['qzga_net_outlayers']) 
        self.q_a_net_mu = FCNet(opts['x_dim'], opts['qzga_net_layers'], [(opts['a_dim'],None)])
        self.q_a_net_sigma = FCNet(opts['x_dim'], opts['qzga_net_layers'], [(opts['a_dim'],nn.Softplus())])
        self.q_aea_net = FCNet(opts['qzga_net_outlayers'][-1][0], opts['qzga_net_layers'], opts['qzga_net_outlayers'])
        self.q_r_net = FCNet(opts['r_dim'], opts['qzgr_net_layers'], opts['qzgr_net_outlayers'])
        self.q_r_g_x_a_mu = FCNet(opts['qzgx_net_outlayers'][-1][0]+ opts['qzga_net_outlayers'][-1][0], opts['qzgr_net_layers'], opts['qrgx_net_outlayers'])
        self.q_r_g_x_a_sigma = FCNet(opts['qzgx_net_outlayers'][-1][0]+opts['qzga_net_outlayers'][-1][0], opts['qzgr_net_layers'], opts['qrgx_net_outlayers_sigma'])
        self.p_x_net_mu = FCNet(opts['z_dim'], opts['pxgz_net_layers'], opts['pxgz_net_outlayers'])
        self.p_x_net_sigma = FCNet(opts['z_dim'], opts['pxgz_net_layers'], opts['pxgz_net_outlayers_sigma'])
        self.p_a_net_mu = FCNet(opts['z_dim'], opts['pagz_net_layers'], opts['pagz_net_outlayers'])
        self.p_a_net_sigma = FCNet(opts['z_dim'], opts['pagz_net_layers'], opts['pagz_net_outlayers_sigma'])
        self.p_r_net_mu = FCNet(opts['z_dim'] + opts['a_dim'], opts['prgza_net_layers'], opts['prgza_net_outlayers'])
        self.p_r_net_sigma = FCNet(opts['z_dim'] + opts['a_dim'], opts['prgza_net_layers'], opts['prgza_net_outlayers_sigma'])
        self.p_z_net = FCNet(opts['z_dim'], opts['pzgz_net_layers'], opts['pzgz_net_outlayers'])
        self.p_a_net = FCNet(opts['a_dim'], opts['pzga_net_layers'], opts['pzga_net_outlayers'])     
        self.p_zeza_net = FCNet(opts['pzgz_net_outlayers'][-1][0]+opts['pzga_net_outlayers'][-1][0], opts['pzgz_net_layers'], opts['pzgz_net_outlayers'])
        self.lstm_net = nn.LSTM(opts['qzgxar_net_outlayers'][-1][0], opts['lstm_dim'], len(opts['lstm_net_layers']), batch_first=True)
        
        self.st_net = FCNet(opts['z_dim'] + opts['lstm_dim'], opts['st_net_layers'], opts['st_net_outlayers'])
        self.h_mu_net = FCNet(opts['st_net_outlayers'][-1][0], opts['h_mu_net_layers'], opts['h_mu_net_outlayers'])
        self.h_sigma_net = FCNet(opts['st_net_outlayers'][-1][0], opts['h_sigma_net_layers'], opts['h_sigma_net_outlayers'])

    def st_approx(self, prev, element):
        z_prev, mu_prev, cov_prev = prev
        h, a_fea = element

        combined = torch.cat([z_prev, h], dim=-1)
        h_next = self.st_net(combined)

        mu = self.h_mu_net(h_next)
        sigma = torch.nn.functional.softplus(self.h_sigma_net(h_next))

        eps = torch.randn(mu.size(), device=self.device)
        z = mu + eps * torch.sqrt(1e-8 + sigma)

        return z, mu, sigma

    def p_z(self):
        mu_z = torch.zeros(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        cov_z = torch.ones(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        eps = torch.randn(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        z = mu_z + eps * torch.sqrt(1e-8 + cov_z)
        return z

    def p_x_g_z(self, z):
        mu = self.p_x_net_mu(z)
        sigma =  self.p_x_net_sigma(z)

        return mu, sigma

    def p_a_g_z(self, z):
        mu = self.p_a_net_mu(z)
        sigma = self.p_a_net_sigma(z)
        return mu, sigma

    def p_r_g_z_a(self, z, a):
        zau_fea = torch.cat([z, a], dim=-1)
        mu = self.p_r_net_mu(zau_fea)
        sigma = self.p_r_net_sigma(zau_fea)
        return mu, sigma

    def p_z_g_z_a(self, z, a):
        z_fea = self.p_z_net(z)
        a_fea = self.p_a_net(a)

        if len(z.shape) > 2:
            az_fea = torch.cat([z_fea, a_fea], 2)
        else:
            az_fea = torch.cat([z_fea, a_fea], 1)

        h_az_fea = self.p_zeza_net(az_fea)
        h_mu = self.h_mu_net(h_az_fea)

        if self.opts['gated']:
            hg_az_fea = self.q_aea_net(az_fea)
            gate = self.st_net(hg_az_fea)
            mu = gate * h_mu + (1 - gate) * self.h_sigma_net(az_fea)
        else:
            mu = h_mu

        sigma = self.h_sigma_net(h_az_fea)

        return mu, sigma

    def q_z_g_z_x_a_r(self, x_seq, a_seq, r_seq, mask=None):
        x_fea = self.encoder(x_seq)
        a_fea = self.q_a_net(a_seq)
        r_fea = self.q_r_net(r_seq)
        concat_xar = torch.cat([x_fea, a_fea, r_fea], dim=2)
        xar_fea = self.xarencoder(concat_xar)
        h_r, _ = self.lstm_net(xar_fea)
        if self.opts['inference_model_type'] == 'LR':
            h_l, _ = self.lstm_net(xar_fea)
            h = (h_r + h_l) / 2
        else:
            h = h_r

        z_0 = torch.zeros(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        mu_0 = torch.zeros(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        cov_0 = torch.ones(self.opts['batch_size'], self.opts['z_dim'], device=self.device)
        h = h.transpose(0, 1)  
        h = h.unsqueeze(1)
        a_fea = self.q_aea_net(a_fea).transpose(0, 1)
        a_fea = torch.cat([torch.ones(1, self.opts['batch_size'], a_fea.size(2), device=self.device), a_fea[:-1]], 0).unsqueeze(1)
        ha_concat = torch.cat([h, a_fea], 1)
        ha_split = ha_concat.split(1, 0)
        ha_list = [split.squeeze(0) for split in ha_split]

        output_q = []
        z, mu, cov = z_0, mu_0, cov_0
        for element in ha_list:
            z, mu, cov = self.st_approx((z, mu, cov), element)
            output_q.append((z, mu, cov))

        z = torch.stack([item[0] for item in output_q]).transpose(0, 1)
        mu = torch.stack([item[1] for item in output_q]).transpose(0, 1)
        cov = torch.stack([item[2] for item in output_q]).transpose(0, 1)

        if x_seq.dim() == 2:
            z = z.squeeze(1)
            mu = mu.squeeze(1)
            cov = cov.squeeze(1)
        return z, mu, cov

    def q_a_g_x(self, x):
        return self.q_a_net_mu(x),  self.q_a_net_sigma(x)

    def q_r_g_x_a(self, x, a):
        x_fea = self.encoder(x)
        a_fea = self.q_a_net(a)
        ax_fea = torch.cat([x_fea, a_fea], dim=-1)
        return self.q_r_g_x_a_mu(ax_fea), self.q_r_g_x_a_sigma(ax_fea)

    def neg_elbo(self, x_seq, a_seq, r_seq, anneal=1.0, mask=None):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)
        eps = torch.randn((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']), device=self.device)
        z_q_samples = mu_q + eps * torch.sqrt(1e-8 + cov_q)
        mu_p, cov_p = self.p_z_g_z_a(z_q_samples, a_seq)

        mu_prior = torch.cat([torch.zeros(self.opts['batch_size'], 1, self.opts['z_dim'], device=self.device), mu_p[:, :-1]], 1)
        cov_prior = torch.cat([torch.ones(self.opts['batch_size'], 1, self.opts['z_dim'], device=self.device), cov_p[:, :-1]], 1)
        kl_divergence = gaussianKL(mu_prior, cov_prior, mu_q, cov_q, mask)

        mu_pxgz, cov_pxgz = self.p_x_g_z(z_q_samples)
        mu_pagz, cov_pagz = self.p_a_g_z(z_q_samples)
        mu_prgza, cov_prgza = self.p_r_g_z_a(z_q_samples, a_seq)
        mu_qagx, cov_qagx = self.q_a_g_x(x_seq)
        mu_qrgxa, cov_qrgxa = self.q_r_g_x_a(x_seq, a_seq)

        nll_pxgz = gaussianNLL(x_seq, mu_pxgz, cov_pxgz, mask)
        nll_pagz = gaussianNLL(a_seq, mu_pagz, cov_pagz, mask)
        nll_prgza = gaussianNLL(r_seq, mu_prgza, cov_prgza, mask)
        nll_qagx = gaussianNLL(a_seq, mu_qagx, cov_qagx, mask)
        nll_qrgxa = gaussianNLL(r_seq, mu_qrgxa, cov_qrgxa, mask)

        nll = nll_pxgz + nll_pagz + nll_prgza + anneal * kl_divergence + nll_qagx + nll_qrgxa
        return nll, kl_divergence

    def gen_xar_seq_g_z(self, z_0):
        z_0_shape = z_0.size()
        if len(z_0_shape) > 2:
            z_0 = z_0.view(z_0_shape[0], z_0_shape[2])

        output_xar = []
        for _ in range(self.opts['nsteps']):
            z, x_prev, a_prev, r_prev = self.gen_st_approx((z_0, None), None)
            output_xar.append((x_prev, a_prev, r_prev))

        return torch.stack([item[0] for item in output_xar]).transpose(0, 1)

    def recons_xar_seq_g_xar_seq(self, x_seq, a_seq, r_seq, mask):
        z_q, mu_q, cov_q = self.q_z_g_z_x_a_r(x_seq, a_seq, r_seq, mask)
        eps = torch.randn((self.opts['batch_size'], self.opts['nsteps'], self.opts['z_dim']), device=self.device)
        z_q_samples = mu_q + eps * torch.sqrt(1e-8 + cov_q)

        mu_pxgz, cov_pxgz = self.p_x_g_z(z_q_samples)
        mu_pagz, cov_pagz = self.p_a_g_z(z_q_samples)
        mu_prgza, cov_prgza = self.p_r_g_z_a(z_q_samples, a_seq)

        return mu_pxgz, mu_pagz, mu_prgza

    def gen_z_g_x(self, x):
        a, _ = self.q_a_g_x(x)
        r, _ = self.q_r_g_x_a(x, a)
        _, z, _ = self.q_z_g_z_x_a_r(x, a, r)
        return z

    def train_model(self, data):
        batch_num = data.train_num // self.opts['batch_size']
        counter = self.opts.get('counter_start', 0)

        train_nll = []
        train_kl = []
        train_x_loss = []
        train_a_loss = []
        train_r_loss = []
        validation_nll = []
        validation_kl = []
        validation_x_loss = []
        validation_a_loss = []
        validation_r_loss = []

        optimizer = optim.Adam(self.parameters(), lr=self.opts['lr'])

        for epoch in range(self.opts['epoch_start'], self.opts['epoch_start'] + self.opts['epoch_num']):
            if epoch > self.opts['epoch_start'] and epoch % self.opts['save_every_epoch'] == 0:
                torch.save(self.state_dict(), os.path.join('/media/disk2/dyw/cvae/DRL/vaedtr/saved_model/', 'model_checkpoints', f'model_decon_{counter}.pth'))

            ids_perm = np.random.permutation(data.train_num)

            for itr in range(batch_num):
                start_time = time.time()

                batch_ids = ids_perm[itr * self.opts['batch_size']:(itr + 1) * self.opts['batch_size']]
                x_batch = torch.tensor(data.x_train[batch_ids], dtype=torch.float32).to(self.device)
                a_batch = torch.tensor(data.a_train[batch_ids], dtype=torch.float32).to(self.device)
                r_batch = torch.tensor(data.r_train[batch_ids], dtype=torch.float32).to(self.device)
                mask_batch = torch.tensor(data.mask_train[batch_ids], dtype=torch.float32).to(self.device)

                optimizer.zero_grad()
                nll, kl_dist = self.neg_elbo(x_batch, a_batch, r_batch, anneal=self.opts['anneal'], mask=mask_batch)
                x_recons_tr, a_recons_tr, r_recons_tr = self.recons_xar_seq_g_xar_seq(x_batch, a_batch, r_batch, mask_batch)
                x_tr_loss = recons_loss(self.opts['recons_cost'], x_batch, x_recons_tr)
                a_tr_loss = recons_loss(self.opts['recons_cost'], a_batch, a_recons_tr)
                r_tr_loss = recons_loss(self.opts['recons_cost'], r_batch, r_recons_tr)
                
                # Combine all losses
                loss = nll + kl_dist + x_tr_loss + a_tr_loss + r_tr_loss
                loss.backward()
                optimizer.step()

                train_nll.append(nll.item())
                train_kl.append(kl_dist.item())
                train_x_loss.append(x_tr_loss.item())
                train_a_loss.append(a_tr_loss.item())
                train_r_loss.append(r_tr_loss.item())

                elapsed_time = time.time() - start_time

                print(f'Epoch: {epoch}, Iteration: {itr}, Train NLL: {nll.item():.6f}, Train KL: {kl_dist.item():.6f}, '
                    f'Train X Loss: {x_tr_loss.item():.6f}, Train A Loss: {a_tr_loss.item():.6f}, Train R Loss: {r_tr_loss.item():.6f}, '
                    f'Elapsed Time: {elapsed_time:.2f} seconds')

                counter += 1

                if counter % self.opts['plot_every'] == 0:
                    with torch.no_grad():
                        val_ids = np.random.choice(data.validation_num, self.opts['batch_size'], replace=False)
                        x_val = torch.tensor(data.x_validation[val_ids], dtype=torch.float32).to(self.device)
                        a_val = torch.tensor(data.a_validation[val_ids], dtype=torch.float32).to(self.device)
                        r_val = torch.tensor(data.r_validation[val_ids], dtype=torch.float32).to(self.device)
                        mask_val = torch.tensor(data.mask_validation[val_ids], dtype=torch.float32).to(self.device)

                        nll_val, kl_val = self.neg_elbo(x_val, a_val, r_val,self.opts['anneal'], mask_val)
                        x_recons_val, a_recons_val, r_recons_val = self.recons_xar_seq_g_xar_seq(x_val, a_val, r_val, mask_val)
                        x_val_loss = recons_loss(self.opts['recons_cost'], x_val, x_recons_val)
                        a_val_loss = recons_loss(self.opts['recons_cost'], a_val, a_recons_val)
                        r_val_loss = recons_loss(self.opts['recons_cost'], r_val, r_recons_val)

                    validation_nll.append(nll_val.item())
                    validation_kl.append(kl_val.item())
                    validation_x_loss.append(x_val_loss.item())
                    validation_a_loss.append(a_val_loss.item())
                    validation_r_loss.append(r_val_loss.item())

                    print(f'Validation - NLL: {nll_val.item():.6f}, KL: {kl_val.item():.6f}, '
                        f'X Loss: {x_val_loss.item():.6f}, A Loss: {a_val_loss.item():.6f}, R Loss: {r_val_loss.item():.6f}')

                    x_0_sample_value = torch.tensor(data.x_validation[val_ids][:, 0, :], dtype=torch.float32).unsqueeze(1).to(self.device)
                    x_seq_sampling = self.gen_xar_seq_g_z(self.gen_z_g_x(x_0_sample_value))

                    # filename = f'result_plot_epoch_{epoch}_itr_{itr}.png'
                    # save_plots(self.opts, data.x_train[batch_ids], data.x_validation[val_ids], x_recons_tr, x_recons_val,
                    #         train_nll, train_kl, validation_nll, validation_kl, train_x_loss, validation_x_loss,
                    #         train_a_loss, validation_a_loss, train_r_loss, validation_r_loss, x_seq_sampling, filename)

        torch.save(self.state_dict(), os.path.join(self.opts['work_dir'], 'saved_model', f'model_decon_{counter}.pth'))
