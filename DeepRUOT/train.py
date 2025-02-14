__all__ = ['train']

import os, sys, json, math, itertools
import pandas as pd, numpy as np
import warnings

# from tqdm import tqdm
from tqdm.notebook import tqdm

import torch

from .utils import sample, generate_steps
from .losses import MMD_loss, OT_loss1, OT_loss2, Density_loss, Local_density_loss
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
from DeepRUOT.models import velocityNet, growthNet, scoreNet, dediffusionNet, indediffusionNet, FNet, ODEFunc2, ODEFunc3
from DeepRUOT.utils import group_extract, sample, to_np, generate_steps, cal_mass_loss, parser, _valid_criterions


def train_un1(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,

    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    device=None,
    logger=None,
    use_pinn=False,

    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=0.1,
    lambda_mass=1,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    best_model_path=None,
):

    # Create the indicies for the steps that should be used
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()

    # Send model to cuda and specify it as training mode
    if use_cuda:
        model = model.cuda()
    
    model.train()
    model.to(device)
    step=0
    print('begin local loss')
    # Initialize the minimum Otloss with a very high value
    min_ot_loss = float('inf')

# Specify the path to save the best model
    #best_model_path = './best_scRNA_velo_new_sinkhorn'  

    for batch in tqdm(range(n_batches)):

        
        # apply local loss
        if local_loss and not global_loss:
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (initial_size)).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            data_t0.to(device)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] # TODO: Currently does not work if hold_out='random'. Do to_ignore before. 
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out: # TODO: This `if` can be deleted since the groups does not include the ho timepoint anymore
                    continue                              # i.e. it is always False. 
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=size1, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                data_t1.to(device)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])
                time.to(device)
                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)
                # prediction

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)
                # loss between prediction and sample t1

                relative_mass_now = relative_mass[i_mass]
                #m0 = torch.zeros_like(lnw0).to(device)  
                data_t0=data_t0.to(device)
                data_t1=data_t1.to(device)
                initial_state_energy = (data_t0, lnw0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                
                x_t, lnw_t=odeint(ODEFunc2(model),initial_state_energy,t,options=dict(step_size=0.01),method='euler')
                lnw_t_last = lnw_t[-1]
                mu = torch.exp(lnw_t_last)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
               
                loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
                i_mass=i_mass+1
             
                local_mass_loss = cal_mass_loss(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0])
                lnw0=lnw_t_last.detach()
                data_t0=x_t[-1].detach()
            
                print('Otloss')
                print(loss_ot)
                print('mass loss')
                print(local_mass_loss)
                loss=(lambda_ot*loss_ot+lambda_mass*local_mass_loss)


                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
               # Detach the loss from the computation graph and get its scalar value
            current_ot_loss = loss_ot.item()
            
            # Check if the current Otloss is the new minimum
            if current_ot_loss < min_ot_loss:
                min_ot_loss = current_ot_loss
                # Save the model's state_dict
                torch.save(model.state_dict(), best_model_path)
                print(f'New minimum otloss found: {min_ot_loss}. Model saved.')
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses


#%%
from .utils import density1, trace_df_dz
def train_all(
    model, df, groups, optimizer, n_batches=20, 
    criterion=MMD_loss(),
    use_cuda=False,

    sample_size=(100, ),
    sample_with_replacement=False,

    local_loss=True,
    global_loss=False,

    hold_one_out=False,
    hold_out='random',
    apply_losses_in_time=True,

    top_k = 5,
    hinge_value = 0.01,
    use_density_loss=False,
    use_local_density=False,

    lambda_density = 1.0,
    datatime0=None,
    device=None,
    autoencoder=None, 
    use_emb=False,
    use_gae=False,

    use_gaussian:bool=False, 
    add_noise:bool=False, 
    noise_scale:float=0.1,
    
    logger=None,
    use_pinn=False,
    sf2m_score_model=None,
    use_penalty=True,
    lambda_energy=0.01,
    lambda_pinn=1,
    lambda_ot=10,
    relative_mass=None,
    initial_size=None,
    reverse:bool = False,
    sigmaa=0.1,
    lambda_initial=None,
):

    if autoencoder is None and (use_emb or use_gae):
        use_emb = False
        use_gae = False
        warnings.warn('\'autoencoder\' is \'None\', but \'use_emb\' or \'use_gae\' is True, both will be set to False.')

    noise_fn = torch.randn if use_gaussian else torch.rand
    def noise(data):
        return noise_fn(*data.shape).cuda() if use_cuda else noise_fn(*data.shape)
    # Create the indicies for the steps that should be used
    steps = generate_steps(groups)

    if reverse:
        groups = groups[::-1]
        steps = generate_steps(groups)

    
    # Storage variables for losses
    batch_losses = []
    globe_losses = []
    if hold_one_out and hold_out in groups:
        groups_ho = [g for g in groups if g != hold_out]
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in generate_steps(groups_ho) if hold_out not in [t0, t1]}
    else:
        local_losses = {f'{t0}:{t1}':[] for (t0, t1) in steps}
        
    density_fn = Density_loss(hinge_value) # if not use_local_density else Local_density_loss()

    # Send model to cuda and specify it as training mode
    if use_cuda:
        model = model.cuda()
    
    model.train()
    step=0
    print('begin local loss')


    for batch in tqdm(range(n_batches)):

        
        # apply local loss
        if local_loss and not global_loss:
            size0=(df[df['samples']==0].values.shape[0],)
            data_0 = sample(df, 0, size=size0, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            P0=0
            num=data_0.shape[0]
            time=torch.tensor([0.0]).to(device)
            time=time.expand(num,1)
            data_0=data_0.to(device)
            sf2m_score_model=sf2m_score_model.to(device)
            s2=sf2m_score_model(time, data_0) 
            data_0.requires_grad_(True)
  
            
            i_mass=1
            lnw0 = torch.log(torch.ones(sample_size[0],1) / (initial_size)).to(device)  
            m0 = (torch.zeros(sample_size[0],1) / (initial_size)).to(device)
            data_t0 = sample(df, 0, size=sample_size, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
            # for storing the local loss with calling `.item()` so `loss.backward()` can still be used

            batch_loss = []
            if hold_one_out:
                groups = [g for g in groups if g != hold_out] # TODO: Currently does not work if hold_out='random'. Do to_ignore before. 
                steps = generate_steps(groups)
            for step_idx, (t0, t1) in enumerate(steps):  
                if hold_out in [t0, t1] and hold_one_out: # TODO: This `if` can be deleted since the groups does not include the ho timepoint anymore
                    continue                              # i.e. it is always False. 
                optimizer.zero_grad()
                
                #sampling, predicting, and evaluating the loss.
                # sample data
                size1=(df[df['samples']==t1].values.shape[0],)
                data_t1 = sample(df, t1, size=size1, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                time = torch.Tensor([t0, t1]).cuda() if use_cuda else torch.Tensor([t0, t1])

                if add_noise:
                    data_t0 += noise(data_t0) * noise_scale
                    data_t1 += noise(data_t1) * noise_scale
                if autoencoder is not None and use_gae:
                    data_t0 = autoencoder.encoder(data_t0)
                    data_t1 = autoencoder.encoder(data_t1)
                # prediction

                if autoencoder is not None and use_emb:        
                    data_tp, data_t1 = autoencoder.encoder(data_tp), autoencoder.encoder(data_t1)

                relative_mass_now = relative_mass[i_mass]
                data_t0=data_t0.to(device)
                lnw0=lnw0.to(device)
                m0=m0.to(device)
                model=model.to(device)
                initial_state_energy = (data_t0, lnw0,m0)
                t=time.to(device)
                t.requires_grad=True
                data_t0.requires_grad=True
                lnw0.requires_grad=True
                m0.requires_grad=True
                x_t, lnw_t,m_t=odeint2(ODEFunc3(model,sf2m_score_model,sigmaa),initial_state_energy,t,options=dict(step_size=0.1),method='euler')
                lnw_t_last = lnw_t[-1]
                m_t_last=m_t[-1]
                mu = torch.exp(lnw_t_last)
                mu = mu / mu.sum()
                nu = torch.ones(data_t1.shape[0],1)
                nu = nu / nu.sum()
                mu = mu.squeeze(1)
                nu=nu.squeeze(1)
                data_t1=data_t1.to(device)
                loss_ot = criterion(x_t[-1], data_t1, mu, nu,device=device)
   
                i_mass=i_mass+1
                local_mass_loss = cal_mass_loss(data_t1, x_t[-1], lnw_t_last, relative_mass_now, sample_size[0])
                m0=m_t_last.clone().detach()
                lnw0=lnw_t_last.clone().detach()
                data_t0=x_t[-1].clone().detach()
            
                print('Otloss')
                print(loss_ot)
                print('mass loss')
                print(local_mass_loss)
                print('energy loss')
                print(m_t_last.mean())
                loss_ot=loss_ot.to(device)
                loss=(1*loss_ot+1000*local_mass_loss+m_t_last.mean())
                


                if use_density_loss:                
                    density_loss = density_fn(data_t0, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    print('density loss')
                    print(density_loss)

                if use_pinn:
                    P1=0
                    nnum=data_t1.shape[0]
                    ttime=time[1].to(device)
                    data_t1=data_t1.to(device)

                    vv, gg, _, _ = model(ttime, data_t1)
                    ttime=ttime.expand(nnum,1)

                    ss=sf2m_score_model(ttime, data_t1)

                    rrho = torch.exp(ss*2/sigmaa**2) 
                    rrho_t = torch.autograd.grad(outputs=rrho, inputs=ttime, grad_outputs=torch.ones_like(rrho),create_graph=True)[0]

                    vv_rho = vv * rrho
                    ddiv_v_rho = trace_df_dz(vv_rho, data_t1).unsqueeze(1)
                    ppinn_loss = torch.abs(rrho_t + ddiv_v_rho - gg * rrho)
                    pppinn_loss = ppinn_loss

                    mean_pppinn_loss = torch.mean(pppinn_loss)

                    size0=(df[df['samples']==0].values.shape[0],)
                    data_0 = sample(df, 0, size=size0, replace=sample_with_replacement, to_torch=True, use_cuda=use_cuda)
                    P0=0
                    num=data_0.shape[0]
                    time=torch.tensor([0.0]).to(device)
                    time=time.expand(num,1)
                    data_0=data_0.to(device)
                    sf2m_score_model=sf2m_score_model.to(device)

                    s2=sf2m_score_model(time, data_0) 

                    data_0.requires_grad_(True)
                    density_values = density1(data_0,datatime0,device)
                    loss2=0
                    loss2=torch.mean((torch.exp(s2*2/sigmaa**2)-density_values)**2)
  
                    if loss2 < 1e-6:
                        lambda_initial=0
                        loss2=0
                    if mean_pppinn_loss < 1e-6:
                        lambda_pinn=0
                        mean_pppinn_loss=0
                    print('pinloss')
                    print(mean_pppinn_loss)
                    
                    loss += lambda_pinn * mean_pppinn_loss+lambda_initial*loss2


                # apply local loss as we calculate it
                if apply_losses_in_time and local_loss:
                    loss.backward()
                    optimizer.step()
                    model.norm=[]
                # save loss in storage variables 
                local_losses[f'{t0}:{t1}'].append(loss.item())
                batch_loss.append(loss)
        
        
            # convert the local losses into a tensor of len(steps)
            batch_loss = torch.Tensor(batch_loss).float()
            if use_cuda:
                batch_loss = batch_loss.cuda()
            
            if not apply_losses_in_time:
                batch_loss.backward()
                optimizer.step()

            # store average / sum of local losses for training
            ave_local_loss = torch.mean(batch_loss)
            sum_local_loss = torch.sum(batch_loss)            
            batch_losses.append(ave_local_loss.item())
        
                     
    print_loss = globe_losses if global_loss else batch_losses 
    if logger is None:      
        tqdm.write(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    else:
        logger.info(f'Train loss: {np.round(np.mean(print_loss), 5)}')
    return local_losses, batch_losses, globe_losses