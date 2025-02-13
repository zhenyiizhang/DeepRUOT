import torch, torch.nn as nn

class velocityNet(nn.Module):
    # input x, t to get v= dx/dt
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation='Tanh'):
        super().__init__()
        Layers = [in_out_dim+1]
        for i in range(n_hiddens):
            Layers.append(hidden_dim)
        Layers.append(in_out_dim)
        
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        

        self.net = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(Layers[i], Layers[i + 1]),
                self.activation,
            )
                for i in range(len(Layers) - 2)
            ]
        )
        self.out = nn.Linear(Layers[-2], Layers[-1])

    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        #print(num)
        t = t.expand(num, 1)  
        #print(t)
        state  = torch.cat((t,x),dim=1)
        #print(state)
        
        ii = 0
        for layer in self.net:
            if ii == 0:
                x = layer(state)
            else:
                x = layer(x)
            ii =ii+1
        x = self.out(x)
        return x

class growthNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  
        state  = torch.cat((t,x),dim=1)
        return self.net(state)
        #return torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)

class scoreNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1) 
        state  = torch.cat((t,x),dim=1)
        return self.net(state)


class dediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)  
        state  = torch.cat((t,x),dim=1)
        return self.net(state)

class indediffusionNet(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):
        # x is N*2
        num = x.shape[0]
        t = t.expand(num, 1)
        #state  = torch.cat((t,x),dim=1)
        return self.net(t)

class FNet(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, n_hiddens, activation):
        super(FNet, self).__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.v_net = velocityNet(in_out_dim, hidden_dim, n_hiddens, activation)  # v = dx/dt
        self.g_net = growthNet(in_out_dim, hidden_dim, activation)  # g
        self.s_net = scoreNet(in_out_dim, hidden_dim, activation)  # s = log rho
        self.d_net = indediffusionNet(in_out_dim, hidden_dim, activation)  # d = sigma(t)

    def forward(self, t, z):
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            t.requires_grad_(True)

            v = self.v_net(t, z).float()
            g = self.g_net(t, z).float()
            s = self.s_net(t, z).float()
            d = self.d_net(t, z).float()

        return v, g, s, d

class ODEFunc2(nn.Module):
    def __init__(self, f_net):
        super(ODEFunc2, self).__init__()
        self.f_net = f_net

    def forward(self, t, state):
        z, _= state
        v, g, _, _ = self.f_net(t, z)
        
        dz_dt = v
        dlnw_dt = g
        #w = torch.exp(lnw)
        #dm_dt = (torch.norm(v, p=2,dim=1).unsqueeze(1) + g**2) * w
        
        return dz_dt.float(), dlnw_dt.float()


class ODEFunc(nn.Module):
    def __init__(self, v_net):
        super(ODEFunc, self).__init__()
        self.v_net = v_net

    def forward(self, t, z):
        dz_dt = self.v_net(t, z)
        return dz_dt.float()

# %%
class scoreNet2(nn.Module):
    # input x, t to get g
    def __init__(self, in_out_dim, hidden_dim, activation='Tanh'):
        super().__init__()
        if activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU()

        self.net = nn.Sequential(
            nn.Linear(in_out_dim+1, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,hidden_dim),
            self.activation,
            nn.Linear(hidden_dim,1))
    def forward(self, t, x):

        state  = torch.cat((t,x),dim=1)
        return self.net(state)
    
    def compute_gradient(self, t, x):
        x = x.requires_grad_(True)
        output = self.forward(t, x)
        gradient = torch.autograd.grad(outputs=output, inputs=x,
                                       grad_outputs=torch.ones_like(output),
                                       create_graph=True)[0]
        return gradient

# %%
class ODEFunc3(nn.Module):
    def __init__(self, f_net,sf2m_score_model,sigma):
        super(ODEFunc3, self).__init__()
        self.f_net = f_net
        self.sf2m_score_model = sf2m_score_model
        self.sigma=sigma

    def forward(self, t, state):
        z, lnw, m = state
        z=z.requires_grad_(True)
        lnw.requires_grad_(True)
        m.requires_grad_(True)
        t.requires_grad_(True)


        v, g, _, _ = self.f_net(t, z)
        v.requires_grad_(True)
        g.requires_grad_(True)
        #s.requires_grad_(True)
        time=t.expand(z.shape[0],1)
        time.requires_grad_(True)
        s=self.sf2m_score_model(time,z)
\
 
        
        dz_dt = v
        dlnw_dt = g
        w = torch.exp(lnw)
        z=z.requires_grad_(True)
        #grad_s = torch.autograd.grad(s.sum(), z)[0].requires_grad_() #need to change 
        grad_s = torch.autograd.grad(outputs=s, inputs=z,grad_outputs=torch.ones_like(s),create_graph=True)[0]

        norm_grad_s = torch.norm(grad_s, dim=1).unsqueeze(1).requires_grad_(True)
        #norm_grad_s=0
        #print(norm_grad_s.shape)
        
        
        dm_dt = (torch.norm(v, p=2, dim=1).unsqueeze(1) / (2) + 
                 (norm_grad_s ** 2) / 2 -
                 (1 / 2 * self.sigma ** 2 *g + s* g) + g ** 2) * w
        #print(dm_dt.shape)
        return dz_dt, dlnw_dt, dm_dt
