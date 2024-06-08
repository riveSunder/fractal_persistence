import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdamAutomaton():
    
    def __init__(self, **kwargs):
        
        alpha = kwargs["alpha"] if "alpha" in kwargs.keys() else 1e-1
        beta_1 = kwargs["beta_1"] if "beta_1" in kwargs.keys() else 1e-3
        beta_2 = kwargs["beta_2"] if "beta_2" in kwargs.keys() else 1e-4
        epsilon = kwargs["epsilon"] if "epsilon" in kwargs.keys() else 1e-8
        
        self.set_alpha(alpha)
        self.set_beta_1(beta_1)
        self.set_beta_2(beta_2)
        self.set_epsilon(epsilon)
        
        self.init_growth()
    
    def init_growth(self, mu=0.167, sigma=0.013):
    
        def growth(x):
            
            return 2*torch.exp(-(x-mu)**2/(2*sigma**2))-1
        
        self.growth = growth
    
    def __call__(self, grid, n):
        
        # cell states
        a = grid[:,0:1,:,:]
        
        # neighborhoods
        
        # first and second moments
        m_0 = grid[:,1:2,:,:]
        v_0 = grid[:,2:3,:,:]
        
        # 'gradient', 
        g = self.growth(n[:,0:1,:,:])
        
        m = (self.beta_1) * m_0 + (1-self.beta_1)  * g
        v = (self.beta_2) * v_0 + (1-self.beta_2)  * g**2
        
        # adam update for cell states
        new_a = a + self.alpha * (m / (torch.sqrt(v) + self.epsilon))
        
        new_grid = torch.zeros_like(grid)
        # assign cell states and moments
        new_grid[:,0:1,:,:] = new_a.unsqueeze(0).unsqueeze(0)
        new_grid[:,1:2,:,:] = m.unsqueeze(0).unsqueeze(0)
        new_grid[:,2:3,:,:] = v.unsqueeze(0).unsqueeze(0)
        
        return torch.clamp(new_grid, 0, 1.0)
    
    def set_alpha(self, new_alpha):
        self.alpha = 1.0 * new_alpha
        
    def get_alpha(self):
        return 1.0 * self.alpha
        
    def set_beta_1(self, new_beta_1):
        self.beta_1 = 1.0 * new_beta_1
        
    def get_beta_1(self):
        return 1.0 * self.beta_1

    def set_beta_2(self, new_beta_2):
        self.beta_2 = 1.0 * new_beta_2
        
    def get_beta_2(self):
        return 1.0 * self.beta_2
        
    def set_epsilon(self, new_epsilon):
        self.epsilon = 1.0 * new_epsilon
        
    def get_epsilon(self):
        return 1.0 * self.epsilon
