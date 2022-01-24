import numpy as np
from scipy.optimize import minimize
from models import *
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")


def _moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class bias_corrected_mine():
    def __init__(self, converge_res_thresh=1.4, converge_len_thresh=2, mi_init1=4):
        # hyperparams
        self.converge_res_thresh = converge_res_thresh
        self.converge_len_thresh = converge_len_thresh
        self.mi_init1 = mi_init1  # the initial guess at the MI, best practice is to try a range of values
        self.mi_init2 = mi_init1/1.5  # can be adjusted


    def loss(self, x):
        a = x[0]
        b = x[1]    
        c = np.abs(x[2])  # force this to be positive,to stop bad fits    
        mi_final = x[3]   

        mi_fit = mi_final*(1-(a * np.exp(-b * self.t)))+c*self.t
        gof = np.mean((mi_fit-self.data_arr[self.t])**2)
        self.fit = gof  # always check goodness of fit to evaluate different mi_inits, etc.!
        return gof


    def converge(self, mi_est, current_iter, min_iters, max_iters, residual=0.5, thresh=5, min_mi=0.1):
        diff = np.abs(np.cumsum(mi_est)/np.arange(len(mi_est))-np.array(mi_est))
        if (diff<residual).sum() > thresh:
            return np.where(diff<residual)[0][-1]
        elif (current_iter>=min_iters and mi_est[-1]<min_mi) or current_iter>=max_iters:
            return -1  # return last bias-corrected estimate if mi is too low or we're at the end of training 
        else:
            return False


    def train(self, data_type1, data_type2, min_iters=4e4, max_iters=4e5, step_size=2e4, hidden_size=700, learn_rate=3.3e4, dynamic_start=True): 
        """
        Trains 20k iters at a time looking for convergence between min_iters and max_iters.
        """
        # inits
        data = np.concatenate([data_type1, data_type2], axis=1)
        d1_size = data_type1.shape[1]
        d2_size = data_type2.shape[1]
        residuals = []
        mine = Mine(input_size=d1_size+d2_size, hidden_size=hidden_size).cuda()
        mine_net_optim_joint = optim.Adam(mine.parameters(), lr=learn_rate)
        res = 0

        mi_est = []  # contains the full trajectory
        raw_mi_obs = []
        x_sub = np.arange(int(step_size),int(max_iters+1),int(step_size))
        for i in x_sub:   # this is basically where I would train
            raw_mi_obs.append(train(data,mine,mine_net_optim_joint, iter_num=int(step_size), log_freq='', n_genes=d1_size))

            self.t = np.arange(i).astype(np.int32)
            data_arr = np.concatenate(raw_mi_obs)  # data_arr is an ever growing array of the output every 20k iterations
            # clipping infinities helps a lot!
            self.data_arr = np.clip(data_arr, np.nanmin(data_arr[data_arr != -np.inf]), np.nanmax(data_arr[data_arr != np.inf]))
            if dynamic_start:
                # start fitting only after a burn in period
                start = _moving_average(self.data_arr, 100)
                if (start>0.025).any():
                    start = np.where(start>0.025)[0][0]
                    self.data_arr = self.data_arr[start:]
                    self.t = np.arange(len(self.data_arr))
                else:
                    mi_est.append(0)
                    continue  # skips the rest of the for loop (won't try to fit) and will end with 0
            if res:
                res = minimize(self.loss, [res.x[0], res.x[1]+(np.random.random()-0.5)*res.x[1]/8, 
                                      res.x[2], res.x[3]+(np.random.random()-0.5)/16], 
                               method='BFGS')  # seed with previous values + some small noise at the same scale as params
            else:
                res = minimize(self.loss, [1,0.00001,0,self.mi_init1], method='BFGS')

            residual = np.nanmean(np.abs((res.x[3]*(1-(res.x[0] * np.exp(-res.x[1] * self.t)))+np.abs(res.x[2])*self.t)-data_arr[self.t]))
            residuals.append(residual)
            res2 = minimize(self.loss, [res.x[0], res.x[1],res.x[2],self.mi_init2], method='BFGS')
            residual2 = np.nanmean(np.abs((res2.x[3]*(1-(res2.x[0] * np.exp(-res2.x[1] * self.t)))+np.abs(res2.x[2])*self.t)-data_arr[self.t]))
            if residual2<residual:
                residuals.append(residual2)
                mi_est.append(res2.x[3])
            else:
                mi_est.append(res.x[3])

            converge_ix = self.converge(mi_est, i, min_iters, max_iters, residual=self.converge_res_thresh, thresh=self.converge_len_thresh)
            if len(mi_est)>=2 and converge_ix:
                break
        #print('MI pred:', mi_est[-1], '/ gt:', gt_mis[k,ark_ixs[j][0], ark_ixs[j][1]], len(mi_est))

        # check if convergence worked (is mi_est exactly the same as the init?)
        cix = self.converge(mi_est, i, min_iters, max_iters, residual=self.converge_res_thresh, thresh=self.converge_len_thresh)
        final_mi_est = mi_est[cix]

        if final_mi_est!=self.mi_init1 and final_mi_est!=self.mi_init2: 
            return {'bias_corrected_mi': final_mi_est, 'all_mi_estimates': mi_est, 
                    'raw_observed_mi_trajectories': np.concatenate(raw_mi_obs),'convergence_times': x_sub[cix]}
        else:
            return {'bias_corrected_mi': np.nan, 'all_mi_estimates': np.nan, 'raw_observed_mi_trajectories': np.concatenate(raw_mi_obs),
                    'convergence_times': np.nan}
       
    def refit(self, raw_mi_obs, min_iters=4e4, max_iters=4e5, step_size=2e4, dynamic_start=True, return_all=False):
        """
        Performs some basic refitting, but it is always best to check goodness of fit manually. This function may not be able to find a 
        good fit and it may be useful to sweep through mi_inits another way and check their goodness of fit.
        """
        res = False
        residuals = []
        mi_est = []  # contains the full trajectory
        x_sub = np.arange(int(step_size),int(max_iters+1),int(step_size))
        for i in x_sub:   # this is basically where I would train
            self.t = np.arange(i).astype(np.int32)
            data_arr = np.array(raw_mi_obs)[:i]  # data_arr is an ever growing array of the output every 20k iterations
            # clipping infinities helps a lot!
            self.data_arr = np.clip(data_arr, np.nanmin(data_arr[data_arr != -np.inf]), np.nanmax(data_arr[data_arr != np.inf]))
            if dynamic_start:
                # start fitting only after a burn in period
                start = _moving_average(self.data_arr, 100)
                if (start>0.025).any():
                    start = np.where(start>0.025)[0][0]
                    self.data_arr = self.data_arr[start:]
                    self.t = np.arange(len(self.data_arr))
                else:
                    continue
            if res:
                res = minimize(self.loss, [res.x[0], res.x[1]+(np.random.random()-0.5)*res.x[1]/8, 
                                      res.x[2], res.x[3]+(np.random.random()-0.5)/16], 
                               method='BFGS')  # seed with previous values + some small noise at the same scale as params
            else:
                res = minimize(self.loss, [1,0.00001,0,self.mi_init1], method='BFGS')

            residual = np.nanmean(np.abs((res.x[3]*(1-(res.x[0] * np.exp(-res.x[1] * self.t)))+np.abs(res.x[2])*self.t)-data_arr[self.t]))
            residuals.append(residual)
            res2 = minimize(self.loss, [res.x[0], res.x[1],res.x[2],self.mi_init2], method='BFGS')
            residual2 = np.nanmean(np.abs((res2.x[3]*(1-(res2.x[0] * np.exp(-res2.x[1] * self.t)))+np.abs(res2.x[2])*self.t)-data_arr[self.t]))
            if residual2<residual:
                residuals.append(residual2)
                mi_est.append(res2.x[3])
            else:
                mi_est.append(res.x[3])

            converge_ix = self.converge(mi_est, i, min_iters, max_iters, residual=self.converge_res_thresh, thresh=self.converge_len_thresh)
            if len(mi_est)>=2 and converge_ix:
                break
        return mi_est
