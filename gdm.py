import numpy as np 
import numpy.linalg as lag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import pandas as pd

class GDM(object):
    def __init__(self,params):
        """Sets up a system of n uniformly distributed sites"""
        self.n_sites = params['n_sites']
        self.n_elec = params['n_elec']
        self.T = params['T']
        self.n_steps = 100
        self.E_0 = params['E_0']
        self.sigma = params['sigma']
        self.nu_0 = params['nu_0']
        self.gamma = params['gamma']
        self.K_B = params['K_B']
        if params['init_type']=='uniform':
            self.uniform_sites()
        self.get_rate_matrix()
        

    def uniform_sites(self):
        self.box_width = 1
        self.sites = np.random.uniform(-self.box_width/2,self.box_width/2,(self.n_sites,2))
        self.energies = np.random.normal(self.E_0,self.sigma,self.n_sites)
        self.electrons = np.random.choice(range(self.n_sites),self.n_elec,False)
        self.times = np.zeros(self.n_elec)
        # print(f"Sites is {self.sites}")
        # print(f"energies is {self.energies}")
        # print(f"electrons is {self.electrons}")
    

    def update(self):
        rate_rows = self.R[self.electrons,:]
        clocks = np.random.exponential(1,rate_rows.shape)/rate_rows
        self.electrons = np.argmin(clocks,axis=1)
        self.times += np.min(clocks,axis=1)

    def generate_data(self,folder_name):
        """Generates hop and cumilative time data for each electron, 
        according to parameter dictionary. """
        self.electron_data = np.zeros((self.n_steps,self.n_elec))
        self.time_data = np.zeros((self.n_steps,self.n_elec))
        self.electron_data[0,:] = self.electrons
        self.time_data[0,:] = self.times
        for i in range(1,self.n_steps):
            self.update()
            self.electron_data[i,:] = self.electrons
            self.time_data[i,:] = self.times+self.time_data[i-1,:]
        self.save_csvs(folder_name)
        return self.electron_data,self.time_data
    
    def save_csvs(self,folder_name):
        sites_df = pd.DataFrame(self.sites)
        sites_df.to_csv(f'{folder_name}/sites.csv',index_label='site_no')
        energies_df = pd.DataFrame(self.energies)
        energies_df.to_csv(f'{folder_name}/energies.csv',index_label='site_no')
        electron_data_df = pd.DataFrame(self.electron_data)
        electron_data_df.to_csv(f'{folder_name}/electron_data.csv',index_label='hop_no')
        time_data_df = pd.DataFrame(self.time_data)
        time_data_df.to_csv(f'{folder_name}/time_data.csv',index_label='hop_no')
    
    def get_pvec(self):
        """Generates array of pairwise vectors and distances"""
        self.pvec = self.sites[:,np.newaxis,:]-self.sites[np.newaxis,:,:]
        self.pdist = lag.norm(self.pvec,axis=2)

    def get_envec(self):
        """Generates array of pairwise energy differences"""
        ## CHECK ORDER
        self.envec = self.energies[:,np.newaxis]-self.energies[np.newaxis,:]


    def get_rate_matrix(self):
        """Generates rate matrix according to Miller-Abrahams formula from
        Tress Thesis"""
        self.get_envec()
        self.get_pvec()
        self.R = self.nu_0*np.exp(-2*self.gamma*self.pdist)*np.where(self.envec>0,self.envec,1)
        ##TODO could use markov style rate matrix here but setting diagonal to zero for now
        np.fill_diagonal(self.R,0)
        # np.fill_diagonal(self.R,-np.sum(self.R,axis=0))
        return self.R


class Analsyis(object):
    def __init__(self,folder_name,params):
        self.n_sites = params['n_sites']
        self.n_elec = params['n_elec']
        self.T = params['T']
        self.n_steps = 100
        self.E_0 = params['E_0']
        self.sigma = params['sigma']
        self.nu_0 = params['nu_0']
        self.gamma = params['gamma']
        self.K_B = params['K_B']
        self.load_data(folder_name)

    def load_data(self,folder_name):
        """Reads in data as numpy arrays"""
        self.sites = np.array(pd.read_csv(f'{folder_name}/sites.csv').iloc[:,1:])
        self.energies = np.array(pd.read_csv(f'{folder_name}/energies.csv').iloc[:,1:])
        self.time_data = np.array(pd.read_csv(f'{folder_name}/time_data.csv').iloc[:,1:])
        self.electron_data = np.array(pd.read_csv(f'{folder_name}/electron_data.csv').iloc[:,1:])
        # print(f'{self.sites=}')
        # print(f'{self.energies=}')
        # print(f'{self.time_data=}')
        # print(f'{self.electron_data=}')

    def plot(self,ax,i_hop):
        ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=100)
        e_loc = self.sites[self.electron_data[i_hop].astype(int)]
        disp = 0.02
        ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="blue",s =20)
        return ax
        
    def plot_rate(self,ax):
        m_site_ind = 0 
        m_site = self.sites[m_site_ind]
        cmap = cm.get_cmap("Blues")
        self.get_rate_matrix()
        row = self.R[m_site_ind]
        for i,site in enumerate(self.sites):
            ax.scatter(site[0],site[1],color=cmap(row[i]))
        ax.scatter(m_site[0],m_site[1],c="k",s=100)
        e_loc = self.sites[self.electrons.astype(bool)]
        disp = 0.02
        ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="red",s =20)
        return ax

    def plot_energy(self,ax,i_hop):
        cmap = cm.get_cmap("coolwarm")
        row = self.energies/np.max(self.energies)
        site_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c=row,s=40,cmap='coolwarm')
        # e_loc = self.sites[self.electron_data[i_hop].astype(int)]
        disp = 0.02
        # ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="red",s =20)
        cbar = plt.colorbar(site_scat)
        return ax
    
    def animate(self,fig,ax):
        """Aniamtes a single electron hopping around."""
        dt = 50
        anim_t = np.arange(0,np.max(self.time_data),dt)
        # sites_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=40)
        site_scat = self.plot_energy(ax,0)
        e_loc = self.sites[self.electron_data[0].astype(int)]
        disp = 0.02
        e_scat = ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="green",s =20,marker="^")
        def update(i):
            for j in range(self.n_elec):
                state = np.digitize(anim_t[i],self.time_data[:,j])
                e_loc[j] = self.sites[self.electron_data[state,j].astype(int)]
            disp = np.array([0,0.02])
            e_scat.set_offsets(e_loc+disp)
            # e_scat.set_offsets(self.sites[e_loc][0],self.sites[e_loc][1])
        
        anim = animation.FuncAnimation(fig,update,len(anim_t))
        return anim


