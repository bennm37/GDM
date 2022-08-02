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
        self.n_dim = params['n_dim']
        self.T = params['T']
        self.n_steps = 100
        self.E_0 = params['E_0']
        self.sigma = params['sigma']
        self.nu_0 = params['nu_0']
        self.gamma = params['gamma']
        self.Temp = params['TEMP']
        self.K_B = params['K_B']
        self.energies = np.random.normal(self.E_0,self.sigma,self.n_sites)
        self.electrons = np.random.choice(range(self.n_sites),self.n_elec,False)
        self.times = np.zeros(self.n_elec)
        if params['init_type']=='uniform':
            self.uniform_sites()
        if params['init_type']=='lattice':
            self.lattice_sites()
            self.energies = lag.norm(self.sites,axis=1)
        self.get_rate_matrix()
        

    def uniform_sites(self):
        """Uniformly distributes sites in a box of size 1^n_dim"""
        self.box_width = 1
        self.sites = np.random.uniform(-self.box_width/2,self.box_width/2,(self.n_sites,self.n_dim))
    
    def lattice_sites(self):
        """Creates a lattice with floor(nth root of n_sites)^n sites,
        and updates n_sites accordingly"""
        if self.n_dim == 2:
            n_x = np.floor(np.sqrt(self.n_sites)).astype(int)
            x = np.linspace(-1,1,n_x)
            X,Y = np.meshgrid(x,x)
            self.sites = np.transpose([X.flatten(),Y.flatten()])
            self.n_sites = len(self.sites)

        elif self.n_dim == 3:
            n_x = np.round(self.n_sites**(1/3),0).astype(int)
            x = np.linspace(-1,1,n_x)
            X,Y,Z = np.meshgrid(x,x,x)
            self.sites = np.transpose([X.flatten(),Y.flatten(),Z.flatten()])
            self.n_sites = len(self.sites)
        else:
            print('Failed to make sites: n_dim must be 2 or 3')



    def update(self):
        rate_rows = self.R[self.electrons,:]
        clocks = np.random.exponential(1,rate_rows.shape)/rate_rows
        self.electrons = np.argmin(clocks,axis=1)
        self.times = np.min(clocks,axis=1)

    def generate_data(self,folder_name):
        """Generates hop and cumilative time data for each electron, 
        according to parameter dictionary. """
        self.electron_data = np.zeros((1,self.n_elec))
        self.time_data = np.zeros((1,self.n_elec))
        self.electron_data[0,:] = self.electrons
        self.time_data[0,:] = self.times
        for i in range(1,self.n_steps):
            self.update()
            finished = self.time_data[i-1]+self.times>self.T
            e_row = np.where(finished,self.electron_data[i-1],self.electrons).reshape(1,self.n_elec)
            self.electron_data = np.append(self.electron_data,e_row,axis=0)
            t_row = np.where(finished,self.T,self.times+self.time_data[i-1,:]).reshape(1,self.n_elec)
            self.time_data = np.append(self.time_data,t_row,axis=0)
            if np.all(self.time_data[i]==self.T):
                break
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
        self.envec = self.energies[np.newaxis,:]-self.energies[:,np.newaxis]


    def get_rate_matrix(self):
        """Generates rate matrix according to Miller-Abrahams formula from
        Tress Thesis"""
        self.get_envec()
        self.get_pvec()
        self.R = self.nu_0*np.exp(-2*self.gamma*self.pdist)*np.where(self.envec>0,np.exp(-self.envec/(self.K_B*self.Temp)),1)
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
        """Aniamtes a multiple electrons hopping around."""
        anim_t = np.linspace(0,np.max(self.time_data),100)
        # sites_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=40)
        site_scat = self.plot_energy(ax,0)
        e_loc = self.sites[self.electron_data[0].astype(int)]
        disp = 0.02
        e_scat = ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="green",s =20,marker="^")
        def update(i):
            n_max = self.time_data.shape[0]
            for j in range(self.n_elec):
                state = np.digitize(anim_t[i],self.time_data[:,j])
                state = min(state,n_max-1)
                try:
                    e_loc[j] = self.sites[self.electron_data[state,j].astype(int)]
                except IndexError:
                    print(f'{state=}')
            disp = np.array([0,0.02])
            e_scat.set_offsets(e_loc+disp)
            # e_scat.set_offsets(self.sites[e_loc][0],self.sites[e_loc][1])
        
        anim = animation.FuncAnimation(fig,update,len(anim_t))
        return anim

    def animate_3d(self):
        """Aniamtes a multiple electrons hopping around in 3d."""
        anim_t = np.linspace(0,np.max(self.time_data),100)
        # sites_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=40)
        # site_scat = self.plot_energy(ax,0)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        site_scat = ax.scatter(self.sites[:,0],self.sites[:,1],self.sites[:,2],c=self.energies,s=25,cmap="coolwarm",alpha=0.6)
        e_loc = self.sites[self.electron_data[0].astype(int)]
        disp = 0.02
        e_scat = ax.scatter(e_loc[:,0],e_loc[:,1],e_loc[:,2],c="green",s =20,marker="o",depthshade=0)
        def update(i):
            n_max = self.time_data.shape[0]
            for j in range(self.n_elec):
                state = np.digitize(anim_t[i],self.time_data[:,j])
                state = min(state,n_max-1)
                try:
                    e_loc[j] = self.sites[self.electron_data[state,j].astype(int)]
                except IndexError:
                    print(f'{state=}')
            disp = np.array([0,0,0.02])
            e_scat._offsets3d = e_loc.T
            # e_scat.set_offsets(self.sites[e_loc][0],self.sites[e_loc][1])
        
        anim = animation.FuncAnimation(fig,update,len(anim_t))
        return anim


