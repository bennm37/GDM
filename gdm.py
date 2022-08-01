import numpy as np 
import numpy.linalg as lag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

class GDM(object):
    def __init__(self,params):
        """Sets up a system of n uniformly distributed sites"""
        self.n_sites = params['n_sites']
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
        # self.electrons = np.random.binomial(1,0.2,self.n_sites).astype(int)
        self.electrons = np.zeros(self.n_sites)
        self.electrons[0] = 1

        # print(f"Sites is {self.sites}")
        # print(f"energies is {self.energies}")
        # print(f"electrons is {self.electrons}")
    
    def generate_data(self):
        """Currently using single hop to update."""
        self.electron_data = np.zeros((self.n_steps,self.n_sites))
        self.time_data = np.zeros(self.n_steps)
        self.time = 0
        self.electron_data[0] = self.electrons
        for step in range(1,self.n_steps):
            self.single_hop()
            self.electron_data[step] = self.electrons
            self.time_data[step] = self.time
        return self.electron_data,self.time_data


    def single_hop(self):
        """Markov Chain Style Exponential Hopping. Two potential approaches:
        randomly pick neighbour in neighbour cube with probability V_ij/sum(V_ij)
        and exponential time x_exp/sum(V_ij). Or racing exponential clocks."""
        self.e_loc = np.where(self.electrons==1)
        row = list(self.R[self.e_loc][0])
        clocks = [np.random.exponential(1/r,1)[0] for r in row]
        self.new_loc = np.argmin(clocks)
        self.electrons[self.e_loc] = 0
        self.electrons[self.new_loc] = 1
        self.time += clocks[self.new_loc]
    
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
        self.T = params['T']
        self.n_steps = 100
        self.E_0 = params['E_0']
        self.sigma = params['sigma']
        self.nu_0 = params['nu_0']
        self.gamma = params['gamma']
        self.K_B = params['K_B']
        self.load_data(folder_name)

    def load_data(self,folder_name):
        pass

    def plot(self,ax):
        ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=100)
        e_loc = self.sites[self.electrons.astype(bool)]
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

    def plot_energy(self,ax):
        cmap = cm.get_cmap("coolwarm")
        self.get_rate_matrix()
        row = self.energies/np.max(self.energies)
        site_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c=row,s=40,cmap='coolwarm')
        e_loc = self.sites[self.electrons.astype(bool)]
        disp = 0.02
        ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="red",s =20)
        cbar = plt.colorbar(site_scat)
        return ax
    
    def animate(self,fig,ax):
        """Aniamtes a single electron hopping around."""
        dt = 0.1
        anim_t = np.arange(0,np.max(self.time_data),dt)
        # sites_scat = ax.scatter(self.sites[:,0],self.sites[:,1],c="k",s=40)
        site_scat = self.plot_energy(ax)
        e_loc = self.sites[self.electrons.astype(bool)]
        disp = 0.02
        e_scat = ax.scatter(e_loc[:,0],e_loc[:,1]+disp,c="green",s =20)
        def update(i):
            state = np.digitize(anim_t[i],self.time_data)
            electrons = self.electron_data[state]
            e_loc = self.sites[electrons.astype(bool)]
            disp = np.array([0,0.02])
            e_scat.set_offsets(e_loc+disp)
            # e_scat.set_offsets(self.sites[e_loc][0],self.sites[e_loc][1])
        
        anim = animation.FuncAnimation(fig,update,len(anim_t))
        return anim


