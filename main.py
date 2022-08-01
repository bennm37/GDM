from gdm import *
from p_dict import *

gauss = GDM(sim_dict)

e_data,t_data = gauss.generate_data()
fig,ax = plt.subplots()
anim = gauss.animate(fig,ax)
# ax1 = gauss.plot_energy(ax)
plt.show()
