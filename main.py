from gdm import *
from p_dict import *
from simulation import *


gauss = GDM(test_dict)
test = Sim('test',{})
gauss.save_csvs('data/test')

# e_data,t_data = gauss.generate_data()
# fig,ax = plt.subplots()
# anim = gauss.animate(fig,ax)
# # ax1 = gauss.plot_energy(ax)
# plt.show()
