from gdm import *
from p_dict import *
from simulation import *

a = Analsyis('data/test',test_dict)
fig,ax = plt.subplots()
anim = a.animate(fig,ax)
# a.plot_energy(ax,0)
plt.show()


# gauss = GDM(test_dict)
# test = Sim('test',{})
# if not test.status:
#     gauss.generate_data('data/test')
#     a = Analsyis('data/test',test_dict)
#     fig,ax = plt.subplots()
#     # a.animate(fig,ax)
#     a.plot(ax)
#     plt.show()
# else:
#     print('Leaving Project Unchanged')




