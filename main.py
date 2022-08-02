from gdm import *
from p_dict import *
from simulation import *

folder_name = 'lattice_2d2'


# plt.style.use('ggplot')
# a = Analsyis(f'data/{folder_name}',lattice_3d_dict)
# anim = a.animate_3d()
# # plt.show()
# anim.save(f'media/{folder_name}/test.mp4')

gauss = GDM(lattice_2d_dict)
test = Sim(f'{folder_name}',{})
if not test.status:
    gauss.generate_data(f'data/{folder_name}')
    a = Analsyis(f'data/{folder_name}',lattice_2d_dict)
    fig,ax = plt.subplots()
    anim = a.animate(fig,ax)
    anim.save(f'media/{folder_name}/test.mp4')
else:
    print('Leaving Project Unchanged')


