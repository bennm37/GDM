import os 
from shutil import rmtree

class Sim(object):
    def __init__(self,project_name,param_dicts):
        self.param_dicts = param_dicts
        self.status = self.architect(project_name)

    def architect(self,project_name):
        if os.path.exists(f'data/{project_name}'):
            if input(f'Delete Project {project_name}? (y/n) \n') == 'y':
                rmtree(f'data/{project_name}')
                if os.path.exists(f'media/{project_name}'):
                    rmtree(f'media/{project_name}')
            else:
                return -1 
        os.mkdir(f'data/{project_name}')
        os.mkdir(f'media/{project_name}')
        return 0
    
    def get_parameter_suffix(self):
        """Generates a parameter suffix for csv naming"""
        self.suffixes = ["__".join([key+"_"+str(p_dict[key]) for key in p_dict]) for p_dict in self.param_dicts]
