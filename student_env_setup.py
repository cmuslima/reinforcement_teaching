
#this is about the student env
from grid_envs import basic_grids,cliff_world

def build_env(args):
    if args.tabular:     
        env = basic_grids(args.env, args.columns, args.rows)
        if args.env == 'cliff_world':
            env = cliff_world()
    return env
