import argparse
from datetime import datetime
import os
from tqdm import trange
import numpy as np
from PIL import Image
import gc
try:
    from manta import *
except ImportError:
    pass

parser = argparse.ArgumentParser()

parser.add_argument("--log_dir", type=str, default='data/fire_2d')
parser.add_argument("--num_param", type=int, default=4)
parser.add_argument("--path_format", type=str, default='%d_%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='src_x_pos')
parser.add_argument("--p1", type=str, default='smoke_density')
parser.add_argument("--p2", type=str, default='smoke_temp_diff')
parser.add_argument("--p3", type=str, default='frames')

parser.add_argument("--num_src_x_pos", type=int, default=5)
parser.add_argument("--min_src_x_pos", type=float, default=0.3)
parser.add_argument("--max_src_x_pos", type=float, default=0.7)

parser.add_argument("--src_y_pos", type=float, default=0.1)


parser.add_argument("--num_smoke_density", type=int, default=11)
parser.add_argument("--min_smoke_density", type=int, default=-0.101)
parser.add_argument("--max_smoke_density", type=int, default=-0.001)

parser.add_argument("--num_smoke_temp_diff", type=int, default=11)
parser.add_argument("--min_smoke_temp_diff", type=int, default=0.1)
parser.add_argument("--max_smoke_temp_diff", type=int, default=1.1)

parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)


parser.add_argument("--num_simulations", type=int, default=121000)

parser.add_argument("--resolution_x", type=int, default=52)
parser.add_argument("--resolution_y", type=int, default=52)
parser.add_argument("--buoyancy", type=float, default=-4e-3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=True)
parser.add_argument("--time_step", type=float, default=0.5)
parser.add_argument("--adv_order", type=int, default=2)
parser.add_argument("--clamp_mode", type=int, default=2)

args = parser.parse_args()


def main():
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    field_type = ['v'] #, 'd']
    for field in field_type:
        field_path = os.path.join(args.log_dir,field)
        if not os.path.exists(field_path):
            os.makedirs(field_path)

    args_file = os.path.join(args.log_dir, 'args.txt')
    with open(args_file, 'w') as f:
        print('%s: arguments' % datetime.now())
        for k, v in vars(args).items():
            print('  %s: %s' % (k, v))
            f.write('%s: %s\n' % (k, v))

    p1_space = np.linspace(args.min_src_x_pos, 
                           args.max_src_x_pos,
                           args.num_src_x_pos)
    p2_space = np.linspace(args.min_smoke_density,
                           args.max_smoke_density,
                           args.num_smoke_density)
    p3_space = np.linspace(args.min_smoke_temp_diff,
                           args.max_smoke_temp_diff,
                           args.num_smoke_temp_diff)

    p_list = np.array(np.meshgrid(p1_space, p2_space,p3_space)).T.reshape(-1, 3)
    pi1_space = range(args.num_src_x_pos)
    pi2_space = range(args.num_smoke_density)
    pi3_space = range(args.num_smoke_temp_diff)
    pi_list = np.array(np.meshgrid(pi1_space, pi2_space, pi3_space)).T.reshape(-1, 3)

    res_x = args.resolution_x
    res_y = args.resolution_y
    v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
    # d_ = np.zeros([res_y,res_x], dtype=np.float32)
    # p_ = np.zeros([res_y,res_x], dtype=np.float32)
    # s_ = np.zeros([res_y,res_x], dtype=np.float32)

    v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # d_range = [np.finfo(np.float).max, np.finfo(np.float).min] # 0-1
    # p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
    # s_range = [np.finfo(np.float).max, np.finfo(np.float).min]


    # solver params
    gs = vec3(res_x, res_y, 1)
    gravity = vec3(0,-0.0981,0)

    s = Solver(name='main', gridSize=gs, dim=2)
    s.timestep = args.time_step
    
    flags = s.create(FlagGrid)
    vel = s.create(MACGrid)
    density = s.create(RealGrid)
    react = s.create(RealGrid)
    fuel = s.create(RealGrid)
    heat = s.create(RealGrid)
    flame = s.create(RealGrid)
    pressure = s.create(RealGrid)
    doOpen = args.open_bound

    noise = s.create(NoiseField, loadFromFile=True)
    noise.posScale = vec3(45)
    noise.clamp = True
    noise.clampNeg = 0
    noise.clampPos = 1
    noise.valScale = 1
    noise.valOffset = 0.75
    noise.timeAnim = 0.2

    bWidth = args.bWidth

        
    if (GUI):
        gui = Gui()
        gui.show(True)
        #gui.pause()

    print('start generation')
    for i in trange(len(p_list), desc='scenes'):
        flags.initDomain(boundaryWidth=bWidth)
        flags.fillGrid()
        if doOpen:
            setOpenBound(flags, bWidth, 'yY', FlagOutflow | FlagEmpty)

        vel.clear()
        density.clear()
        react.clear()
        fuel.clear()
        heat.clear()
        flame.clear()
        pressure.clear()
        
        p0, p1, p2 = p_list[i][0], p_list[i][1], p_list[i][2]

        boxSize = vec3(res_x/8, 0.05*res_y, res_x/8)
        boxCenter = gs*vec3(p0, args.src_y_pos, 0.5)
        sourceBox = s.create(Box, center=boxCenter,size=boxSize)
        
        
        for t in range(args.num_frames):
            densityInflow( flags=flags, density=density, noise=noise, shape=sourceBox, scale=1, sigma=0.5 )
            densityInflow( flags=flags, density=heat,noise=noise,  shape=sourceBox, scale=1, sigma=0.5 )
            densityInflow( flags=flags, density=fuel, noise=noise, shape=sourceBox, scale=1, sigma=0.5 )
            densityInflow( flags=flags, density=react,  noise=noise, shape=sourceBox, scale=1, sigma=0.5 )

            processBurn(fuel=fuel, density=density, react=react, heat=heat)

            advectSemiLagrange( flags=flags, vel=vel, grid=density, order=2 )
            advectSemiLagrange( flags=flags, vel=vel, grid=heat,   order=2 )
            advectSemiLagrange( flags=flags, vel=vel, grid=fuel,   order=2 )
            advectSemiLagrange( flags=flags, vel=vel, grid=react, order=2 )
            advectSemiLagrange( flags=flags, vel=vel, grid=vel,   order=2, openBounds=doOpen, boundaryWidth=bWidth )

            if doOpen:
                resetOutflow( flags=flags, real=density )

            vorticityConfinement( vel=vel, flags=flags, strength=0.1 )

            addBuoyancy( flags=flags, density=density, vel=vel, gravity=(gravity*p1))
            addBuoyancy( flags=flags, density=heat,    vel=vel, gravity=(gravity*p2))

            setWallBcs( flags=flags, vel=vel )
            solvePressure( flags=flags, vel=vel, pressure=pressure )

            updateFlame(react=react, flame=flame)

            copyGridToArrayMAC(vel, v_)
            # copyGridToArrayReal(density, d_)
            # copyGridToArrayReal(pressure, p_)
            # copyGridToArrayReal(stream, s_)
            
            v_range = [np.minimum(v_range[0], v_.min()),
                       np.maximum(v_range[1], v_.max())]
            # d_range = [np.minimum(d_range[0], d_.min()),
            #            np.maximum(d_range[1], d_.max())]
            # p_range = [np.minimum(p_range[0], p_.min()),
            #            np.maximum(p_range[1], p_.max())]
            # s_range = [np.minimum(s_range[0], s_.min()),
            #            np.maximum(s_range[1], s_.max())]

            param_ = [p0, p1, p2, t]
            pit = tuple(pi_list[i].tolist() + [t])
            v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
            np.savez_compressed(v_file_path, 
                                x=v_[...,:2],
                                y=param_)
            #timings.display()
            s.step()
        
        gc.collect()

    vrange_file = os.path.join(args.log_dir, 'v_range.txt')
    with open(vrange_file, 'w') as f:
        print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
        f.write('%.3f\n' % v_range[0])
        f.write('%.3f' % v_range[1])

    # drange_file = os.path.join(args.log_dir, 'd_range.txt')
    # with open(drange_file, 'w') as f:
    #     print('%s: density min %.3f max %.3f' % (datetime.now(), d_range[0], d_range[1]))
    #     f.write('%.3f\n' % d_range[0])
    #     f.write('%.3f' % d_range[1])

    # prange_file = os.path.join(args.log_dir, 'p_range.txt')
    # with open(prange_file, 'w') as f:
    #     print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
    #     f.write('%.3f\n' % p_range[0])
    #     f.write('%.3f' % p_range[1])

    # srange_file = os.path.join(args.log_dir, 's_range.txt')
    # with open(srange_file, 'w') as f:
    #     print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
    #     f.write('%.3f\n' % s_range[0])
    #     f.write('%.3f' % s_range[1])
    print('Done')
if __name__ == '__main__':
    main()
