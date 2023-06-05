from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from skimage import measure

def save_output(output_arr, output_size, output_dir, file_idx):
    plot_out_arr = np.array([])
    with_border_arr = np.zeros([output_size, output_size, output_size])
    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                plot_out_arr = np.append(plot_out_arr, output_arr[x_i, y_j, z_k])
    #             print('now:',x_i,z_k,z_k)
    #             print('plot_out_arr',plot_out_arr.shape)
    # print('15 line')            
    text_save = np.reshape(plot_out_arr, (output_size * output_size * output_size))
    np.savetxt(output_dir + '/volume' + str(file_idx) + '.txt', text_save)
    # print('18 line')
    output_image = np.reshape(plot_out_arr, (output_size, output_size, output_size)).astype(np.float32)
    # print('20 line')
    for x_i in range(0, output_size):
        for y_j in range(0, output_size):
            for z_k in range(0, output_size):
                # with_border_arr[x_i + 1, y_j + 1, z_k + 1] = output_image[x_i, y_j, z_k]
                with_border_arr[x_i , y_j , z_k] = output_image[x_i, y_j, z_k]
    if not np.any(with_border_arr):
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes_lewiner(with_border_arr, level = 0, gradient_direction = 'descent')
        faces = faces + 1
    # print('31 line')
    obj_save = open(output_dir + '/volume' + str(file_idx) + '.obj', 'w')
    for item in verts:
        obj_save.write('v {0} {1} {2}\n'.format(item[0], item[1], item[2]))
    for item in normals:
        obj_save.write('vn {0} {1} {2}\n'.format(-item[0], -item[1], -item[2]))
    for item in faces:
        obj_save.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(item[0], item[1], item[2]))
    obj_save.close()