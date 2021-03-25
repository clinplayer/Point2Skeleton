import numpy as np
import os


def load_data_id(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    id_list = []
    linecount = 0

    for line in lines:
        if line == '\n':
            continue
        id_list.append(line.strip('\n'))
        linecount = linecount + 1
    fopen.close()
    return id_list


def check_and_create_dirs(dir_list):
    for dir in dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(dir + ' does not exist. Created.')


def save_off_points(points, path):
    with open(path, "w") as file:
        file.write("OFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(
                str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + "\n")


def save_off_mesh(v, f, path):
    with open(path, "w") as file:
        file.write("OFF\n")
        v_num = len(v)
        f_num = len(f)
        file.write(str(v_num) + " " + str(len(f)) + " " + str(0) + "\n")
        for j in range(v_num):
            file.write(str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
        for j in range(f_num):
            file.write("3 " + str(int(f[j][0])) + " " + str(int(f[j][1])) + " " + str(int(f[j][2])) + "\n")


def save_coff_points(points, colors, path):
    with open(path, "w") as file:
        file.write("COFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + " ")
            file.write(str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]) + "\n")


def save_graph(v, A, path):
    with open(path, "w") as file:
        file.write("g line\n")
        v_num = len(v)
        for j in range(v_num):
            file.write("v " + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
        file.write("g\n")

        # A is a symmetric matrix
        for j in range(v_num):
            for k in range(j + 1, v_num):
                if A[j][k] == 1:
                    file.write("l " + str(j + 1) + " " + str(k + 1) + "\n")


def save_spheres(center, radius, path):
    sp_v, sp_f = load_off('sphere16.off')

    with open(path, "w") as file:
        for i in range(center.shape[0]):
            v, r = center[i], radius[i]
            v_ = sp_v * r
            v_ = v_ + v
            for m in range(v_.shape[0]):
                file.write('v ' + str(v_[m][0]) + ' ' + str(v_[m][1]) + ' ' + str(v_[m][2]) + '\n')

        for m in range(center.shape[0]):
            base = m * sp_v.shape[0] + 1
            for j in range(sp_f.shape[0]):
                file.write(
                    'f ' + str(sp_f[j][0] + base) + ' ' + str(sp_f[j][1] + base) + ' ' + str(sp_f[j][2] + base) + '\n')


def save_skel_mesh(v, f, e, path_f, path_e):
    f_file = open(path_f, "w")
    e_file = open(path_e, "w")
    v_num = len(v)
    f_num = len(f)
    e_num = len(e)

    for j in range(v_num):
        f_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
    for j in range(f_num):
        f_file.write("f " + str(int(f[j][0]) + 1) + " " + str(int(f[j][1]) + 1) + " " + str(int(f[j][2]) + 1) + "\n")

    for j in range(v_num):
        e_file.write('v ' + str(float(v[j][0])) + " " + str(float(v[j][1])) + " " + str(float(v[j][2])) + "\n")
    for j in range(e_num):
        e_file.write("l " + str(int(e[j][0]) + 1) + " " + str(int(e[j][1]) + 1) + "\n")

    f_file.close()
    e_file.close()


def save_skel_xyzr(v, r, path):
    file = open(path, "w")
    v_num = len(v)
    file.write(str(v_num) + "\n")
    for i in range(v_num):
        file.write(
            str(float(v[i][0])) + " " + str(float(v[i][1])) + " " + str(float(v[i][2])) + " " + str(float(r[i])) + "\n")
    file.close()


def save_colored_weights(path, shape_name, weights, samples):
    skel_num = weights.shape[0]
    sample_num = weights.shape[1]
    min_gray = 200
    for i in range(skel_num):
        colors = np.zeros((sample_num, 3)).astype(np.int)
        max_w = max(weights[i].tolist())
        for j in range(sample_num):
            color = min_gray - int((weights[i][j] / max_w) * min_gray)
            colors[j] = np.array([color, color, color], np.int)

        save_coff_points(samples, colors, path + str(shape_name) + '_' + str(i) + '_weight.off')


def load_off(path):
    fopen = open(path, 'r', encoding='utf-8')
    lines = fopen.readlines()
    linecount = 0
    pts = np.zeros((1, 3), np.float64)
    faces = np.zeros((1, 3), np.int)
    p_num = 0
    f_num = 0

    for line in lines:
        linecount = linecount + 1
        word = line.split()

        if linecount == 1:
            continue
        if linecount == 2:
            p_num = int(word[0])
            f_num = int(word[1])
            pts = np.zeros((p_num, 3), np.float)
            faces = np.zeros((f_num, 3), np.int)
        if linecount >= 3 and linecount < 3 + p_num:
            pts[linecount - 3, :] = np.float64(word[0:3])
        if linecount >= 3 + p_num:
            faces[linecount - 3 - p_num] = np.int32(word[1:4])

    fopen.close()
    return pts, faces


def load_ply_points(pc_filepath, expected_point=2000):
    fopen = open(pc_filepath, 'r', encoding='utf-8')
    lines = fopen.readlines()
    pts = np.zeros((expected_point, 3), np.float64)

    total_point = 0
    feed_point_count = 0

    start_point_data = False
    for line in lines:
        word = line.split()
        if word[0] == 'element' and word[1] == 'vertex':
            total_point = int(word[2])
            # if expected_point > total_point:
            #     pts = np.zeros((total_point, 3), np.float64)
            # continue

        if start_point_data == True:
            pts[feed_point_count, :] = np.float64(word[0:3])
            feed_point_count += 1

        if word[0] == 'end_header':
            start_point_data = True

        if feed_point_count >= expected_point:
            break

    fopen.close()
    return pts
