import numpy as np
import torch
import copy
import DistFunc as DF


def find_cir(G, x, target, poten, path=()):
    if path is None:
        path = (x,)
    if poten < 1:
        pass
    elif poten == 1:
        if target in G[x]:
            yield path
    else:
        for y in G[x]:
            if y not in path and not list(set(G[y]) & set(path[0:len(path) - 1])) and poten > 2:
                yield from find_cir(G, y, target, poten - 1, path + (y,))
            elif y not in path and not list(set(G[y]) & set(path[1:len(path) - 1])) and poten == 2:
                yield from find_cir(G, y, target, poten - 1, path + (y,))


def find_all_cirs(G, n):
    s = set()
    s_final = set()
    for k in range(0, len(G)):
        for c in find_cir(G, k, k, n, None):
            before = len(s)
            s.add(tuple(sorted(c)))
            after = len(s)
            if after > before:
                s_final.add(tuple(c))
    return s_final


def extract_skel_face(A):
    pn = A.shape[0]
    faces = []
    for i in range(pn):
        for j in range(i, pn):
            for k in range(j, pn):
                check1, check2, check3 = False, False, False
                if A[i, j] == 1 or A[j, i] == 1:
                    check1 = True
                if A[i, k] == 1 or A[k, i] == 1:
                    check2 = True
                if A[j, k] == 1 or A[k, j] == 1:
                    check3 = True

                if check1 and check2 and check3:
                    faces.append([i, j, k])

    return faces


def extract_skel_edge(A, faces):
    pn = A.shape[0]
    edges = []
    A_count = np.zeros_like(A)

    for f in faces:
        i, j, k = f[0], f[1], f[2]
        A_count[i, j] += 1
        A_count[j, i] += 1
        A_count[i, k] += 1
        A_count[k, i] += 1
        A_count[j, k] += 1
        A_count[k, j] += 1

    for i in range(pn):
        for j in range(i, pn):
            if (A_count[i, j] == 0 or A_count[j, i] == 0) and (A[i, j] == 1 or A[j, i] == 1):
                edges.append([i, j])

    return edges


def extract_nonshared_edge(A, faces):
    A_single = copy.copy(A)
    A_count = np.zeros_like(A)

    for f in faces:
        i, j, k = f[0], f[1], f[2]
        A_count[i, j] += 1
        A_count[j, i] += 1
        A_count[i, k] += 1
        A_count[k, i] += 1
        A_count[j, k] += 1
        A_count[k, j] += 1

        A_single[i, j] = 1
        A_single[j, i] = 1
        A_single[i, k] = 1
        A_single[k, i] = 1
        A_single[j, k] = 1
        A_single[k, j] = 1

    A_single[A_count >= 2] = 0

    return A_single


def convert_adj_linkedlist(A):
    pn = A.shape[0]
    A_link = []
    for i in range(pn):
        A_i_neighbor = []
        for j in range(pn):
            if A[i, j] == 1:
                A_i_neighbor.append(j)
        A_link.append(A_i_neighbor)
    return A_link


def compute_face_areas(vertices, faces):
    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]
    tmp_cross = np.cross(v0 - v2, v1 - v2)

    areas = 0.5 * np.sqrt(np.sum(tmp_cross * tmp_cross, axis=1))
    return areas


def compute_edge_lengths(vertices, edges):
    if edges.shape[0] == 0:
        return np.array([0])

    v0 = vertices[edges[:, 0], :]
    v1 = vertices[edges[:, 1], :]

    lengths = np.linalg.norm(v1 - v0, axis=1)

    return lengths


def rand_sample_points_on_tri_mesh(vertices, faces, num_sample):
    areas = compute_face_areas(vertices, faces)
    probabilities = areas / areas.sum()
    weighted_random_indices = np.random.choice(range(areas.shape[0]), size=num_sample, p=probabilities)

    u = np.random.rand(num_sample, 1)
    v = np.random.rand(num_sample, 1)
    w = np.random.rand(num_sample, 1)

    sum_uvw = u + v + w
    u = u / sum_uvw
    v = v / sum_uvw
    w = w / sum_uvw

    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]
    v0 = v0[weighted_random_indices]
    v1 = v1[weighted_random_indices]
    v2 = v2[weighted_random_indices]

    sampled_v = (v0 * u) + (v1 * v) + (v2 * w)
    sampled_v = sampled_v.astype(np.float32)
    sel_fids = weighted_random_indices
    return sampled_v, sel_fids


def rand_sample_points_on_skeleton_mesh(vertices, faces, edges, r, num_sample):
    faces = np.array(faces)
    areas = compute_face_areas(vertices, faces)
    sum_f = areas.sum()
    prob_f = areas / sum_f

    edges = np.array(edges)
    lengths = compute_edge_lengths(vertices, edges)
    lengths = 0.5 * lengths * lengths
    sum_e = lengths.sum()

    if sum_e != 0:
        prob_e = lengths / sum_e
    else:
        prob_e = 0

    vertices = np.concatenate((vertices, r), axis=1)

    num_sample_f = int((sum_f / (sum_f + sum_e)) * num_sample)
    num_sample_e = num_sample - num_sample_f

    weighted_random_indices = np.random.choice(range(areas.shape[0]), size=num_sample_f, p=prob_f)

    u = np.random.rand(num_sample_f, 1)
    v = np.random.rand(num_sample_f, 1)
    w = np.random.rand(num_sample_f, 1)

    sum_uvw = u + v + w
    u = u / sum_uvw
    v = v / sum_uvw
    w = w / sum_uvw

    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :]
    v2 = vertices[faces[:, 2], :]
    v0 = v0[weighted_random_indices]
    v1 = v1[weighted_random_indices]
    v2 = v2[weighted_random_indices]

    sampled_v_f = (v0 * u) + (v1 * v) + (v2 * w)
    sampled_v_f = sampled_v_f.astype(np.float32)

    if num_sample_e != 0:

        sampled_v_e = np.zeros((1, 4), np.float32)
        weighted_random_indices = np.random.choice(range(lengths.shape[0]), size=num_sample_e, p=prob_e)

        u = np.random.rand(num_sample_e, 1)
        v = np.random.rand(num_sample_e, 1)

        sum_uvw = u + v
        u = u / sum_uvw
        v = v / sum_uvw

        v0 = vertices[edges[:, 0], :]
        v1 = vertices[edges[:, 1], :]
        v0 = v0[weighted_random_indices]
        v1 = v1[weighted_random_indices]

        sampled_v_e = (v0 * u) + (v1 * v)
        sampled_v_e = sampled_v_e.astype(np.float32)

        sampled_all = np.concatenate((sampled_v_f, sampled_v_e), axis=0)

    else:
        sampled_all = sampled_v_f

    return sampled_all


def refine_boundary(faces, skel_p, A, A_recon):
    pn = A.shape[0]
    A_count_f = np.zeros_like(A)
    A_count_e = np.zeros((pn))

    A_fill = copy.copy(A)

    for f in faces:
        i, j, k = f[0], f[1], f[2]
        A_count_f[i, j] += 1
        A_count_f[j, i] += 1
        A_count_f[i, k] += 1
        A_count_f[k, i] += 1
        A_count_f[j, k] += 1
        A_count_f[k, j] += 1
        A_count_e[i] += 1
        A_count_e[j] += 1
        A_count_e[k] += 1

    edges = extract_skel_edge(A, faces)

    for e in edges:
        i, j = e[0], e[1]
        A_count_e[i] += 1
        A_count_e[j] += 1

    boundary_face_points = []
    boundary_edge_points = []

    for i in range(pn):
        if 1 in A_count_f[i]:
            boundary_face_points.append(i)
        if A_count_e[i] == 1:
            boundary_edge_points.append(i)

    link_candidate_f = []
    for i in boundary_face_points:
        for j in boundary_face_points:
            if i != j and A_recon[i][j] == 1:
                link_candidate_f.append([i, j])

    link_candidate_e = []
    for i in boundary_edge_points:
        for j in boundary_edge_points:
            if i != j and A_recon[i][j] == 1:
                link_candidate_e.append([i, j])

    valid_link = link_candidate_f + link_candidate_e
    for link in valid_link:
        i, j = link[0], link[1]
        A_fill[i][j] = 1
        A_fill[j][i] = 1

    return A_fill


def fill_holes(faces, skel_p, input_p, A_init, A_recon, max_loop_len=9, min_nonshared_e=3, min_proj_v=3):
    A_loop_link = convert_adj_linkedlist(A_init)
    A_fill = copy.copy(A_init)
    f_init_num = len(faces)

    cir_id = 0
    tmp_faces = copy.copy(faces)
    f2c_id = []
    all_circles = []

    # extract all loops
    for loop_len in range(4, max_loop_len):
        A_nonshared = extract_nonshared_edge(A_init, tmp_faces)
        circles = find_all_cirs(A_loop_link, loop_len)
        for cir in circles:
            count_nonshared = 0
            for k in range(0, len(cir)):
                if A_nonshared[cir[k]][cir[k + 1] if k < len(cir) - 1 else cir[0]] == 1:
                    count_nonshared += 1

            if count_nonshared < min_nonshared_e:
                continue

            for k in range(0, len(cir) - 2):
                tmp_faces.append([cir[0], cir[k + 1], cir[k + 2]])
                f2c_id.append(cir_id)

            all_circles.append(cir)
            cir_id += 1

    # projection to tmp_faces and count: p2p -> p2f -> f2c -> cir_id
    f2c_id = np.array(f2c_id)
    dense_skel_p, p2f_id = rand_sample_points_on_tri_mesh(skel_p, np.array(tmp_faces), 4000)
    p2f_dist, proj_id = DF.closest_distance_np(input_p, dense_skel_p, is_sum=False)
    proj_id = proj_id.numpy().tolist()

    f_id = p2f_id[proj_id]
    f_id_select = [f - f_init_num for f in f_id if f >= f_init_num]
    cir_id_count = f2c_id[f_id_select]

    cir_id_2_proj_v = {}
    for cir_id in cir_id_count:
        cir_id_2_proj_v[cir_id] = cir_id_2_proj_v.get(cir_id, 0) + 1

    add_faces = []
    cir_id = 0
    for cir in all_circles:
        cir_id += 1
        cir_len = len(cir)
        need_fill_explicit, need_fill_latent = True, True

        # check count from input projection
        if cir_id_2_proj_v.get(cir_id - 1, 0) < min_proj_v:
            need_fill_explicit = False

        # check GAE link prediction by extracting the subgraph
        A1 = A_recon[list(cir[0:cir_len])]
        A2 = A1[:, list(cir[0:cir_len])]
        if np.sum(A2) <= cir_len * 2:
            need_fill_latent = False

        if not need_fill_latent and not need_fill_explicit:
            continue

        for k in range(0, cir_len - 2):
            add_faces.append([cir[0], cir[k + 1], cir[k + 2]])
            A_fill[cir[0]][cir[k + 1]] = 1
            A_fill[cir[k + 1]][cir[0]] = 1

            A_fill[cir[0]][cir[k + 2]] = 1
            A_fill[cir[k + 2]][cir[0]] = 1

            A_fill[cir[k + 2]][cir[k + 1]] = 1
            A_fill[cir[k + 1]][cir[k + 2]] = 1

    return A_fill, faces + add_faces


def generate_skel_mesh(input_xyz, skel_xyz, A_init, A_final):
    batch_size = skel_xyz.size()[0]
    input_xyz_np = input_xyz.detach().cpu().numpy()
    skel_xyz_np = skel_xyz.detach().cpu().numpy()
    A_init_np = A_init.detach().cpu().numpy()
    A_final_np = A_final.detach().cpu().numpy()
    A_mesh_np = np.zeros_like(A_init_np)
    face_batch = []
    edge_batch = []
    for i in range(batch_size):
        faces = extract_skel_face(A_init_np[i])
        A_refine, faces = fill_holes(faces, skel_xyz_np[i], input_xyz_np[i], A_init_np[i], A_final_np[i])
        A_refine = refine_boundary(faces, skel_xyz_np[i], A_refine, A_final_np[i])

        faces = extract_skel_face(A_refine)
        edges = extract_skel_edge(A_refine, faces)

        face_batch.append(faces)
        edge_batch.append(edges)
        A_mesh_np[i] = A_refine

    return face_batch, edge_batch, A_mesh_np


def refine_radius_by_mesh(skel_xyz, skel_r, sample_xyz, weights, skel_faces, skel_edges):
    batch_size = skel_xyz.size()[0]
    skel_xyz_np = skel_xyz.detach().cpu().numpy()
    skel_r_np = skel_r.detach().cpu().numpy()

    dense_sample_num = 10000
    dense_skel_xyz = torch.zeros([batch_size, dense_sample_num, 3])
    for i in range(batch_size):
        dense_skel_sphere = rand_sample_points_on_skeleton_mesh(skel_xyz_np[i], skel_faces[i], skel_edges[i],
                                                                     skel_r_np[i], dense_sample_num)
        dense_skel_xyz[i] = torch.from_numpy(dense_skel_sphere[:, 0:3])

    # recompute the radii
    dense_skel_xyz = dense_skel_xyz.cuda()
    min_dists, min_indices = DF.closest_distance_with_batch(sample_xyz, dense_skel_xyz, is_sum=False)
    skel_r = torch.sum(weights[:, :, :, None] * min_dists[:, None, :, None], dim=2)

    return skel_r