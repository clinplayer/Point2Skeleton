import os
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from SkelPointNet import SkelPointNet
from GraphAE import LinkPredNet
from DataUtil import PCDataset
from datetime import datetime
import DistFunc as DF
import FileRW as rw
import MeshUtil as util
import config as conf


def parse_args():
    parser = argparse.ArgumentParser(description='Point2Skeleton')
    parser.add_argument('--pc_list_file', type=str, default='../data/data-split/all-test.txt',
                        help='file of the names of the point clouds')
    parser.add_argument('--data_root', type=str, default='../data/pointclouds/',
                        help='root directory of all the data')
    parser.add_argument('--point_num', type=int, default=2000, help='input point number')
    parser.add_argument('--skelpoint_num', type=int, default=100, help='output skeletal point number')

    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--load_skelnet_path', type=str,
                        default='../weights/weights-skelpoint.pth',
                        help='directory to load the skeletal point network parameters')
    parser.add_argument('--load_gae_path', type=str,
                        default='../weights/weights-gae.pth',
                        help='directory to load the GAE network parameters')
    parser.add_argument('--save_result_path', type=str, default='../results/',
                        help='directory to save the results')
    args = parser.parse_args()

    return args


def output_results(log_path, batch_id, input_xyz, skel_xyz, skel_r, skel_faces, skel_edges, A_mesh):

    batch_size = skel_xyz.size()[0]
    batch_id = batch_id.numpy()
    input_xyz_save = input_xyz.detach().cpu().numpy()
    skel_xyz_save = skel_xyz.detach().cpu().numpy()
    skel_r_save = skel_r.detach().cpu().numpy()
    
    for i in range(batch_size):

        save_name_input = log_path + str(batch_id[i]) + "_input.off"
        save_name_sphere = log_path + str(batch_id[i]) + "_sphere" + ".obj"
        save_name_center = log_path + str(batch_id[i]) + "_center" + ".off"
        save_name_f = log_path + str(batch_id[i]) + "_skel_face" + ".obj"
        save_name_e = log_path + str(batch_id[i]) + "_skel_edge" + ".obj"
        save_name_A_mesh = log_path + str(batch_id[i]) + "_mesh_graph" + ".obj"

        rw.save_off_points(input_xyz_save[i], save_name_input)
        rw.save_spheres(skel_xyz_save[i], skel_r_save[i], save_name_sphere)
        rw.save_off_points(skel_xyz_save[i], save_name_center)
        rw.save_skel_mesh(skel_xyz_save[i], skel_faces[i], skel_edges[i], save_name_f, save_name_e)
        rw.save_graph(skel_xyz_save[i], A_mesh[i], save_name_A_mesh)
        
        # dense_skel_sphere = util.rand_sample_points_on_skeleton_mesh(skel_xyz_save[i], skel_faces[i], skel_edges[i], skel_r_save[i], 10000)
        # rw.save_spheres(dense_skel_sphere[:,0:3], dense_skel_sphere[:,3,None], save_name_sphere)


if __name__ == "__main__":
    #parse arguments
    args = parse_args()
    pc_list_file = args.pc_list_file
    data_root = args.data_root
    point_num = args.point_num
    skelpoint_num = args.skelpoint_num
    gpu = args.gpu
    load_skelnet_path = args.load_skelnet_path
    load_gae_path = args.load_gae_path
    save_result_path = args.save_result_path

    #create folders
    rw.check_and_create_dirs([save_result_path])

    #load networks
    model_skel = SkelPointNet(num_skel_points=skelpoint_num, input_channels=0, use_xyz=True)
    model_gae = LinkPredNet()

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print("GPU Number:", torch.cuda.device_count(), "GPUs!")
        model_skel.cuda()
        model_skel.eval()
        model_gae.cuda()
        model_gae.eval()
    else:
        print("No CUDA detected.")
        sys.exit(0)

    model_skel.load_state_dict(torch.load(load_skelnet_path))
    model_gae.load_state_dict(torch.load(load_gae_path))

    #load data and test network
    pc_list = rw.load_data_id(pc_list_file)
    test_data = PCDataset(pc_list, data_root, point_num)
    data_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=False)
    for iter, batch_data in enumerate(data_loader):
        print('iter:', iter)
        batch_id, batch_pc = batch_data
        batch_id = batch_id
        batch_pc = batch_pc.cuda().float()

        # get skeletal points and the node features
        skel_xyz, skel_r, sample_xyz, weights, shape_features, A_init, valid_mask, known_mask = model_skel(
            batch_pc, compute_graph=True)
        skel_node_features = torch.cat([shape_features, skel_xyz, skel_r], 2)

        # get predicted mesh
        A_pred = model_gae(skel_node_features, A_init)
        A_final = model_gae.recover_A(A_pred, valid_mask)

        skel_faces, skel_edges, A_mesh = util.generate_skel_mesh(batch_pc, skel_xyz, A_init, A_final)
        skel_r = util.refine_radius_by_mesh(skel_xyz, skel_r, sample_xyz, weights, skel_faces, skel_edges)
        output_results(save_result_path, batch_id, batch_pc, skel_xyz, skel_r, skel_faces, skel_edges, A_mesh)
