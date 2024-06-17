import multiprocessing as mp
import os
import argparse
from roma import roma_outdoor
from tqdm import tqdm
import json
import torch
import imagesize
import numpy as np
import cv2
import trimesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap camera')

    parser.add_argument('--dense_folder', type=str, help='Project dir.')
    parser.add_argument('--image_folder', type=str, default="images")
    parser.add_argument('--camera_file', type=str, default="transforms.json")
    parser.add_argument('--match_interval', type=int, default=8, help="mathing interval for sparse points")
    parser.add_argument('--max_d', type=int, default=256, help='Maximum depth interval number')
    parser.add_argument('--interval_scale', type=float, default=1)
    parser.add_argument('--save_ply', action='store_true', help="save sparse ply")
    parser.add_argument('--nerf2opencv', action='store_true', help="convert from nerf to opencv coord")
    parser.add_argument('--run_single', action='store_true', help="dense_folder is the final folder for a single scene")
    parser.add_argument('--load_from_w2c', action='store_true', help="if we load w2c from camera, we need to change it to c2w")

    parser.add_argument('--theta0', type=float, default=5)
    parser.add_argument('--sigma1', type=float, default=1)
    parser.add_argument('--sigma2', type=float, default=10)

    args = parser.parse_args()

    root_path = args.dense_folder
    from glob import glob

    if args.run_single:
        fs = [args.dense_folder]
    else:
        fs = glob(f"{root_path}/*")
    for f in tqdm(fs):
        args.dense_folder = f
        cameras = json.load(open(os.path.join(args.dense_folder, args.camera_file), "r"))
        roma_model = roma_outdoor(device="cuda")

        match_frames = cameras["frames"][::args.match_interval]
        match_c2ws = [np.asarray(f["transform_matrix"]) for f in match_frames]
        if args.load_from_w2c:
            match_c2ws = [np.linalg.inv(m) for m in match_c2ws]
        match_images = [f'{args.dense_folder}/{args.image_folder}/{f["file_path"].split("/")[-1]}' for f in match_frames]
        image_w, image_h = imagesize.get(match_images[0])

        intrinsic = np.eye(3)
        if "fl_x" in cameras:
            intrinsic[0, 0] = cameras["fl_x"]
            intrinsic[1, 1] = cameras["fl_y"]
            intrinsic[0, 2] = cameras["cx"]
            intrinsic[1, 2] = cameras["cy"]
            origin_w, origin_h = cameras["w"], cameras["h"]
            intrinsic[0, :] *= (image_w / origin_w)
            intrinsic[1, :] *= (image_h / origin_h)

            all_Ks = [intrinsic] * len(cameras["frames"])
            match_Ks = [intrinsic] * len(match_frames)
        else:
            all_Ks = []
            for frame in cameras["frames"]:
                intrinsic = np.eye(3)
                intrinsic[0, 0] = frame["fl_x"]
                intrinsic[1, 1] = frame["fl_y"]
                intrinsic[0, 2] = frame["cx"]
                intrinsic[1, 2] = frame["cy"]
                origin_w, origin_h = frame["w"], frame["h"]
                intrinsic[0, :] *= (image_w / origin_w)
                intrinsic[1, :] *= (image_h / origin_h)
                all_Ks.append(intrinsic)
            match_Ks = all_Ks[::args.match_interval]

        # change nerf coordinates to opencv/colmap one (if needed)
        if args.nerf2opencv:
            rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        else:
            rot = np.eye(4)
        points3d = []
        colors3d = []
        roma_model.sample_thresh = 0.1
        for i in tqdm(range(len(match_frames) - 1), desc="building sparse points..."):
            with torch.no_grad():
                warp, certainty = roma_model.match(match_images[i], match_images[i + 1], device="cuda")

                matches, _ = roma_model.sample(warp, certainty, num=1000)
                src_pts, dst_pts = roma_model.to_pixel_coordinates(matches, image_h, image_w, image_h, image_w)

                src_pts = src_pts.cpu().numpy()
                dst_pts = dst_pts.cpu().numpy()
                proj_0 = match_Ks[i] @ np.linalg.inv(match_c2ws[i] @ rot)[:3, :4]  # (K@w2c)
                proj_1 = match_Ks[i + 1] @ np.linalg.inv(match_c2ws[i + 1] @ rot)[:3, :4]
                points = cv2.triangulatePoints(proj_0[:3, :4], proj_1[:3, :4], src_pts.T, dst_pts.T)
                points = points.T

                colors = cv2.imread(match_images[i])[:, :, ::-1]
                coord = src_pts.astype(np.int32)
                colors = colors[coord[:, 1], coord[:, 0], :]

                points = points[:, :3] / points[:, 3:4]

                xs_sorted = sorted(points[:, 0])
                x_min = xs_sorted[int(len(xs_sorted) * .02)]
                x_max = xs_sorted[int(len(xs_sorted) * .98)]
                ys_sorted = sorted(points[:, 1])
                y_min = ys_sorted[int(len(ys_sorted) * .02)]
                y_max = ys_sorted[int(len(ys_sorted) * .98)]
                zs_sorted = sorted(points[:, -1])
                z_min = zs_sorted[int(len(zs_sorted) * .02)]
                z_max = zs_sorted[int(len(zs_sorted) * .98)]

                no_inf_idx = ~np.isinf(points).reshape(-1, 3)
                points = points[no_inf_idx].reshape(-1, 3)
                colors = colors[no_inf_idx].reshape(-1, 3)

                range_idx = (points[:, 0] > x_min) * (points[:, 0] < x_max) * (points[:, 1] > y_min) * (points[:, 1] < y_max) * (points[:, -1] > z_min) * (points[:, -1] < z_max)
                points = points[range_idx, :]
                colors = colors[range_idx, :]

                points3d.append(points)
                colors3d.append(colors)

        points3d = np.concatenate(points3d, axis=0)
        colors3d = np.concatenate(colors3d, axis=0)

        if args.save_ply:
            ply = trimesh.PointCloud(vertices=points3d, colors=colors3d)
            _ = ply.export(f"{args.dense_folder}/sparse.ply")

        # depth range and interval
        depth_ranges = {}
        valid_idx_dict = {}
        c2ws = [np.asarray(f["transform_matrix"]) for f in cameras["frames"]]
        if args.load_from_w2c:
            c2ws = [np.linalg.inv(m) for m in c2ws]
        points4d = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)
        for i in range(len(cameras["frames"])):
            points4d_cam = (np.linalg.inv(c2ws[i] @ rot) @ points4d.T).T
            points3d_cam = points4d_cam[:, :3] / points4d_cam[:, 3:4]
            points2d_cam = (all_Ks[i] @ points3d_cam.T).T
            points2d_cam = points2d_cam[:, :2] / points2d_cam[:, 2:3]
            # we only retain valid points in each view
            try:
                valid_idx = (0 <= points2d_cam[:, 0]) * (points2d_cam[:, 0] < image_w) * (0 <= points2d_cam[:, 1]) * (points2d_cam[:, 1] < image_h) * (points3d_cam[:, 2] > 0.5)
                points3d_valid = points3d_cam[valid_idx, :]
                valid_idx_dict[i] = valid_idx

                depth_min = max(0.5, np.min(points3d_valid[:, -1]))
                depth_max = max(10.0, np.max(points3d_cam[:, -1]))
            except:
                depth_min = 0.5
                depth_max = 10.0
            depth_interval = (depth_max - depth_min) / (args.max_d - 1) / args.interval_scale
            depth_ranges[i] = (depth_min, depth_interval, args.max_d, depth_max)

        # view selection
        # center_point = np.mean(points3d, axis=0)
        score = np.zeros((len(cameras["frames"]), len(cameras["frames"])))
        queue = []
        for i in range(len(cameras["frames"])):
            for j in range(i + 1, len(cameras["frames"])):
                queue.append((i, j))


        def calc_score(inputs):
            i, j = inputs
            cam_center_i = c2ws[i][:3, 3]  # [3,]
            cam_center_j = c2ws[j][:3, 3]
            # find co-valid points for i, j views
            valid_pos_i = np.where(valid_idx_dict[i])[0]
            valid_pos_j = np.where(valid_idx_dict[j])[0]
            valid_idx = np.array(list(set(valid_pos_i) & set(valid_pos_j)))
            try:
                valid_points = points3d[valid_idx, :]
                # random downsample from valid_points
                valid_points_sampled = np.random.choice(np.arange(valid_points.shape[0]), size=min(200, valid_points.shape[0]), replace=False)
                valid_points_sampled = valid_points[valid_points_sampled, :]
                score = 0
                for pid in range(valid_points_sampled.shape[0]):
                    p = valid_points_sampled[pid]
                    theta = (180 / np.pi) * np.arccos(np.dot(cam_center_i - p, cam_center_j - p) / np.linalg.norm(cam_center_i - p) / np.linalg.norm(cam_center_j - p))
                    score += np.exp(-(theta - args.theta0) * (theta - args.theta0) / (2 * (args.sigma1 if theta <= args.theta0 else args.sigma2) ** 2))
            except:
                score = 0
            return i, j, score


        print("optimize for view selection...")
        p = mp.Pool(processes=mp.cpu_count())
        result = p.map(calc_score, queue)

        for i, j, s in result:
            score[i, j] = s
            score[j, i] = s
        view_sel = []
        for i in range(len(cameras["frames"])):
            sorted_score = np.argsort(score[i])[::-1]
            view_sel.append([(k, score[i, k]) for k in sorted_score[:10]])

        cam_dir = os.path.join(args.dense_folder, 'cams')
        os.makedirs(cam_dir, exist_ok=True)

        for i in range(len(cameras["frames"])):
            rescale_K = all_Ks[i].copy()
            with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
                f.write('extrinsic\n')
                w2c = np.linalg.inv(c2ws[i] @ rot)
                for j in range(4):
                    for k in range(4):
                        f.write(str(w2c[j, k]) + ' ')
                    f.write('\n')
                f.write('\nintrinsic\n')
                for j in range(3):
                    for k in range(3):
                        f.write(str(rescale_K[j, k]) + ' ')
                    f.write('\n')
                f.write('\n%f %f %f %f\n' % (depth_ranges[i][0], depth_ranges[i][1], depth_ranges[i][2], depth_ranges[i][3]))

        with open(os.path.join(args.dense_folder, 'pair.txt'), 'w') as f:
            f.write('%d\n' % len(cameras["frames"]))
            for i, sorted_score in enumerate(view_sel):
                f.write('%d\n%d ' % (i, len(sorted_score)))
                for image_id, s in sorted_score:
                    f.write('%d %f ' % (image_id, s))
                f.write('\n')

        with open(f"{args.dense_folder}/image_map.txt", "w") as w:
            for i in range(len(cameras["frames"])):
                w.write(cameras["frames"][i]["file_path"].split("/")[-1] + "\n")
