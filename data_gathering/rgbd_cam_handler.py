import os
import cv2
import numpy as np
import pyrealsense2 as rs
import json

def save_intrinsics(rgb_intrinsics, depth_intrinsics, output_folder:str = None, filename:str = None):
    rgb_camera_mat = np.array([[rgb_intrinsics.fx, 0, rgb_intrinsics.ppx], [0, rgb_intrinsics.fy, rgb_intrinsics.ppy], [0,0,1]])
    depth_camera_mat = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx], [0, depth_intrinsics.fy, depth_intrinsics.ppy], [0,0,1]])

    camera_data = {
        'rgb_camera_matrix': rgb_camera_mat.tolist(),
        'rgb_distortion_coefficients': rgb_intrinsics.coeffs,
        'depth_camera_matrix': depth_camera_mat.tolist(),
        'depth_distortion_coefficients': depth_intrinsics.coeffs,
    }
    if output_folder is not None and filename is not None:
        with open(f"{output_folder}/{filename}.json", 'w') as f:
            json.dump(camera_data, f, indent=4)




if __name__ == "__main__":
    colorizer = rs.colorizer()

    pipeline = rs.pipeline()
    pipeline.start()

    frame_idx = 0
    try:
        os.remove(f"rgbd_cam_output")
    except OSError as error:
        print(error)
        print("Old cam not removed")
    os.makedirs(f"rgbd_cam_output", exist_ok=True)


    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
    print(f"depth_scale = {depth_scale}")

    created_intrinsics = False

    try:
        while True:

            frames = pipeline.wait_for_frames()

            rgb_frame = np.asanyarray(frames.get_color_frame().get_data())
            depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
            depth_frame_scaled = depth_frame * depth_scale


            if not created_intrinsics:
                created_intrinsics = True
                rgb_intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                depth_intrinsics = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
                save_intrinsics(rgb_intrinsics=rgb_intrinsics, depth_intrinsics=depth_intrinsics, output_folder="rgbd_cam_output", filename = "robot_calibration")

            if not frames.get_depth_frame() or not frames.get_color_frame():
                continue

            cv2.imshow("Depth", np.asanyarray(colorizer.colorize(frames.get_depth_frame()).get_data()))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            os.makedirs(f"rgbd_cam_output/{frame_idx}", exist_ok=True)
            cv2.imwrite(f"rgbd_cam_output/{frame_idx}/rgb.png", cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
            np.save(f"rgbd_cam_output/{frame_idx}/depth.npy", depth_frame_scaled)

            print(f"min distance in meters: {np.quantile(depth_frame_scaled, 0.2)}, max: {np.quantile(depth_frame_scaled, 0.8)}")

            #print(f"rgb_frame_shape = {rgb_frame.shape}, depth_frame_shape = {depth_frame.shape}")

            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()