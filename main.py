import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    folder_path = input("Please enter the input folder path = ")

    #Given parameters
    sensor_size = (11.3, 7.1)
    focal_length = 8.2
    sensor_resolution = (1920, 1200)
    camera_distance_origin = 420
    total_camera = 4

    # calculating field of view (FoV)
    fov_x = 2 * np.arctan(sensor_size[0] / (2 * focal_length))
    fov_y = 2 * np.arctan(sensor_size[1] / (2 * focal_length))

    #Calculating "extent" == dimensions of the 3D space that wanting to reconstruct
    total_extent = max(2 * camera_distance_origin * np.tan(fov_x / 2), 2 * camera_distance_origin * np.tan(fov_y / 2))

    # Calculating voxel size
    voxel_size = min(total_extent / sensor_resolution[0], total_extent / sensor_resolution[1])

    # Calculating voxel resolution 
    voxel_resolution = int((total_extent / voxel_size)/20)

    masked_frames = read_masked_images(folder_path)
    calibration_matrices = calculate_calibration_matrices(sensor_size, focal_length, sensor_resolution, camera_distance_origin)
    voxel_grid = create_voxel_grid(calibration_matrices, masked_frames, total_camera, total_extent, voxel_resolution)
    
    plot_voxel_grid(voxel_grid, voxel_resolution, total_extent, voxel_size)



def read_masked_images(folder_path):
    file_list = os.listdir(folder_path)
    print(file_list)
    image_files = sorted([file for file in file_list if "_msk.jpg" in file], key=lambda x: int(x.split("_cam")[1][0]))
    masked_frames = []

    threshold_value = 128
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # images are 0 or 255 (black or white)
        _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        masked_frames.append(thresholded_image)
    
    return masked_frames

def calculate_calibration_matrices(sensor_size, focal_length, sensor_resolution, camera_distance_origin):
    
    fx = focal_length * sensor_resolution[0] / sensor_size[0] # Focal length in "pixel" --> by similarity
    fy = focal_length * sensor_resolution[1] / sensor_size[1]
    cx = sensor_resolution[0] / 2 # "Principal point" is at the "center of the image".
    cy = sensor_resolution[1] / 2

     # Intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    # Rotation matrices for each camera for extrinsic matrix
    rotation_matrices = [
        # Top camera: rotate 180 degrees around the x-axix. (Look in the negative z-direction)
        np.array([[1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]]),
        # Front camera: rotate -90 degrees around the x-axis. (look in the negative y-direction)
        np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, -1, 0]]),
        # back camera: rotate 90 degrees around the x-axis. (look in the positive y-direction)
        np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]]),
        # bottom camera: No rotation. look in the positive z-direction
        np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    ]
    # Translation vectors for each camera
    translation_vectors = [
        np.array([0, 0, -camera_distance_origin]), # Top camera
        np.array([0, -camera_distance_origin, 0]), # Front camera
        np.array([0, camera_distance_origin, 0]), # Back camera
        np.array([0, 0, camera_distance_origin]) # Bottom camera
    ]

    # Construct the calibration matrices for each camera
    calibration_matrices = []
    for camera_index in range(4):
        R = rotation_matrices[camera_index]
        t = translation_vectors[camera_index].reshape(-1, 1)
        
        # Construct the extrinsic matrix from R and t
        extrinsic_matrix = np.hstack([R, t])
        extrinsic_matrix = np.vstack([extrinsic_matrix, [0, 0, 0, 1]])
        
        # Multiply intrinsic and extrinsic matrices for the calibration matrix
        calibration_matrix = np.dot(K, extrinsic_matrix[:3, :])
        calibration_matrices.append(calibration_matrix)

    return calibration_matrices


def create_voxel_grid(calibration_matrices, masked_frames, total_camera, total_extent, voxel_resolution):
    
    # Create 3D grid of voxel centers
    x = np.linspace(-total_extent / 2, total_extent / 2, voxel_resolution)
    y = np.linspace(-total_extent / 2, total_extent / 2, voxel_resolution)
    z = np.linspace(-total_extent / 2, total_extent / 2, voxel_resolution)

    #multi-dimensional array
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # flatten grids and combine them vertically
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel(), np.ones(grid_x.size)])

    voxel_detection_count = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))

    # Checking for each camera
    for camera_index in range(total_camera):

        # getting cameras' frames and calibration matrices 
        frame = masked_frames[camera_index]
        calibration = calibration_matrices[camera_index]

         # Transform world coordinates to image coordinates
        image_coordinates_homogeneous = np.dot(calibration, grid_points)
        image_coordinates = image_coordinates_homogeneous[:2] / image_coordinates_homogeneous[2]

        # Reshape back to 3D grid
        pixel_x = (image_coordinates[0].reshape(voxel_resolution, voxel_resolution, voxel_resolution)).astype(int)
        pixel_y = (image_coordinates[1].reshape(voxel_resolution, voxel_resolution, voxel_resolution)).astype(int)

        # Check if the coordinates are within the frame and the corresponding pixel is 255
        bounds_x = (0 <= pixel_x) & (pixel_x < frame.shape[1])
        bounds_y = (0 <= pixel_y) & (pixel_y < frame.shape[0])
        valid_pixels = bounds_x & bounds_y
        
            
        # Increment the detection count for voxels where the pixel value is 255
        voxel_detection_count[valid_pixels] += (frame[pixel_y[valid_pixels], pixel_x[valid_pixels]] == 255)

    # Create the final voxel grid. Voxel which are detected by least 2 cameras
    voxel_grid = voxel_detection_count >= 2
    return voxel_grid

def plot_voxel_grid(voxel_grid, voxel_resolution, total_extent, voxel_size):

    # count filled voxels and calculate the each voxel's volume
    filled_voxel_count = np.sum(voxel_grid)

    # Calculate object's volume and transform cm^3
    object_volume = int(filled_voxel_count * voxel_size / 1000)
    print("Volume of the object (integer type): ", object_volume)

    # 3d plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # getting coordinates of filled voxels
    filled_voxels = np.where(voxel_grid == 1)
    ax.scatter(filled_voxels[0], filled_voxels[1], filled_voxels[2],
               marker='o', s=50 , c='blue', alpha=0.5)


    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.show()

if __name__ == "__main__":
    main()
