import imageio

file_path = '/home/wangzimo/VTT/IsaacLab/source/VTT/camera_output/frames/'

with imageio.get_writer(uri='/home/wangzimo/VTT/IsaacLab/source/VTT/camera_output/test.mp4', fps=25) as writer:
    for i in range(599):
        if i:
            writer.append_data(imageio.imread(file_path + f'{i}.png'))

print("Video Generate Complete!")