import imageio

file_path = '/home/wangzimo/VTT/IsaacLab/source/VTT/camera_output/frames/'

with imageio.get_writer(uri='/home/wangzimo/VTT/IsaacLab/source/VTT/camera_output/test.gif', mode='I', fps=25) as writer:
    for i in range(99):
        if i:
            writer.append_data(imageio.imread(file_path + f'{i}.png'))

print("GIF Generate Complete!")