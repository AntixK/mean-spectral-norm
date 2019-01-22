import imageio

def create_gif(img_path, gif_name, out_path = None, num_imgs = None):
    images = []

    for i in range(num_imgs):

        filename = img_path+'fake_samples_epoch_%03d'%i + '.png'
        print("Processing image : ", filename)
        images.append(imageio.imread(filename))

    if out_path is not None:
        imageio.mimsave(out_path+gif_name+'.gif', images)
    else:
        imageio.mimsave(gif_name+'.gif', images)

create_gif('./log/MSNGAN/','MSNGAN_results', './log/MSNGAN/', 400)
create_gif('./log/SNGAN/','SNGAN_results', './log/SNGAN/', 400)

