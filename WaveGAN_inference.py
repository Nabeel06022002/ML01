#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#Number of audio files to be output
sample_size = 16
#The size of the random number to enter
z_dim = 20
#Sampling rate of voice to handle
sampling_rate = 16000

#Read the learned model
netG = Generator(z_dim=z_dim)
trained_model_path = "./output/generator_trained_model_cpu.pth"
netG.load_state_dict(torch.load(trained_model_path))
#Switch to inference mode
netG.eval()
#Noise generation
noise = torch.Tensor(sample_size,z_dim).uniform_(-1,1)
#Enter GENERATOR and get output image
generated_sound = netG(noise)
#Create if there is no output directory
output_dir = "./output/inference"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)
#Output of audio file
save_sounds("./output/inference/",generated_sound,sampling_rate)
