#encoding:utf-8

from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

#Formatting the path to each data to make_dataPath_list for datasets
dataset_path = './dataset/**/*.wav'
#Batch size
batch_size = 16
#The size of the random number to enter
z_dim = 20
#Number of epochs
num_epochs = 500
#Learning rate used for Optimizer
lr = 0.0001
#Input and output sound sampling rate
sampling_rate = 16000
#How many times to learn Discripor per study at GENERATOR
D_updates_per_G_update = 5
#Generate_sounds_interval [Epoch] Outputs the learning status every time you learn
generate_sounds_interval = 20

#Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

#Read training data, create dataset
train_sound_list = make_datapath_list(dataset_path)
train_dataset = GAN_Sound_Dataset(file_list=train_sound_list,device=device,batch_size=batch_size)
#generator用
dataloader_for_G = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
#discriminator用
dataloader_for_D = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

# #Functions for initializing networks
def weights_init(m):
	if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m,nn.Linear):
		nn.init.kaiming_normal_(m.weight.data)

#Generate Generator instance
netG = Generator(z_dim=z_dim)
#Move the network to the device
netG = netG.to(device)
#Initialization of network
netG.apply(weights_init)

#Generate DISCRIMINATOR instance
netD = Discriminator()
#Move the network to the device
netD = netD.to(device)
#Initialization of network
netD.apply(weights_init)

#Set the optimization method to ADAM
beta1 = 0.5
beta2 = 0.9
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,beta2))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,beta2))

#Start of learning
#Variables to follow the learning process
G_losses = []
D_losses = []
iters = 0
#Noise to enter Generator to follow the learning process
generating_num = 5#How many sounds do you want to output?
z_sample = torch.Tensor(generating_num,z_dim).uniform_(-1,1).to(device)

print("Starting Training")

#Save the learning start time
t_epoch_start = time.time()
#Loop for each epoch
for epoch in range(num_epochs):
	#BATCH_SIZE Take out and learn from the dataset
	for generator_i,real_sound_for_G in enumerate(dataloader_for_G, 0):
		#-------------------------
 		#Learning of DISCRIMINATOR
		#-------------------------
		#Loss function e [Real Voice Judgment Results] -E [Judgment Results of False Voice]+Learn to maximize the gradient constraints of the gradient
		#Learn D_Updates_per_g_Update times Discriminator per study of Generator
 		#-------------------------
		errD_loss_sum = 0#Variables for taking the average of losses during learning
		for discriminator_i,real_sound_for_D in enumerate(dataloader_for_D, 0):
			if(discriminator_i==D_updates_per_G_update): break
			#Number of audio data actually taken out
			minibatch_size = real_sound_for_D.shape[0]
			#If the number of mini batches taken out is 1, it will be an error in the process of finding the gradient, so skip the processing.
			if(minibatch_size==1): continue
			#If you can use GPU, transfer to GPU
			real_sound_for_D = real_sound_for_D.to(device)
			#Generate noise and make Z
			z = torch.Tensor(minibatch_size,z_dim).uniform_(-1,1).to(device)
			#Put noise in GENERATOR and generate fake sounds and make Fake_sound.
			fake_sound = netG.forward(z)
			#Judge the real sound and store the results in D
			d_real = netD.forward(real_sound_for_D)
			#Judge the false sound and store the result in D_
			d_fake = netD.forward(fake_sound)

			#Take the average of the judgment results for each mini batch
			loss_real = d_real.mean()#-E. Calculate [Real Voice Judgment Results]
			loss_fake = d_fake.mean()#-E. Calculate [Judgment Results of False Vehicle]
			#Calculation of gradient constraints
			loss_gp = gradient_penalty(netD,real_sound_for_D.data,fake_sound.data,minibatch_size)
			beta_gp = 10.0
			#E[Real audio judgment result] -E [Judgment result of false audio]+gradient constraints calculation
			errD = -loss_real + loss_fake + beta_gp*loss_gp
			#The inclination calculated in the previous itelation has remained, so reset it.
			optimizerD.zero_grad()
			#Calculate the inclination of the loss
			errD.backward()
			#Actually propagate errors
			optimizerD.step()
			#Record Loss to take the average later
			errD_loss_sum += errD.item()
		
		#-------------------------
 		#Generator learning
		#-------------------------
		#Loss function -E Learn to maximize [Judgment Results of False Voice]
 		#-------------------------
		#Number of audio data actually taken out
		minibatch_size = real_sound_for_G.shape[0]
		#If the number of mini batches taken out is 1, it will be an error in the process of finding the gradient, so skip the processing.
		if(minibatch_size==1): continue
		#If you can use GPU, transfer to GPU
		real_sound_for_G = real_sound_for_G.to(device)
		#Generate noise
		z = torch.Tensor(minibatch_size,z_dim).uniform_(-1,1).to(device)
		#Enter the noise into the Generator and make the output audio as Fake_sound.
		fake_sound = netG.forward(z)
		#Output audio fake_sound is inferred or fake sound in Discriminator
		d_fake = netD.forward(fake_sound)

		# WGAN_GP takes an average for all inference results in the mini batch and use it for erroneous propagation.
		errG = -d_fake.mean()#E Calculate the [Judgment Results of False Voice]
		#The inclination calculated in the previous itelation has remained, so reset it.
		optimizerG.zero_grad()
		#Calculate the inclination of the loss
		errG.backward()
		#Actually propagate errors
		optimizerG.step()

		#Record Loss to output to the graph later
		G_losses.append(errG.item())
		D_losses.append(errD_loss_sum/D_updates_per_G_update)

		iters += 1
		#Break for testing
		#break
	
	#Output the learning status
	if (epoch%generate_sounds_interval==0 or epoch==num_epochs-1):
		print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
				% (epoch, num_epochs,
					errD_loss_sum/D_updates_per_G_update, errG.item()))
		#Create if there is no output directory
		output_dir = "./output/train/generated_epoch_{}".format(epoch)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		#Output of generated audio
		with torch.no_grad():
			generated_sound = netG(z_sample)
			save_sounds(output_dir,generated_sound,sampling_rate)

#-------------------------
#Output of execution result
#-------------------------

#Output the time spent on learning
#Record the time at the end of learning
t_epoch_finish = time.time()
total_time = t_epoch_finish - t_epoch_start
with open('./output/train/time.txt', mode='w') as f:
	f.write("total_time: {:.4f} sec.\n".format(total_time))
	f.write("dataset size: {}\n".format(len(train_sound_list)))
	f.write("num_epochs: {}\n".format(num_epochs))
	f.write("batch_size: {}\n".format(batch_size))

#Output a learned Generator model (for CPU)
torch.save(netG.to('cpu').state_dict(),"./output/generator_trained_model_cpu.pth")

#Output Loss graph
plt.clf()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./output/train/loss.png')

print("data generated.")

