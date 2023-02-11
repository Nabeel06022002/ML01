#encoding:utf-8

from .importer import *

def make_datapath_list(target_path):
	#Read the dataset
	path_list = []#Create a list of dataset file paths and return
	for path in glob.glob(target_path,recursive=True):
		path_list.append(path)
		##If you need to display all reading paths, remove the comment out
		#print(path)
	#Display the number of audio data to be read
	print("sounds : " + str(len(path_list)))
	return path_list

class GAN_Sound_Dataset(data.Dataset):
	#Voice dataset class
	def __init__(self,file_list,device,batch_size,sound_length=65536,sampling_rate=16000,dat_threshold=1100):
		#file_list     : List of voice paths to read
		#device        : Decide whether to process with GPU
		#batch_size    : Batch size
		#sound_length  : Length of sound used for learning
		#sampling_rate : Sampling rate when reading audio
		#dat_threshold : If the total number of files in the dataset is below Dat_threshold, hold the file content
		self.file_list = file_list
		self.device = device
		self.batch_size = batch_size
		self.sound_length = sound_length
		self.sampling_rate = sampling_rate
		self.dat_threshold = dat_threshold
		#If the total number of files in the dataset is below Dat_threshold, hold the file content
		if(len(self.file_list)<=dat_threshold):
			self.file_contents = []
			for file_path in self.file_list:
				#Sound is Numpy.ndarray, and the data of the chronological sound is stored.
				sound,_ = librosa.load(file_path,sr=self.sampling_rate)
				self.file_contents.append(sound)

	#Returns the larger batch size and the total number of files
	def __len__(self):
		return max(self.batch_size, len(self.file_list))
	#Get data in Tensor format with pre -processed audio
	def __getitem__(self,index):
		if(len(self.file_list)<=self.dat_threshold):
			sound = self.file_contents[index%len(self.file_list)]
		else:
			#Take out one from the list of paths
			sound_path = self.file_list[index%len(self.file_list)]
			#Sound is Numpy.ndarray, and the data of the chronological sound is stored.
			sound,_ = librosa.load(sound_path,sr=self.sampling_rate)
		#Convert to Tensor format
		sound = (torch.from_numpy(sound.astype(np.float32)).clone()).to(self.device)
		#If there is an element that is larger than 1 in the time series sound data, it is normalized so that it will be 1.
		max_amplitude = torch.max(torch.abs(sound))
		if max_amplitude > 1:
			sound /= max_amplitude
		#Make the length of the loaded sound as LOADED_SOUND_LENGTH
		loaded_sound_length = sound.shape[0]
		#If the length of the loaded sound is below Sound_length,
		#Fill the front and rear of the sound by 0 and align the length to Self.sound_length
		if loaded_sound_length < self.sound_length:
			padding_length = self.sound_length - loaded_sound_length
			left_zeros = torch.zeros(padding_length//2).to(self.device)
			right_zeros = torch.zeros(padding_length - padding_length//2).to(self.device)
			sound = torch.cat([left_zeros,sound,right_zeros],dim=0).to(self.device)
			loaded_sound_length = self.sound_length
		#Choose a random part from the readable sound for the length of the sound used for learning and cut it out.
		if loaded_sound_length > self.sound_length:
			#Select the starting point randomly
			start_index = torch.randint(0,(loaded_sound_length-self.sound_length)//2,(1,1))[0][0].item()
			end_index = start_index + self.sound_length
			sound = sound[start_index:end_index]
		#At this point, Sound.shape is TORCH.SIZE ([3, Self.Sound_length]),
		#Convert this to torch.size ([3, 1, self.sound_length])
		sound = sound.unsqueeze(0)
		return sound

#Produced voice output function
def save_sounds(path,sounds,sampling_rate):
	now_time = time.time()
	for i,sound in enumerate(sounds):
		sound = sound.squeeze(0)
		sound = sound.to('cpu').detach().numpy().copy()
		hash_string = hashlib.md5(str(now_time).encode()).hexdigest()
		file_path = os.path.join(path,f"generated_sound_{i}_{hash_string}.wav")
		print(file_path)
		sf.write(file_path,sound,sampling_rate,format="WAV")

#Operation confirmation
# train_wav_list = make_datapath_list('../dataset/**/*.wav')

# batch_size = 3
# dataset = GAN_Sound_Dataset(file_list=train_wav_list,device="cpu",batch_size=batch_size)

# dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

# batch_iterator = iter(dataloader)
# sounds = next(batch_iterator)
# save_sounds(sounds,16000)




