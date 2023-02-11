# WaveGAN
## overview
It is an implementation of "Wavegan" method that generates audio by machine learning.  
Reference:https://arxiv.org/abs/1802.04208  
Commentary article:https://qiita.com/zassou65535/items/5a9d5ef44dedea94be8a  

## Assumed environment
Ubuntu20.04  
python 3.7.1  
`pip install -r requirements.txt`You can prepare the environment. 

## program
* `WaveGAN_train.py`Is a program that runs learning and outputs the process and results. 
* In learning, the audio (`.wav` format) is selected from the dataset for each itelation, and the learning is cut from random parts for about 4 seconds. 
* `Wavegan_inference.py` is a program that reads the learning results (weight) output by` wavegan_train.py` into GENERATOR and output audio data. 
	* The output WAV file is a voice of about 4 seconds. 

## 使い方
1. Create `./DataSet` directory in a directory with` wavegan_train.py`
1. `./DataSet` In the directory, put the audio file you want to use for learning in the form of` ./dataSet/**/*. Wav`
1. Execute `python wavegan_train.py` in the directory with` wavegan_train.py` and start learning
	* The learning process is output to `./Output/Train/`
	* The learning result is output as `./Output/Generator_trained_model_cpu.pth`
1. Run `python wavegan_infernce.py` in the directory of` wavegan_inference.py` and infer
	* The reasoning result is output to `./Output/Inference/`
	* Note that if there is `./Output/Generator_trained_model_cpu.pth` (learned model), it will be an error.

Learning may take more than 12 hours depending on the environment.   
