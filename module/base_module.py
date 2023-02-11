#encoding:utf-8

from .importer import *

class PhaseShuffle(nn.Module):
	#Definition of layers to perform phaseshuffle
	def __init__(self,n):
		super().__init__()
		self.n = n#The range of how to shift is defined as [-n, n] in the paper.

	def forward(self, x):
		#If N is 0, it is equivalent to phaseshuffle in the first place
		if self.n == 0:
			return x
		#The integer belonging to [-n, n] is randomly generated and shift
		shift = torch.Tensor(x.shape[0]).random_(-self.n,self.n+1).type(torch.int)
		#Store the result of applying phaseshuffle to X in x_shuffled and return it as a return value.
		x_shuffled = x.clone()
		for i,shift_num in enumerate(shift):
			if(shift_num==0): continue
			dim = len(x_shuffled[i].size()) - 1
			origin_length = x[i].shape[dim]
			if shift_num > 0:
				left = torch.flip(torch.narrow(x[i],dim,1,shift_num),[dim])
				right = torch.narrow(x[i],dim,0,origin_length-shift_num)
			else:
				shift_num = -shift_num
				left = torch.narrow(x[i],dim,shift_num,origin_length-shift_num)
				right = torch.flip(torch.narrow(x[i],dim,origin_length-shift_num-1,shift_num),[dim])
			x_shuffled[i] = torch.cat([left,right],dim)

		return x_shuffled

#Functions that require the "Gradient_penalty" function required for calculating the gradient constraints of Discripor
#In WGAN-GP, the loss function of DISRIMINATOR is represented as E [Judgment Results of Real Voice] -E [Judgment Results of False Vehical Vehicle]+gradient constraints.
#In Generator, it is described as E [Judgment Results of False Voice]
def gradient_penalty(netD,real,fake,batch_size,gamma=1):
	device = real.device
	#For Tensor where Requires_grad is valid, the backward method can be called and can automatically calculate the differentiation.
	alpha = torch.rand(batch_size,1,1,requires_grad=True).to(device)
	#Mix the real and fake at a random ratio
	x = alpha*real + (1-alpha)*fake
	#Put it in Discriminator and make the result D_
	d_ = netD.forward(x)
	#Output D_ and input x
	#It is known that if the L2 norm calculated from the inclination becomes 1, it will produce good results.
	#Therefore, calculate Gradient_penalty so that this can be learned so that this approaches 1
	g = torch.autograd.grad(outputs=d_, inputs=x,
							grad_outputs=torch.ones(d_.shape).to(device),
							create_graph=True, retain_graph=True,only_inputs=True)[0]
	g = g.reshape(batch_size, -1)
	return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()


