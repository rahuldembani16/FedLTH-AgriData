class LR_UPDATE():
	def __init__(self, init_lr, decrease_rate, min_lr, decrease_frequency: int):
		"""
		Initialize the learning rate updater.

		Parameters:
		init_lr (float): Initial learning rate.
		decrease_rate (float): Rate at which the learning rate decreases.
		min_lr (float): Minimum learning rate.
		decrease_frequency (int): Frequency (in steps) at which the learning rate decreases.
		"""
		self.counter = 0
		self.init_lr = init_lr
		self.lr = init_lr
		self.decrease_rate = decrease_rate
		self.min_lr = min_lr
		self.decrease_frequency = decrease_frequency

	def __call__(self, reset=False):
		"""
		Update the learning rate.

		Parameters:
		reset (bool): If True, reset the learning rate to the initial value.

		Returns:
		float: The updated learning rate.
		"""
		self.counter += 1
		if reset:
			self.lr = self.init_lr
			self.counter = 0
			return self.lr
		if self.counter % self.decrease_frequency == 0:
			self.lr = max(self.min_lr, self.lr * (1. - self.decrease_rate))
		return self.lr
	
def get_model_bits(model):
    state_dict = model.state_dict()
    total_bits = 0
    for param_name, param_tensor in state_dict.items():
      total_bits += param_tensor.numel() * param_tensor.element_size() * 8
    return total_bits