from abc import ABCMeta, abstractmethod
import sys
sys.path.append("..")
from constants import *

class Bot(metaclass=ABCMeta):
	@abstractmethod
	def act(self, xdif, ydif, vel):
		pass

	@abstractmethod
	def dead(self, score):
		pass

	@abstractmethod
	def stop(self):
		pass
