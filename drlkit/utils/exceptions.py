class AgentMissing(Exception):
	""" Raised when play() is called before training or loading the agent"""
	message = "No agent was initialized"
	
	def __str__(self):
		return AgentMissing.message