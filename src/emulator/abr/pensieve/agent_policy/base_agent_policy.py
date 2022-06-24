class BaseAgentPolicy:

    def select_action(self, state):
        raise NotImplementedError

    def evaluate(self, net_env):
        raise NotImplementedError
