from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def get_action_mapping(self):
        pass

    @abstractmethod
    def get_action(self, action_check_fn=None, last_action=None):
        pass

    @abstractmethod
    def optimize(self, samples, optimizer, criterion, config):
        pass

    @abstractmethod
    def save(self, model_save_path, optimizer, lr_schedule, config, step_count):
        pass