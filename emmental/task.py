import logging

from torch import nn

logger = logging.getLogger(__name__)

class Operation:
    """A single operation to execute in a task flow

    The `name` attributes defaults to `module_name` since most of the time, each module
    is used only once per task flow. For more advanced flows where the same module is
    used multiple times per forward pass, a name may be explicitly given to
    differentiate the Operations.
    """

    def __init__(self, module_name, inputs, name=None):
        self.name = name or module_name
        self.module_name = module_name
        self.inputs = inputs

    def __repr__(self):
        return (
            f"Operation(name={self.name}, "
            f"module_name={self.module_name}, "
            f"inputs={self.inputs}"
        )
        
class EmmentalTask(object):
    """Task class to define task in Emmental model.

    :param name: The name of the task (Primary key).
    :type name: str
    :param module_pool: A dict of modules that uses in the task.
    :type module_pool: nn.ModuleDict
    :param task_flow: The task flow among modules to define how the data flows.
    :type task_flow: list
    :param loss_func: The function to calculate the loss.
    :type loss_func: function
    :param output_func: The function to generate the output.
    :type output_func: function
    :param scorer: The class of metrics to evaluate the task.
    :type scorer: Scorer class
    """

    def __init__(self, name, module_pool, task_flow, loss_func, output_func, scorer):
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer

        logger.info(f"Created task: {self.name}")

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
