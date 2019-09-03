from functools import partial

import torch.nn.functional as F
from emmental.scorer import Scorer
from emmental.task import EmmentalTask, Operation
from modules.bert_module import BertModule, BertLastCLSModule
from task_config import LABEL_MAPPING, METRIC_MAPPING
from torch import nn
from torch.nn import MSELoss


def ce_loss(task_name, immediate_ouput_dict, Y, active):
    module_name = f"{task_name}_pred_head"
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def mse_loss(task_name, immediate_ouput_dict, Y, active):
    mse = MSELoss()
    module_name = f"{task_name}_pred_head"
    return mse(
        immediate_ouput_dict[module_name][0][active].view(-1), Y[active].view(-1)
    )


def output(task_name, immediate_ouput_dict):
    module_name = f"{task_name}_pred_head"
    return immediate_ouput_dict[module_name][0]


def get_gule_task(task_names, bert_model_name, last_hidden_dropout_prob=0.0):

    tasks = dict()

    bert_module = BertModule(bert_model_name)
    bert_output_dim = 768 if "base" in bert_model_name else 1024

    for task_name in task_names:
        task_cardinality = (
            len(LABEL_MAPPING[task_name].keys())
            if LABEL_MAPPING[task_name] is not None
            else 1
        )

        metrics = METRIC_MAPPING[task_name]

        if task_name == "STS-B":
            loss_fn = partial(mse_loss, task_name)
        else:
            loss_fn = partial(ce_loss, task_name)

        task = EmmentalTask(
            name=task_name,
            module_pool=nn.ModuleDict(
                {
                    "bert_module": bert_module,
                    f"{task_name}_feature": BertLastCLSModule(
                    dropout_prob=last_hidden_dropout_prob
                    ),
                    f"{task_name}_pred_head": nn.Linear(bert_output_dim, task_cardinality),

                }
            ),
            task_flow=[
                Operation(
                name=f"{task_name}_bert_module",
                module_name="bert_module",
                inputs=[
                    ("_input_", "token_ids"),
                    ("_input_", "token_segments"),
                    ("_input_", "token_masks"),
                       ],
                ),
            Operation(
                name=f"{task_name}_feature",
                module_name=f"{task_name}_feature",
                inputs=[(f"{task_name}_bert_module", 0)],
            ),
            Operation(
                name=f"{task_name}_pred_head",
                module_name=f"{task_name}_pred_head",
                inputs=[(f"{task_name}_feature", 0)],
            ),
            ],
            loss_func=loss_fn,
            output_func=partial(output, task_name),
            scorer=Scorer(metrics=metrics),
        )

        tasks[task_name] = task

    return tasks
