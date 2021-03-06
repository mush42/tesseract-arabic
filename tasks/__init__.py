# coding: utf-8

from invoke import task, Collection
from . import training_tasks

ns = Collection()
ns.add_collection(training_tasks, name="just")
ns.add_task(training_tasks.clean)
ns.add_task(training_tasks.done)
ns.add_task(training_tasks.import_files)
ns.add_task(training_tasks.evaluate)


@ns.add_task
@task(pre=[
    training_tasks.clean,
    training_tasks.extract,
    training_tasks.gen_images,
    training_tasks.gen_unicharset,
    training_tasks.lang_dict,
    training_tasks.proto,
    training_tasks.gen_lstmf,
    training_tasks.mv_lstmf,
    training_tasks.write_lstmf_files_list,
])
def train(c, *args, **kwargs):
    return training_tasks.train(c, *args, **kwargs)


@ns.add_task
@task(pre=[
    training_tasks.extract,
    training_tasks.gen_unicharset,
    training_tasks.lang_dict,
    training_tasks.proto,
    training_tasks.write_lstmf_files_list,
])
def train_ready(c, *args, **kwargs):
    return training_tasks.train(c, *args, **kwargs)


@ns.add_task
@task
def cont_training(c, *args, **kwargs):
    return training_tasks.train(c, cont=True)
