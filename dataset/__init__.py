from .glue import GLUEDataModule, GLUE_TASKS
from .superglue import SuperGLUEDataModule, SUPERGLUE_TASKS


__all__ = ['GLUEDataModule', 'GLUE_TASKS', 'SuperGLUEDataModule', 'SUPERGLUE_TASKS']


def get_data_module(args):
    if args.task in GLUE_TASKS:
        dm_class = GLUEDataModule
    elif args.task in SUPERGLUE_TASKS:
        dm_class = SuperGLUEDataModule
    else:
        raise ValueError(f"Task {args.task} is not supported")

    dm = dm_class(
        model_name_or_path=args.model_name,
        task_name=args.task,
        max_seq_length=args.max_seq_length,
        sample_size=args.samplesize,
        train_batch_size=args.batchsize,
        validation_sample_size=args.samplesize_validation,
        eval_batch_size=args.batchsize,
        soft_prompt=args.soft_prompt,
        hf_token=args.hf_token
    )
    dm.setup(stage='fit')

    return dm
