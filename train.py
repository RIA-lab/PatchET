import os
import argparse
from transformers import TrainingArguments
from model_trainer import ModelTrainer
import numpy as np
import wandb
from models import load_model
from dataset_temperature import load_dataset
from torch.optim import AdamW
from utils import (load_config,
                   save_config,
                   load_weight,
                   freeze_model,
                   count_parameters,
                   write_json,
                   scatter_plot_with_density,
                   interval_evaluation_opt,
                   interval_evaluation_stability,
                   plot_interval_evaluation,
                   load_metrics,
                   overlap_ratio)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--run_config', type=str, help='Path to the YAML config file')
    parser.add_argument('--model_config', type=str, help='Path to the YAML config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_config = load_config(args.run_config)

    task = run_config['data_path'].split('/')[-1]
    ver = run_config['data_path'].split('/')[1]
    Dataset = load_dataset(task)
    metrics = load_metrics(task)

    dataset_train = Dataset(os.path.join(run_config['data_path'], 'train.csv'))
    dataset_val = Dataset(os.path.join(run_config['data_path'], 'val.csv'))
    dataset_test = Dataset(os.path.join(run_config['data_path'], 'test.csv'))

    print(f'train: {len(dataset_train)}')
    print(f'val: {len(dataset_val)}')
    print(f'test: {len(dataset_test)}')

    model_config = load_config(args.model_config)
    Model, Collator = load_model(model_config['name'])

    model_name = args.model_config.split('/')[-1].split('.')[0][:-6] if model_config['name'].endswith('_range') else \
    args.model_config.split('/')[-1].split('.')[0]
    run_name = f'{ver}_{task}_{model_name}'
    output_dir = f'output/{ver}/{task}/{model_name}'
    results_dir = f'results/{ver}/{task}/{model_name}'

    # # transfer learning
    # if task == 'range' and 'segment' in model_config['name']:
    #     run_name += '_transfer_learning'
    #     output_dir += '_transfer_learning'
    #     results_dir += '_transfer_learning'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #save model_confis as a yaml file in the output directory
    save_config(model_config, f'{output_dir}/model_config.yaml')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if 'reda' in model_config['name']:
        model_config['task'] = task
        model_config['aug_data'] = run_config['data_path']
        model_config['aug_db_output_path'] = output_dir

    model = Model(model_config)
    if not task == 'range':
        temp_ranges = dataset_train.default_ranges
        weights = dataset_train.calculate_weights()
        model.loss_fct.set_ranges_and_weights(temp_ranges, weights)

    collate_fn = Collator(model_config['pretrain_model'])
    try:
        freeze_model(model.pretrain_model)
    except:
        print('No pretrain model to freeze')
    print(f'trainable parameters: {round(count_parameters(model) / 1000000, 2)}M')
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(run_config['training']['lr']))

    wandb.init(project='Thermal parameter prediction')



    # # transfer learning
    # if task == 'range' and 'segment' in model_config['name']:
    #     load_weight(model, f'output/{ver}/opt/segment_dgsa2_s2/checkpoint-3168/model.safetensors')


    wandb.run.name = run_name
    print(f'wandb run name: {run_name}')

    train_batch_size = run_config['training']['train_batch_size']
    steps_per_epoch = len(dataset_train) // train_batch_size
    eval_every_n_epochs = 8
    eval_steps = steps_per_epoch * eval_every_n_epochs
    save_limit = 48 / eval_every_n_epochs

    args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=f'{output_dir}/log',
        logging_strategy='steps',  # log every few steps
        logging_steps=steps_per_epoch,  # log once per epoch (optional)
        save_strategy="steps",  # <--- Match here
        save_steps=eval_steps,
        learning_rate=float(run_config['training']['lr']),
        per_device_train_batch_size=run_config['training']['train_batch_size'],
        per_device_eval_batch_size=run_config['training']['eval_batch_size'],
        num_train_epochs=run_config['training']['num_epochs'],
        weight_decay=float(run_config['training']['weight_decay']),
        evaluation_strategy="steps",  # <-- key change
        eval_steps=eval_steps,
        dataloader_num_workers=run_config['training']['dataloader_num_workers'],
        dataloader_pin_memory=run_config['training']['dataloader_pin_memory'],
        run_name=wandb.run.name,
        overwrite_output_dir=True,
        # save_total_limit=run_config['training']['save_total_limit'],
        save_total_limit=10,
        remove_unused_columns=False,
        report_to=["wandb"],
        fp16=run_config['training']['fp16'],
        max_grad_norm=run_config['training']['max_grad_norm'],
        load_best_model_at_end=True,
    )

    trainer = ModelTrainer(
        model=model,
        optimizers=(optimizer, None),
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        test_dataset=dataset_test,
        data_collator=collate_fn,
        compute_metrics=metrics,
    )

    trainer.train(resume_from_checkpoint=False)



    datasets = {'train': dataset_train, 'val': dataset_val, 'test': dataset_test}
    for k, v in datasets.items():
        print(f'------------------------{k}--------------------------')
        predictions, labels, metrics = trainer.predict(v)

        write_json(metrics, f'{results_dir}/{k}_metrics.json')

        if k == 'test':
            np.save(f'{results_dir}/predictions.npy', predictions)
            np.save(f'{results_dir}/labels.npy', labels)

            if task == 'range':
                ratio = overlap_ratio(labels, predictions, save_dir=f'{results_dir}')
                print(f'overlap ratio: {ratio}')
                scatter_plot_with_density(labels[:, 0], predictions[:, 0], xlabel='experimental temperature values',
                                          ylabel='predicted temperature values', save_dir=f'{results_dir}',
                                          title='Scatter_plot_for_low_temperature')
                scatter_plot_with_density(labels[:, 1], predictions[:, 1], xlabel='experimental temperature values',
                                            ylabel='predicted temperature values', save_dir=f'{results_dir}',
                                            title='Scatter_plot_for_high_temperature')
            else:
                scatter_plot_with_density(labels, predictions, xlabel='experimental temperature values',
                                          ylabel='predicted temperature values', save_dir=f'{results_dir}')

                if task == 'opt':
                    metrics_interval = interval_evaluation_opt(labels, predictions)
                elif task == 'stability':
                    metrics_interval = interval_evaluation_stability(labels, predictions)
                write_json(metrics_interval, f'{results_dir}/{k}_metrics_interval.json')
                plot_interval_evaluation(metrics_interval, save_dir=f'{results_dir}')

    wandb.finish()