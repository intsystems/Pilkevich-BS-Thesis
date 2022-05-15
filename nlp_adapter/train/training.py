import os
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm, trange
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output

from ..utils import turn_on_grad, turn_off_grad, cleanup, load_model, set_seed, static_vars
from .model import LinearAdapter
from .dataset import PairsDataset, DataCollatorWithPadding, read_dataset


def classify_texts(model, tokenizer, texts,  batch_size=32, verbose=False):
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i:i+batch_size]]
        inputs = tokenizer(*inputs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)
        preds = torch.log_softmax(model(**inputs).logits, -1)
        res.append(preds)
    return torch.cat(res, axis=0)


def calculate_loss(model, batch, loss_model, adapter_model):
    output = model(**batch, output_hidden_states=True)
    ce_loss = output.loss

    mask_for_loss = batch['decoder_attention_mask']
    model_loss = torch.softmax(
        loss_model(
            inputs_embeds=adapter_model(output.logits), attention_mask=mask_for_loss

        ).logits,
        -1
    )[:, 1].mean()
    # TODO: для разных моделей разный аргумент

    return ce_loss, model_loss


def calculate_loss_for_adapter(model, batch, neu_tox, loss_model, adapter_model):
    output = model(**batch)

    mask_for_loss = batch['decoder_attention_mask']
    log_adapter_tox = torch.log_softmax(
        loss_model(
            inputs_embeds=adapter_model(output.logits), attention_mask=mask_for_loss

        ).logits,
        -1
    )

    kl_loss = F.kl_div(log_adapter_tox, neu_tox.to(model.device), log_target=True)
    return kl_loss


def evaluate_model(model, test_dataloader, loss_model=None, adapter_model=None):
    ce_num, ce_den = 0, 0
    model_num, model_den = 0, 0

    for batch in test_dataloader:
        with torch.no_grad():
            batch.pop('neutral_toxicity', None)
            batch['labels'][batch['labels'] == 0] = -100
            batch = {k: v.to(model.device) for k, v in batch.items()}

            ce_loss, model_loss = calculate_loss(model, batch, loss_model, adapter_model)
            ce_num += len(batch) * ce_loss.item()
            ce_den += len(batch)
            model_num += len(batch) * model_loss.item()
            model_den += len(batch)
    ce_val_loss = ce_num / ce_den
    model_val_loss = model_num / model_den
    return ce_val_loss, model_val_loss


@static_vars(cur_accum_step=0)
def model_optimize_step(
        model, batch, loss_model, adapter_model, step, model_optimizer,
        ce_w: float = 0.0, model_w: float = 1.0, mode: str = 'sum',
        num_accum_steps: int = 1,
):
    model.train()
    turn_on_grad(model)

    adapter_model.eval()
    turn_off_grad(adapter_model)

    try:
        ce_loss, model_loss = calculate_loss(model, batch, loss_model, adapter_model)
        if mode == 'sum':
            loss = model_w * model_loss + ce_w * ce_loss
        elif mode == 'prod':
            loss = model_loss * ce_loss
        else:
            raise ValueError('Доступно только две агрегации лосса: sum, prod')

        loss /= num_accum_steps
        loss.backward()
        model_optimize_step.cur_accum_step += 1
    except Exception as e:
        print('Model error on step', step, e)
        cleanup()
        return None, None

    if model_optimize_step.cur_accum_step == num_accum_steps:
        model_optimizer.step()
        model_optimizer.zero_grad()
        model_optimize_step.cur_accum_step = 0

    return ce_loss, model_loss


@static_vars(cur_accum_step=0)
def adapter_optimize_step(
        model, batch, neu_tox, loss_model, adapter_model, step, adapter_optimizer,
        num_accum_steps: int = 1,
):
    model.eval()
    turn_off_grad(model)

    adapter_model.train()
    turn_on_grad(adapter_model)

    try:
        kl_loss = calculate_loss_for_adapter(model, batch, neu_tox, loss_model, adapter_model)
        loss = kl_loss / num_accum_steps
        loss.backward()
        adapter_optimize_step.cur_accum_step += 1
    except Exception as e:
        print('Adapter error on step', step, e)
        cleanup()
        return None

    if adapter_optimize_step.cur_accum_step == num_accum_steps:
        adapter_optimizer.step()
        adapter_optimizer.zero_grad()
        adapter_optimize_step.cur_accum_step = 0

    return kl_loss


def visualization_callback(history: Dict[str, Dict[str, List]], name='tmp.pdf'):
    plt.figure(figsize=(18, 9), constrained_layout=False)
    num_plots = len(history)
    for i, key in enumerate(history):
        plt.subplot((num_plots + 1) // 2, 2, i + 1)
        for label in history[key]:
            plt.plot(history[key][label]['step'], history[key][label]['loss'], label=label)
        plt.legend()
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title(f'Loss {key}')

    plt.tight_layout()
    plt.savefig(name)


def is_model_turn(
        step: int,
        model_period_train: int,
        adapter_period_train: int,
):
    local_step = step % (model_period_train + adapter_period_train)
    if local_step // model_period_train == 0:
        return True
    return False


def train_loop(
        model, train_dataloader, val_dataloader,
        loss_model, adapter_model,
        max_epochs: int = 30,
        max_steps: int = 1_000,
        adapter_lr: float = 3e-5,
        model_lr: float = 3e-5,
        cleanup_step: int = 100,
        report_step: int = 1_000,
        window: int = 100,
        period_of_dump: int = 1_000,
        dump_model_name: str = 'tmp/base',
        ce_w: float = 1.0,
        model_w: float = 0.0,
        train_model: bool = True,
        train_adapter: bool = False,
        loss_mode: str = 'sum',
        num_accum_steps: int = 1,
        model_period_train: Optional[int] = None,
        adapter_period_train: Optional[int] = None,
):
    dir_path = os.path.join(*dump_model_name.split('/')[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cleanup()
    model_optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=model_lr,
    )
    adapter_optimizer = torch.optim.AdamW(
        params=adapter_model.parameters(),
        lr=adapter_lr,
    )

    ce_ewm_loss = 0
    model_ewm_loss = 0
    kl_ewm_loss = 0

    history = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    step = 0
    best_ce_loss = None
    ce_loss, model_loss, kl_loss = None, None, None

    for epoch in trange(max_epochs):
        if step >= max_steps:
            break
        tq = tqdm(train_dataloader)
        for i, batch in enumerate(tq):

            neu_tox = batch['neutral_toxicity'].detach().clone().to(model.device)
            batch.pop('neutral_toxicity', None)
            batch['labels'][batch['labels'] == 0] = -100

            batch = {k: v.to(model.device) for k, v in batch.items()}

            # TODO: добавить реальное чередование
            if train_model \
                    and (model_period_train is None or is_model_turn(step, model_period_train, adapter_period_train)):
                ce_loss, model_loss = model_optimize_step(
                    model, batch, loss_model, adapter_model, step, model_optimizer,
                    ce_w=ce_w, model_w=model_w, mode=loss_mode,
                    num_accum_steps=num_accum_steps,
                )
            if train_adapter \
                    and (adapter_period_train is None or not is_model_turn(step, model_period_train, adapter_period_train)):
                kl_loss = adapter_optimize_step(
                    model, batch, neu_tox, loss_model, adapter_model, step, adapter_optimizer,
                    num_accum_steps=num_accum_steps,
                )

            step += 1

            if step % period_of_dump == 0:
                if train_adapter:
                    dir_path = os.path.join(*dump_model_name.split('/')[:-1])
                    torch.save(adapter_model.state_dict(), os.path.join(dir_path, f'adapter_{step}.pt'))
                if train_model:
                    model.save_pretrained(dump_model_name + f'_{step}')

            if step >= max_steps:
                model.save_pretrained(dump_model_name + f'_{step}')
                break

            if i % cleanup_step == 0:
                cleanup()

            w = 1 / min(i + 1, window)

            info_to_display = ''
            if train_model and ce_loss is not None:
                ce_ewm_loss = ce_ewm_loss * (1 - w) + ce_loss.item() * w
                history['CE']['train']['loss'].append(ce_ewm_loss)
                history['CE']['train']['step'].append(step)
                info_to_display = info_to_display + f'CE: {ce_ewm_loss:4.4f}. '
            if train_model and model_loss is not None:
                model_ewm_loss = model_ewm_loss * (1 - w) + model_loss.item() * w
                history['model']['train']['loss'].append(model_ewm_loss)
                history['model']['train']['step'].append(step)
                info_to_display = info_to_display + f'model_loss: {model_ewm_loss:4.4f}. '
            if train_adapter and kl_loss is not None:
                kl_ewm_loss = kl_ewm_loss * (1 - w) + kl_loss.item() * w
                history['KL']['train']['loss'].append(kl_ewm_loss)
                history['KL']['train']['step'].append(step)
                info_to_display = info_to_display + f'KL: {kl_ewm_loss:4.4f}. '

            tq.set_description(info_to_display)

            if (i and i % report_step == 0 or i == len(train_dataloader) - 1) \
                    and val_dataloader is not None \
                    and train_model:
                model.eval()
                ce_eval_loss, model_eval_loss = evaluate_model(model, val_dataloader, loss_model, adapter_model)

                if train_model and (best_ce_loss is None or best_ce_loss > ce_eval_loss):
                    best_ce_loss = ce_eval_loss
                    model.save_pretrained(dump_model_name + f'_{step}')

                history['CE']['val']['loss'].append(ce_eval_loss)
                history['CE']['val']['step'].append(step)
                history['model']['val']['loss'].append(model_eval_loss)
                history['model']['val']['step'].append(step)

                dir_path = os.path.join(*dump_model_name.split('/')[:-1])
                visualization_callback(history, name=os.path.join(dir_path, 'graph.pdf'))

                print(f'epoch {epoch}/{max_epochs}, epoch_step {i}/{len(train_dataloader)}, global_step {step}/{max_steps}')
                print(f'ce train loss: {ce_ewm_loss:4.4f}, model train loss: {model_ewm_loss:4.4f}, kl loss: {kl_ewm_loss:4.4f}')
                print(f'ce val loss: {ce_eval_loss:4.4f}, model val loss: {model_eval_loss:4.4f}')
                print()

    cleanup()


def set_up_exp(
        data_path: str,
        transformer_model_name: str = 'sberbank-ai/ruT5-base',
        tokenizer_model_name: str = 'sberbank-ai/ruT5-base',
        loss_model_name: str = 'SkolkovoInstitute/russian_toxicity_classifier',
        adapter_input_dim: int = 32128,
        adapter_output_dim: int = 768,
        pretrain_adapter_path: Optional[str] = None,
        adapter_lr: float = 3e-5,
        model_lr: float = 3e-5,
        batch_size: int = 8,
        max_epochs: int = 1000,
        max_steps: int = 10000,
        dump_model_name: str = 'tmp/base',
        report_step: int = 500,
        period_of_dump: int = 2500,
        ce_w: float = 1.0,
        model_w: float = 0.0,
        train_model: bool = True,
        train_adapter: bool = False,
        seed: int = 666,
        test_size: Optional[float] = 0.1,
        device: str = '4',
        loss_mode: str = 'sum',
        num_accum_steps: int = 1,
        model_period_train: Optional[int] = None,
        adapter_period_train: Optional[int] = None,
):
    sns.set(palette='Set2', font_scale=1.3)

    os.environ['CUDA_VISIBLE_DEVICES'] = device
    set_seed(seed)

    # Готовим модели
    print('Start loading models...')
    detox_model = T5ForConditionalGeneration.from_pretrained(transformer_model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    adapter = LinearAdapter(adapter_input_dim, adapter_output_dim).cuda()
    if pretrain_adapter_path is not None:
        adapter.load_state_dict(torch.load(pretrain_adapter_path))
    loss_model, loss_tokenizer = load_model(loss_model_name, use_cuda=True)

    turn_off_grad(loss_model)
    turn_off_grad(adapter)
    print('All models are loading.\n')

    # Готовим данные
    print('Start preparing data...')
    df = read_dataset(data_path)
    neutral_toxicity = classify_texts(
        model=loss_model,
        tokenizer=loss_tokenizer,
        texts=df['toxic_comment'].tolist(),
        batch_size=64,
        verbose=False
    ).detach().cpu().numpy().tolist()

    if test_size is not None:
        toxic_train, toxic_val, neutral_train, neutral_val, neu_tox_train, neu_tox_val = train_test_split(
            df['toxic_comment'].tolist(), df['neutral_comment'].tolist(), neutral_toxicity,
            test_size=test_size, random_state=seed
        )
    else:
        toxic_train, neutral_train, neu_tox_train = (
            df['toxic_comment'].tolist(), df['neutral_comment'].tolist(), neutral_toxicity
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = PairsDataset(tokenizer(toxic_train), tokenizer(neutral_train), neu_tox_train)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=data_collator,
    )

    if test_size is not None:
        val_dataset = PairsDataset(tokenizer(toxic_val), tokenizer(neutral_val), neu_tox_val)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, drop_last=False, shuffle=True, collate_fn=data_collator,
        )
    else:
        val_dataloader = None

    print('All data are ready.\n')

    print('Start training...')
    train_loop(
        detox_model, train_dataloader, val_dataloader, loss_model=loss_model, adapter_model=adapter,
        max_epochs=max_epochs,
        max_steps=max_steps,
        adapter_lr=adapter_lr,
        model_lr=model_lr,
        cleanup_step=100,
        report_step=report_step,
        window=100,
        period_of_dump=period_of_dump,
        dump_model_name=dump_model_name,
        ce_w=ce_w,
        model_w=model_w,
        train_model=train_model,
        train_adapter=train_adapter,
        loss_mode=loss_mode,
        num_accum_steps=num_accum_steps,
        model_period_train=model_period_train,
        adapter_period_train=adapter_period_train,
    )
