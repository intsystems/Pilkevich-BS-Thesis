import os
import argparse
import pandas as pd

from typing import Optional
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from nltk.translate.chrf_score import corpus_chrf
from tqdm.auto import tqdm

from ..utils import load_model
from .metrics import evaluate_style_transfer


def paraphrase(text, model, tokenizer, n=None, max_length='auto', temperature=0.0, beams=3):
    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors='pt', padding=True)['input_ids'].to(model.device)
    if max_length == 'auto':
        max_length = int(inputs.shape[1] * 1.2) + 10
    result = model.generate(
        inputs,
        num_return_sequences=n or 1,
        do_sample=False,
        temperature=temperature,
        repetition_penalty=3.0,
        max_length=max_length,
        bad_words_ids=[[2]],  # unk
        num_beams=beams,
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]
    if not n and isinstance(text, str):
        return texts[0]
    return texts


def inference(
        data_path: str,
        model_name: str,
        tokenizer_model_name: str,
        result_path: str,
        device: str = '0',
        batch_size: int = 64
):
    os.environ['CUDA_VISIBLE_DEVICES'] = device

    data = pd.read_csv(data_path, sep='\t')
    toxic = data['toxic_comment'].tolist()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()

    para_results = []
    # problematic_batch = []  # if something goes wrong you can track such bathces

    for i in tqdm(range(0, len(toxic), batch_size)):
        batch = [sentence for sentence in toxic[i:i + batch_size]]
        try:
            para_results.extend(paraphrase(batch, model, tokenizer, temperature=0.0))
        except Exception as e:
            print(i)
            para_results.append(toxic[i:i + batch_size])

    with open(result_path, 'w') as file:
        file.writelines([sentence + '\n' for sentence in para_results])


def evaluate_and_dump(
        gold_label_path: str,
        predicts_path: str,
        name: str,
        output_path: Optional[str] = None,
        batch_size: int = 16,
        device: Optional[str] = None,
        target_metric='accuracy',
):
    use_cuda = True if device is not None else False
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    print('Load data...')
    df = pd.read_csv(gold_label_path, sep='\t').fillna('')
    toxic_inputs = df['toxic_comment'].tolist()

    with open(predicts_path, 'r') as f:
        rewritten = f.readlines()
        rewritten = [sentence.strip() for sentence in rewritten]

    print('Load models...')
    style_model, style_tokenizer = load_model('SkolkovoInstitute/russian_toxicity_classifier', use_cuda=use_cuda)
    meaning_model, meaning_tokenizer = load_model('cointegrated/LaBSE-en-ru', use_cuda=use_cuda, model_class=AutoModel)
    fluency_model, fluency_tolenizer = load_model('SkolkovoInstitute/rubert-base-corruption-detector', use_cuda=use_cuda)

    results = evaluate_style_transfer(
        original_texts=toxic_inputs,
        rewritten_texts=rewritten,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=fluency_model,
        cola_tokenizer=fluency_tolenizer,
        style_target_label=0,
        batch_size=batch_size,
        aggregate=True
    )

    neutral_references = []
    for index, row in df.iterrows():
        neutral_references.append([row['neutral_comment1'], row['neutral_comment2'], row['neutral_comment3']])

    results['chrf'] = corpus_chrf(neutral_references, rewritten)
    results[f'{target_metric}_chrf'] = results[target_metric] * results['chrf']

    if output_path is not None:
        if not os.path.exists(output_path):
            with open(output_path, 'w') as f:
                f.writelines(f'| Model | ACC | SIM | FL | J | ChrF1 | {target_metric}*ChrF1 |\n')
                f.writelines('| ----- | --- | --- | -- | - | ----- | --------- |\n')

        with open(output_path, 'a') as res_file:
            res_file.writelines(
                f"{name}|{results['accuracy']:.4f}|{results['similarity']:.4f}|{results['fluency']:.4f}|"
                f"{results['joint']:.4f}|{results['chrf']:.4f}|{results[f'{target_metric}_chrf']:.4f}|\n"
            )

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputs", help="path to test sentences", required=True)
    parser.add_argument('-p', "--preds", help="path to predictions of a model", required=True)
    parser.add_argument('-r', "--output_path", help="path to result .md file", default=None, type=Optional[str])
    parser.add_argument('-n', "--name", help="model name", default='test', type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default=None, type=Optional[str])

    args = parser.parse_args()

    evaluate_and_dump(
        gold_label_path=args.inputs,
        predicts_path=args.preds,
        name=args.name,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device=args.device,
    )
