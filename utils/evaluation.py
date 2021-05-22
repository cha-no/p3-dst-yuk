"""evaluation slot value

"""

import json
import argparse

from typing import Dict, List

from .eval_utils import DSTEvaluator


SLOT_META_PATH = 'data/train_dataset/slot_meta.json'


def _evaluation(preds : Dict[str, List[str]], labels : Dict[str, List[str]], slot_meta : List[str]) -> Dict[str, float]:
    """Evaluation Joint Joint goal accuracy, turn accuracy, slot f1 score

    Args:
        preds (Dict[str, List[str]]): slot's predictions
        labels (Dict[str, List[str]]): slot's ground truth
        slot_meta (List[str])

    Raises:
        Exception: key is not in the predictions

    Returns:
        Dict[str, float]: Joint goal accuracy, turn accuracy, slot f1 score
    """
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels)

    for k, l in labels.items():
        p = preds.get(k)
        if p is None:
            raise Exception(f"{k} is not in the predictions!")
        evaluator.update(l, p)

    result = evaluator.compute()
    print(result)
    return result


def evaluation(gt_path : str, pred_path : str) -> Dict[str, float]:
    """Evaluation Joint Joint goal accuracy, turn accuracy, slot f1 score

    Args:
        gt_path (str): slot's ground truth path
        pred_path (str): slot's predictions path

    Returns:
        Dict[str, float]: Joint goal accuracy, turn accuracy, slot f1 score
    """
    slot_meta = json.load(open(SLOT_META_PATH))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_result = _evaluation(preds, gts, slot_meta)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path)
