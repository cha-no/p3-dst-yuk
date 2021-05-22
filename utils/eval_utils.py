"""utils of evaluation

"""

from typing import List, Dict

class DSTEvaluator:
    """DST Evaluator
    """
    def __init__(self : object, slot_meta : List[str]) -> None:
        """Initialize DST Evaluator

        Args:
            self (object)
            slot_meta (List[str])
        """
        self.slot_meta = slot_meta
        self.init()

    def init(self : object) -> None:
        """initialize score

        Args:
            self (object)
        """
        self.joint_goal_hit = 0
        self.all_hit = 0
        self.slot_turn_acc = 0
        self.slot_F1_pred = 0
        self.slot_F1_count = 0

    def update(self : object, gold : List[str], pred : List[str]) -> None:
        """Update Scores

        Args:
            self (object)
            gold (List[str]): slot's ground truth
            pred (List[str]): slot's prediction
        """
        self.all_hit += 1
        if set(pred) == set(gold):
            self.joint_goal_hit += 1

        temp_acc = compute_acc(gold, pred, self.slot_meta)
        self.slot_turn_acc += temp_acc

        temp_f1, _, _, count = compute_prf(gold, pred)
        self.slot_F1_pred += temp_f1
        self.slot_F1_count += count

    def compute(self : object) -> Dict[str, float]:
        """Compute Joint goal accuracy, turn accuracy, slot f1 score

        Args:
            self (object)

        Returns:
            Dict[str, float]: Joint goal accuracy, turn accuracy, slot f1 score
        """
        turn_acc_score = self.slot_turn_acc / self.all_hit
        slot_F1_score = self.slot_F1_pred / self.slot_F1_count
        joint_goal_accuracy = self.joint_goal_hit / self.all_hit
        eval_result = {
            "joint_goal_accuracy": joint_goal_accuracy,
            "turn_slot_accuracy": turn_acc_score,
            "turn_slot_f1": slot_F1_score,
        }
        return eval_result


def compute_acc(gold : List[str], pred : List[str], slot_meta : List[str]) -> float:
    """Compute turn accuracy score

    Args:
        gold (List[str]): slot's ground truth
        pred (List[str]): slot's predictions
        slot_meta (List[str])

    Returns:
        float: turn accuracy score
    """
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_meta)
    ACC = len(slot_meta) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold : List[str], pred : List[str]):
    """Compute slot f1 score

    Args:
        gold (List[str]): slot's ground truth
        pred (List[str]): slot's predictions

    Returns:
        [type]: slot f1 score
    """
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = (
            2 * precision * recall / float(precision + recall)
            if (precision + recall) != 0
            else 0
        )
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


class AverageMeter(object):
    """compute average scores

    Args:
        object
    """
    def __init__(self : object) -> None:
        """Initialize AverageMeter

        Args:
            self (object)
        """
        self.reset()

    def reset(self : object) -> None:
        """Reset scores

        Args:
            self (object)
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self : object, val : float, n : int = 1) -> None:
        """Update Average scores

        Args:
            self (object)
            val (float): update scores
            n (int, optional): number of batches. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
