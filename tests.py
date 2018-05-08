from confusion_matrix import ConfusionMatrix
from sklearn.metrics import classification_report
from weighting import voting
import numpy as np
from project_main import load_preds
from project_main import store_preds

def test_confusion_matrix():
    """Test ConfusionMatrix class"""

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

    cm = ConfusionMatrix(yactual, ypred, "Sample_classifier")

    number_cm = cm.get_number_cm()
    normalized_cm = cm.get_normalized_cm()
    metrics = {"TP": cm.get_true_pos(), "TN": cm.get_true_neg(), "FP": cm.get_false_pos(), "FN": cm.get_false_neg()}

    print("Confusion matrix:")
    print(number_cm)

    print("Normalized confusion matrix:")
    print(normalized_cm)

    print("Precision per label is:", cm.get_precision())
    print("Metrics:", metrics)

    print("MCC is", cm.get_mcc())

    return cm.get_precision()

def test_weighting():
    """Test weighting.py"""

    # Create confusion matrices for random classifiers
    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred1 = np.random.randint(3, size=12)

    cm1 = ConfusionMatrix(yactual, ypred1, "cls_1")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred2 = np.random.randint(3, size=12)

    cm2 = ConfusionMatrix(yactual, ypred2, "cls_2")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred3 = np.random.randint(3, size=12)

    cm3 = ConfusionMatrix(yactual, ypred3, "cls_3")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred4 = np.random.randint(3, size=12)

    cm4 = ConfusionMatrix(yactual, ypred4, "cls_4")

    weight_pairs = [[cm1, ypred1], [cm2, ypred2], [cm3, ypred3], [cm4, ypred4]]

    # Check that CEN score is being calculated
    #print(cm1.get_CEN_score(), cm2.get_CEN_score(), cm3.get_CEN_score(), cm4.get_CEN_score())

    # Get final votes based on pairs
    votes_p = voting(weight_pairs, "Precision")
    votes_CEN_p = voting(weight_pairs, "CEN_Precision")
    votes_CEN = voting(weight_pairs, "CEN")
    votes_eq = voting(weight_pairs, "Equal_Vote")

    # Check metrics
    print(classification_report(yactual, votes_p))
    print(classification_report(yactual, votes_CEN_p))
    print(classification_report(yactual, votes_CEN))
    print(classification_report(yactual, votes_eq))

    # Create final confusion matrices depending on votes
    p_cm = ConfusionMatrix(yactual, votes_p, "Precision_Voting")
    p_CEN_cm = ConfusionMatrix(yactual, votes_CEN_p, "CEN_Precision_Voting")
    CEN_cm = ConfusionMatrix(yactual, votes_CEN, "CEN_Voting")
    eq_cm = ConfusionMatrix(yactual, votes_eq, "Equal_Voting")

    # Store confusion matrices
    p_cm.store_cm()
    p_CEN_cm.store_cm()
    CEN_cm.store_cm()
    eq_cm.store_cm()

    #print(votes)


def test_storing_loading():
    """Test store_preds and load_preds"""

    # Create confusion matrices for random classifiers
    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred1 = np.random.randint(3, size=12)

    cm1 = ConfusionMatrix(yactual, ypred1, "cls_1")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred2 = np.random.randint(3, size=12)

    cm2 = ConfusionMatrix(yactual, ypred2, "cls_2")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred3 = np.random.randint(3, size=12)

    cm3 = ConfusionMatrix(yactual, ypred3, "cls_3")

    yactual = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    ypred4 = np.random.randint(3, size=12)

    cm4 = ConfusionMatrix(yactual, ypred4, "cls_4")

    preds = [ypred1, ypred2, ypred3, ypred4]

    print("Preds before saving", preds)

    store_preds(preds, yactual, 1)
    new_preds, actual = load_preds(1)

    print("Preds after saving", new_preds, "Actual after saving", actual)

def test_run_multiple_voting():
    """Run tests on multiple weighting systems given stored predictions of the classifiers in project_main.py"""

    # Load predictions from all classifiers and actual labels for test_set_1
    preds1, actual1 = load_preds(1)

    # Load predictions from all classifiers and actual labels for test_set_2
    preds2, actual2 = load_preds(2)

    # Create confusion matrices for each classifier
    p_cm = ConfusionMatrix(actual1, preds1[0], "Proximity")
    v_cm = ConfusionMatrix(actual1, preds1[1], "Voting")
    # b_cm = ConfusionMatrix(actual1, preds1[2], "Bayes")
    r_cm = ConfusionMatrix(actual1, preds1[2], "LSTM")

    confusionMatrices = [p_cm, v_cm, r_cm]
    # confusionMatices = [p_cm, v_cm, b_cm, r_cm]

    # Save individual confusion matrices to files
    for cm in confusionMatrices:
        cm.store_cm()

    print("Individual confusion matrices created and stored!")

    # Weight second set of results, using confusion matrices from first set
    weightingInput = [
        [confusionMatrices[0], preds2[0]],
        [confusionMatrices[1], preds2[1]],
        # [confusionMatrices[2] ,b.batchTest(test_set2)],
        [confusionMatrices[3], preds2[2]]
    ]

    # Get the weighted voting results
    votes_p = voting(weightingInput, "Precision")
    votes_CEN_p = voting(weightingInput, "CEN_Precision")
    votes_CEN = voting(weightingInput, "CEN")
    votes_eq = voting(weightingInput, "Equal_Vote")

    # Check metrics
    print(classification_report(actual2, votes_p))
    print(classification_report(actual2, votes_CEN_p))
    print(classification_report(actual2, votes_CEN))
    print(classification_report(actual2, votes_eq))

    # Create final confusion matrices depending on votes
    p_cm = ConfusionMatrix(actual2, votes_p, "Precision")
    p_CEN_cm = ConfusionMatrix(actual2, votes_CEN_p, "CEN_Precision")
    CEN_cm = ConfusionMatrix(actual2, votes_CEN, "CEN")
    eq_cm = ConfusionMatrix(actual2, votes_eq, "Equal")

    # Store confusion matrices

    p_cm.store_cm()
    p_CEN_cm.store_cm()
    CEN_cm.store_cm()
    eq_cm.store_cm()

def main():

    #test_confusion_matrix()
    #test_weighting()
    #test_storing_loading()
    test_run_multiple_voting()

if __name__ == "__main__":
    main()

