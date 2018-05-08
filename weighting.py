def voting(weighting_input, weighting_opt):
    """Creates label:weight dictionary, voting_list and CEN dictionary for each classifier
    and calls voting methods according to selected option"""

    voting_list = []
    weighted_votes = []
    CEN_scores = {}
    num_results = len(weighting_input[0][1]) #number of tweets
    num_cls = 0

    # For every pair of (confusion_matrix, predictions)
    for pair in weighting_input:

        conf_matrix = pair[0]

        # Make the CEN scores as {cls1:score1, cls2:score2,...}
        CEN_scores[num_cls] = conf_matrix.get_CEN_score()

        # Create the weight dictionary for a given model in the format {label:weight}
        weight_dict = dict()
        label = 0

        if weighting_opt == "Precision":
            for weight in conf_matrix.get_precision():  # precision = [0.7, 0.2, 0.1]
                weight_dict[label] = weight
                label += 1

        elif weighting_opt == "CEN_Precision":
            for weight in conf_matrix.get_precision():  # precision = [0.7, 0.2, 0.1]
                weight_dict[label] = weight * (1 - CEN_scores[num_cls])
                label += 1

        elif weighting_opt == "CEN":
            weight_dict[0] = (1 - CEN_scores[num_cls])
            weight_dict[1] = (1 - CEN_scores[num_cls])
            weight_dict[2] = (1 - CEN_scores[num_cls])

        elif weighting_opt == "Equal_Vote":
            weight_dict[0] = 1
            weight_dict[1] = 1
            weight_dict[2] = 1

        #print("Classifier", pair[0].get_name(), ":")
        #print("Weight dict:", weight_dict)

        preds = pair[1]

        # Get list of votes transformed to weights [{0:0.7, 1:0, 2:0}, {0:0, 1:0.6, 2:0}, {0:0, 1:0, 2:0.4}, ...]
        votes = cls_vote(weight_dict, preds)

        #print("List of votes:", votes)

        # Append to list of weighted votes
        weighted_votes.append(votes)

        # Create voting_list in the format [[weighted_votes_cls1], [weighted_votes_cls2], [weighted_votes_cls3], ...]
        voting_list.append(weighted_votes)

        num_cls += 1

    return final_vote(num_results, voting_list)

def cls_vote(weights, preds): # Single classifier vote
    """Takes a dictionary of voting weights and the prediction list, for one classifier
        Returns a list of votes in terms of the weights of the dictionary

        A vote has the format {0:weight0, 1:weight1, 2:weight2}, where:
            - weightx = precision/CEN/weight score if the cls returned x
            - 0 otherwise
        Ex. Vote for neutral tweet = {0: 0.73, 1:0, 2:0}
    """
    votes = []

    # For every prediction
    for pred in preds:
        vote = {}

        # For all the possible labels
        for label in range(3):

            # Assign the value of the precision score to the prediction, else 0
            if label == pred:
                vote[label] = weights[pred]
            else:
                vote[label] = 0
        votes.append(vote)

    return votes


def final_vote(n_results, v_list):
    """Returns list of final votes as integers"""

    final_votes = []
    cls_num = 0

    f_votes_list = []

    # Get the final vote for every tweet in the list
    for i in range(n_results):

        # Contains aggregate votes for all the classifiers, f_vote = {0:1.4, 1:0.1, 2:0.6}
        f_vote = {}

        # For every list of weighted votes
        for cls_votes in v_list[cls_num]:

            # Get the vote dict for tweet i
            vote = cls_votes[i]

            for label in vote.keys():
                if label in f_vote.keys():
                    f_vote[label] += vote[label]
                else:
                    f_vote[label] = vote[label]


        f_votes_list.append(f_vote)
        final_vote = 0
        score = 0

        #print(f_vote)
        #print(type(f_vote[0]))

        # Get the label that gets the highest aggregate precision score
        for l in f_vote.keys():
            if f_vote[l] > score:
                score = f_vote[l]
                final_vote = l

        final_votes.append(final_vote)

    #print("Sample votes are:")
    #print(f_votes_list[0])
    #print(f_votes_list[1])
    #print(final_votes)
    return final_votes
