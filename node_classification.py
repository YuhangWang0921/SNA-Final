from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def node_classification_eva(node_reps, node_labels):
    """
    Trains logistic regression models for nodes based on their representations and evaluates them.

    Args:
    node_reps (dict): A dictionary where keys are hyperparameters and values are node representations.
    node_labels (list or array): The true labels for the nodes.

    Returns:
    dict: A dictionary with keys as hyperparameters and values being a dictionary of Micro F1 and Macro F1 scores.
    """
    performance_scores = {}

    for hp, representations in node_reps.items():
        # Split the dataset into training and testing sets
        print(f"Hyperparameter: {hp}, Representations length: {len(representations)}, Labels length: {len(node_labels)}")
        # make sure length is equal
        if len(representations) != len(node_labels):
            raise ValueError(f"Length mismatch for hyperparameter {hp}: {len(representations)} representations vs {len(node_labels)} labels")
        X_train, X_test, y_train, y_test = train_test_split(representations, node_labels, test_size=0.3, random_state=42)

        # Initialize the Logistic Regression model
        model = LogisticRegression(solver='liblinear', multi_class='ovr')

        # Train the model
        model.fit(X_train, y_train)

        # Predict labels for the test set
        predictions = model.predict(X_test)

        # Calculate F1 scores
        micro_f1 = f1_score(y_test, predictions, average='micro')
        macro_f1 = f1_score(y_test, predictions, average='macro')

        # Store the performance scores
        performance_scores[hp] = {'Micro F1': micro_f1, 'Macro F1': macro_f1}
    print(f"Performance as follows:{performance_scores}")
    # calculate the average of Micro F1  Macro F1 
    total_micro_f1 = sum(score['Micro F1'] for score in performance_scores.values())
    total_macro_f1 = sum(score['Macro F1'] for score in performance_scores.values())

    average_micro_f1 = total_micro_f1 / len(performance_scores)
    average_macro_f1 = total_macro_f1 / len(performance_scores)

    print(f"Average Micro F1: {average_micro_f1}")
    print(f"Average Macro F1: {average_macro_f1}")
    return performance_scores,average_micro_f1,average_macro_f1

# Example usage
# BC_embeddings = load_embeddings(dataset)
# BC_embeddings = dict(BC_embeddings)
# node_labels = [...]  # Your node labels
# scores = node_classification(BC_embeddings, node_labels)

