import numpy as np
import pandas as pd
from pprint import pprint


# calculate entropy
def calculate_entropy(target_col):
    # Store the target value and number in elements and counts.
    elements, counts = np.unique(target_col, return_counts=True)

    # Calculate entropy
    total = np.sum(counts)
    entropy = -np.sum([(counts[i] / total) * np.log2(counts[i] / total) for i in range(len(elements))])

    return entropy


# Calculate information gain
def information_gain(dataset, split_attribute_name, target_name):
    # Calculate root entropy
    root_entropy = calculate_entropy(dataset[target_name])

    elements, counts = np.unique(dataset[split_attribute_name], return_counts=True)

    # Calculate the sum of the weighted child entropy.
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) *
                               calculate_entropy(
                                   dataset.where(dataset[split_attribute_name] == elements[i]).dropna()[target_name])
                               for i in range(len(elements))])

    info_gain = root_entropy - weighted_entropy

    return info_gain


# Making decision tree
def decision_tree(dataset, original_dataset, features, target_attribute_name, parent_node_class=None):
    # Determine when to stop splitting

    # 1. If the target value has a single value, return the target value
    if len(np.unique(dataset[target_attribute_name])) <= 1:
        return np.unique(dataset[target_attribute_name])[0]

    # 2. When there are no more records, Returns the target value with the maximum value from the original dataset
    elif len(dataset) == 0:
        return np.unique(original_dataset[target_attribute_name])[
            np.argmax(np.unique(original_dataset[target_attribute_name], return_counts=True)[1])]

    # 3. When there is no more attribute to compare, Returns the target value of the parent node
    elif len(features) == 0:
        return parent_node_class

    # Grow the tree

    else:
        # Define the target value of the parent node(Nothing, Respond)
        parent_node_class = np.unique(dataset[target_attribute_name])[
            np.argmax(np.unique(dataset[target_attribute_name], return_counts=True)[1])]

        # Select attribute to split
        # Obtain information gain for each attribute and select the attribute with the maximum information gain
        item_values = [information_gain(dataset, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Generate tree structure
        tree = {best_feature: {}}

        # Exclude attributes with maximum information gain for selecting the next attribute to split
        features = [i for i in features if i != best_feature]

        # Grow branch
        for value in np.unique(dataset[best_feature]):
            # Split records of the same values.
            # Remove rows and columns with missing values
            sub_data = dataset.where(dataset[best_feature] == value).dropna()

            # Recursive partitioning
            subtree = decision_tree(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree


# Read dataset
dataset = pd.read_excel('PHW4_1_dataset.xlsx', sheet_name='dataset')

tree = decision_tree(dataset, dataset, ['District', 'House Type', 'Income', 'Previous Customer'], "Outcome")

pprint(tree)
