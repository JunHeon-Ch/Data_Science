import numpy as np
from pprint import pprint

# dataset
data = pd.DataFrame({"District": ["Suburban", "Suburban", "Suburban", "Suburban", "Suburban", "Rural", "Rural", "Rural",
                                  "Rural", "Urban", "Urban", "Urban", "Urban", "Urban"],
                     "House Type": ["Detached", "Detached", "Semi-detached", "Semi-detached", "Semi-detached",
                                    "Semi-detached", "Semi-detached", "Semi-detached", "Semi-detached", "Detached",
                                    "Detached", "Detached", "Detached", "Detached"],
                     "Income": ["High", "High", "High", "High", "High", "Low", "Low", "Low", "Low", "Low", "Low", "Low",
                                "Low", "Low"],
                     "Previous Customer": ["No", "No", "Yes", "Yes", "Yes", "No", "No", "No", "No", "Yes", "Yes", "No",
                                           "No", "No"],
                     "Outcome": ["Nothing", "Nothing", "Respond", "Respond", "Respond", "Respond", "Respond", "Respond",
                                 "Respond", "Nothing", "Nothing", "Respond", "Respond", "Respond"]},
                    columns=["District", "House Type", "Income", "Previous Customer", "Outcome"])

print(data)

# descriptive features
features = data[["District", "House Type", "Income", "Previous Customer"]]

# target feature
target = data["Outcome"]


# calculate entropy
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum(
        [(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# calculate infogain
def InfoGain(data, split_attribute_name, target_name):
    # calculate entropy
    total_entropy = entropy(data[target_name])
    print('Entropy(D) = ', round(total_entropy, 5))

    # calculate weight entropy
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) *
                               entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
                               for i in range(len(vals))])
    print('H(', split_attribute_name, ') = ', round(Weighted_Entropy, 5))

    # calculate infogain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


print('InfoGain( District ) = ', round(InfoGain(data, "District", "Outcome"), 5))
print()
print('InfoGain( House Type ) = ', round(InfoGain(data, "House Type", "Outcome"), 5))
print()
print('InfoGain( Income ) = ', round(InfoGain(data, "Income", "Outcome"), 5))
print()
print('InfoGain( Previous Customer ) = ', round(InfoGain(data, "Previous Customer", "Outcome"), 5))
print()


# node split decision
def Split(data, originaldata, features, target_attribute_name, parent_node_class=None):
    # If the target property has a single value: return the destination property
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # When there is no data: Returns the target property with the maximum value from the source data
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name]) \
            [np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # When no technical properties exist: Return target properties for parent node
    elif len(features) == 0:
        return parent_node_class

    # tree growth
    else:
        # Define target properties for the parent node
        parent_node_class = np.unique(data[target_attribute_name]) \
            [np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # Select properties to split data
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # tree structure generation
        tree = {best_feature: {}}

        # Exclude technical attributes that show maximum information
        features = [i for i in features if i != best_feature]

        # branch growth
        for value in np.unique(data[best_feature]):
            # Split data. dropna(): rows with missing values, remove columns
            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = Split(sub_data, data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return (tree)


tree = Split(data, data, ["District", "House Type", "Income", "Previous Customer"], "Outcome")
print()
pprint(tree)
