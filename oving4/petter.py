# Importing the necessary libraries.
import pandas as pd
import numpy as np
import random as r
from graphviz import Digraph


# Entropy of boolen value.
def B(x):
    # Avoid log2(0)
    if (x == 0) or (x == 1):
        return 0
    return -(x*np.log2(x) + (1 - x)*np.log2(1 - x))


# Find out where to split a continuous value.
def find_threshold(df, A):
    """
    Only split on unique states to avoid unnecessary function calls.
    I found this method to be more efficient than sorting the values and
    iteration through every value, while yielding the same results.
    """
    states = df[A].unique()
    gains = []
    for i in states:
        # Avoid changing values in the original dataframe by creating a copy.
        temp = df.copy()
        # Using <= to split yields the same results as using the splitpoint (v1+v2)/2.
        temp[A] = temp[A].apply(lambda x: 0 if x <= i else 1)
        gains.append(gain(temp, A))
    return states[np.argmax(gains)]


# Calculating the gain from a specific attribute.
def gain(df, A):
    # Assume that the possible states of the attribute we are predicting are 0 and 1.
    p = len(df[df[check] == 1])
    n = len(df[df[check] == 0])
    remainder = 0
    temp = df.copy()
    """
    Will count more that 5 unique states as a non-categorical variable.
    If there were attributes with more than 5 states of a non-int/float type
    we could take care of this as also using type=int or float as a condition,
    but this is not an issue in this data set.
    """
    if len(df[A].unique()) > 5:
        states = [0, 1]
        split = find_threshold(temp, A)
        temp[A] = temp[A].apply(lambda x: 0 if x <= split else 1)
    else:
        states = df[A].unique()
    for i in states:
        # Ignore missing values.
        if str(i) == "nan":
            continue
        pk = len(temp[(temp[check] == 1) & (temp[A] == i)])
        nk = len(temp[(temp[check] == 0) & (temp[A] == i)])
        remainder += (pk + nk)/(p + n) * B(pk/(pk + nk))
    return B(p/(n+p)) - remainder



# Define node to make the calculations easier and the visualization go smoothly.
class Node:
    def __init__(self, a, p = 0, s = None):
        self.attribute = a
        self.children = {}
        # For cases where the test data has a state not previously seen or nan.
        self.plurality = p
        # Get split point when the variable is continuous.
        self.splitpoint = s
        

# Function to get all possible states from initial data set, so that no states
# get lost as the data gets reduced in the recursive process.
def get_states(data, attributes):
    d = {}
    for i in attributes:
        d[i] = list(data[i].unique())
    return d


# Build the decision tree as described in the pseudocode in the book.
# Also with adjustments to handle non-categorical variables.
def build_tree(data, attributes, states, parent_data = None):
    if data.empty:
        if parent_data[check].value_counts()[0] < parent_data[check].value_counts()[1]:
            return Node(positive)
        elif parent_data[check].value_counts()[0] > parent_data[check].value_counts()[1]:
            return Node(negative)
        else:
            rand = round(r.random())
            if rand == 0:
                return Node(positive)
            else:
                return Node(negative)
    # Check if all datapoints have the same classification.    
    elif len(list(data[check].unique())) == 1:
        if data.iloc[0,0] == 0:
            return Node(negative)
        else:
            return Node(positive)
    # Check if all attributes have been used.    
    elif len(attributes) == 0:
        if data[check].value_counts()[0] < data[check].value_counts()[1]:
            return Node(positive)
        else:
            return Node(negative)
        
    else:
        A = np.argmax([gain(data, a) for a in attributes])
        att = attributes[A]
        # Making a separate treatment for non-categorical variables.
        if len(states[att]) > 5:
            split = find_threshold(data, att)
            root = Node(att, data.Survived.mode()[0], split)
            # Separate data into two new nodes based on the split.
            # 'nan' values will be left out.
            exs = data[data[att] <= split]
            exs2 = data[data[att] > split]
            reduced_attributes = attributes[:]
            reduced_attributes.pop(A)
            subtree = build_tree(exs, reduced_attributes, states, data)
            root.children["Under "+str(split)] = subtree
            subtree2 = build_tree(exs2, reduced_attributes, states, data)
            root.children["Over "+str(split)] = subtree2
        # Otherwise follow pseudocode.
        else: 
            root = Node(att, data.Survived.mode()[0])
            for state in states[att]:
                exs = data[data[att] == state]
                reduced_attributes = attributes[:]
                reduced_attributes.pop(A)
                subtree = build_tree(exs, reduced_attributes, states, data)
                root.children[state] = subtree
        return root
    

# Use graphviz to visualize the tree from the root node.
def draw_graph(root):
    dot = Digraph(comment = 'My tree')
    queue = [(root, "")]
    dot.node(root.attribute, root.attribute)
    while len(queue) > 0:
        n, tag = queue.pop(0)
        for key, value in n.children.items():
            queue.append((value, tag+str(key)))
            dot.node(tag+str(key)+str(value.attribute), value.attribute)
            dot.edge(tag+n.attribute, tag+str(key)+str(value.attribute),
                     label=str(key))
        
    dot.view()
    

# Use tree to predict result for test data.
def predict(datapoint, root, states):
    if root.attribute == positive:
        return 1
    elif root.attribute == negative:
        return 0
    else:
        # For states not before seen or nan.
        if datapoint[root.attribute] not in states[root.attribute]:
            return root.plurality
        # Different treatment when predicting based on non-categorical variables.
        elif len(states[root.attribute]) > 5:
            s = root.splitpoint
            if datapoint[root.attribute] > s:
                return predict(datapoint, root.children["Over "+str(s)], states)
            else:
                return predict(datapoint, root.children["Under "+str(s)], states)
        else:
            return predict(datapoint, root.children[datapoint[root.attribute]], states)
    
def predict_all(data, root, states):
    points = data.shape[0]
    results = np.zeros(points)
    for i in range(points):
        results[i] = predict(data.iloc[i, :], root, states)
    return results
    

# Determine the data and state we are going to work with.
df = pd.read_csv('./titanic/train.csv')
test_data = pd.read_csv('./titanic/test.csv')
check = 'Survived'    # This will be the attribute we want to predict.
positive = 'Survived' # The positive outcome of the predicted value.
negative = 'Died'     # The negative outcome of the predicted value.

# These will be the attributes we work with in task 1a)
atts_a = ['Sex', 'Pclass', 'Embarked']
states_a = get_states(df, atts_a)


# These will be the attributes we work with in task 1b)
atts_b = ['Age', 'Sex', 'Pclass', 'Parch', 'SibSp', 'Fare']
states_b = get_states(df, atts_b)


# Perform the tasks from the assignment description.
def task(name):
    if name == 'a':
        ro = build_tree(df, atts_a, states_a)
        re = predict_all(test_data, ro, states_a)
    elif name == 'b':
        ro = build_tree(df, atts_b, states_b)
        re = predict_all(test_data, ro, states_b)
    else:
        return print("No such task.")
    draw_graph(ro)
    z = 0
    y = 0
    for i in range(test_data.shape[0]):
        z += 1
        if re[i] == test_data.iloc[i,0]:
            y += 1
        
    print('Accuracy of predictions in task', name+':',y/z*100,'%')


if __name__ == '__main__':
    # task('a')
    task('b')