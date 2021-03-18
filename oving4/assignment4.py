import pandas as pd 
import numpy as np 
import random as r
from graphviz import Digraph
import operator

def b(q):
    """
    function to calculate entropy of boolean values
    params: 
            q = probability
    """
    if (q==0 or q==1):
        return 0
    else:
        return -(q*np.log2(q)+(1-q)*np.log2(1-q))

def gain(df, a):
    """
    function to calculate the gain for a given atribute
    params: 
            df = the dataframe you want to calculate the gain from
            a = the attribute for which you want to calculate the gain 
    """
    p = len(df[df["Survived"] == 1])
    n = len(df[df["Survived"] == 0])
    remainder = 0
    temp = df.copy()
    total = len(df)
    #Test to chack if the attribute is continous
    # If the attribute is continous, you have to calculate the best split
    # and get the ramiander given this split
    if len(df[a].unique()) > 5:
        attributes = [0, 1]
        split1 = split(temp, a)
        temp[a] = temp[a].apply(lambda x: 0 if x <= split1 else 1)
        #If some fields in the attributes coloumn does not have data
        for i in attributes:
            if str(i) == "nan":
                continue
        positive = len(temp[(temp["Survived"] == 1) & (temp[a] == i)])
        negative = len(temp[(temp["Survived"] == 0) & (temp[a] == i)])
        current_tot = b(positive/(positive+negative))
        current = (positive+negative)/total * current_tot
        remainder += current
    #If the attributes is not continous
    #Calculate the ramainder
    else:
        attributes = df[a].unique()
        for i in attributes:
            if str(i) == "nan":
                continue
            positive = len(df.loc[(df["Survived"] == 1) & (df[a] == i)])
            negative = len(df.loc[(df["Survived"] == 0) & (df[a] == i)])
            current_tot = b(positive/(positive+negative))
            current = (positive+negative)/total * current_tot
            remainder += current
    #Returns the gain with the formula in AIMA
    return b(p/(n+p)) - remainder

def split(data, attribute): 
    """
    function to calculate the best split for a continous attribute
    params: 
            data = the dataframe you want to calculate the split on
            attribute = the attribute which you want to find the split
    """
    s = data[attribute].unique()
    gains = []
    #Iterates through all the unique variables as elements
    #Checks the attribute in the whole dataset compared to the given element
    #Change the attribute to 0 if attribute from dataset is less than the given element else 1
    #Calculate the gain for each element, the element with the highest gain is the best split
    for el in s: 
        data_copy = data.copy()
        data_copy[attribute] = data_copy[attribute].apply(lambda x: 0 if  x <= el else 1)
        gains.append(gain(data_copy, attribute))
    max_idx, max_val = max(enumerate(gains), key=operator.itemgetter(1))
    return s[max_idx]


class Node:
    """
    class for a Node, which is the node represented in a tree
    contains attributes, a dictionaty with childring and the split
    """
    def __init__(self, attributes, split=None):
        self.attributes = attributes
        self.children = {}
        self.split = split
    
def decision_tree_learning(data, attributes, parent_data=[]):
    """
    function to build a decision tree based on the algoritm from AIMA
    params:
            data = the dataframe you want to build the tree on
            attributes = list of attributes you want use in the build
            parent_data = dataframe from the parent node
    """
    #If the data is empty, the tree is at one of the leaf nodes
    if data.empty:
        #If all of the remaining rows in the parents data survived -> child node is survived
        if parent_data["Survived"].value_counts()[0] == 1:
            return Node("Survived")
        #If all of the remaining rows in the parents data died -> child node is died
        elif parent_data["Survived"].value_counts()[0] == 0:
            return Node("Dead")
        #If there are equally as many dead as survived, return random dead or survived
        else:
            if r.randint(0,1) == 1:
                return Node("Survived")
            else:  
                return Node("Dead")
    #If there are no ramaining Survived in the data
    elif (len(data.loc[data['Survived'] == 1]) == 0):  
        return Node("Dead")
    #If there are no ramainind Dead in the data
    elif (len(data.loc[data['Survived'] == 0]) == 0):
        return Node("Survived")
    #If there are no attributes lefft to calculate on
    elif not attributes:
        #If there are more survived than dead, return survived
        if data["Survived"].value_counts()[0] < data["Survived"].value_counts()[1]:
            return Node("Survived")
        #If there are more dead than survived, return dead
        elif data["Survived"].value_counts()[0] > data["Survived"].value_counts()[1]:
            return Node("Dead")
        #If there are equally as many dead as survived, return random dead or survived
        else:
            if round(r.random()) == 1:
                return Node("Survived")
            else:
                return Node("Dead")
    #The tree build is not at one of the leaf-nodes and have to work its way down
    else: 
        #Make a list of the gains for each attribute and choose the one with the best gain
        gains = []
        for elements in attributes:
            gains.append(gain(data, elements))
        max_gain = max(gains)
        max_gain_index = gains.index(max_gain)
        a = attributes[max_gain_index]
        #Check if the attribute with the most gain is one of the continous ones
        if (a == "Age") or (a == "Fare") or (a == "Parch") or (a == "SibSp"):
            split1 = split(data, a)
            root = Node(a, split1)
            above_split = data[data[a] > split1]
            below_split = data[data[a] <= split1]
            attributes_copy = attributes[:]
            attribute_index = attributes.index(a)
            attributes_copy.pop(attribute_index)
            subtree1 = decision_tree_learning(above_split, attributes_copy, data)
            root.children["Higher than " + str(split1)] = subtree1
            subtree2 = decision_tree_learning(below_split, attributes_copy, data)
            root.children["Lower than " + str(split1)] = subtree2
        #If not continous 
        else:
            root = Node(a)
            for state in data[a].unique():
                attributes_copy = attributes[:]
                updated_data = data.loc[data[a] == state]
                attribute_index = attributes.index(a)
                attributes_copy.pop(attribute_index)
                subtree = decision_tree_learning(updated_data, attributes_copy, data)
                root.children[state] = subtree
    return root




def test(node, row):
    """
    test the accuracy of a single row in the dataset
    compares test set with the decision tree calculated with the decision_tree_learning function
    params:
            node: the given node you want to calculate the accuracy on
            row: the row you want to compare with the node to compare with the row
    """
    #Have to iterate down to a leaf-node to compare with the row
    while ((node.attributes != "Survived") or (node.attributes != "Dead")) & (len(node.children)>0): 
        row_value = row[node.attributes]
        #To handle the continous attributes
        if len(df_test[node.attributes].unique()) > 5:  
            if row_value > node.split:
                test(node.children["Higher than "+str(node.split)], row)
                node = node.children["Higher than " + str(node.split)]
            else:
                test(node.children["Lower than "+str(node.split)], row)
                node = node.children["Lower than " + str(node.split)]
        #If the value is not seen before, random if dead or alive
        elif row_value not in node.children.keys():
            return r.randint(0,1)
        else:
            node = node.children[row_value]
            
    if (node.attributes == "Survived"):
        return 1
    elif (node.attributes == "Dead"):
        return 0

def test_all(data, data_test, attributes):
    """
    test all of the rows in a dataframe, to find the complete accuracy for the decision tree
    params: 
            data: dataframe you want to calculate accuracy on
            attributes: the attributes you want to calculate the tree with
    """
    sum_test = 0
    root = decision_tree_learning(data, attributes)
    for i in range(len(data_test)):
        row = data_test.iloc[i]
        y = test(root, row)
        if y == row.Survived:
            sum_test += 1
    return sum_test/len(data_test)



def draw_graph(root):
    #dot = Digraph("My Decision Tree")
    dot = Digraph("My Decision Tree")
    q = [(root, "")]
    dot.node(root.attributes, label=root.attributes)
    while q:
        parent_node, parent_name = q.pop(0)
        for edge, child in parent_node.children.items():
            q.append((child, parent_name + str(edge)))
            dot.node(parent_name + str(edge) + str(child.attributes), label=str(child.attributes))
            dot.edge(parent_name + parent_node.attributes, parent_name + str(edge) + str(child.attributes), label=str(edge))
    dot.view()


attributes_non = [ "Sex", "Pclass", "Embarked", "Parch"]
attributes_continous = [ "Sex", "Pclass", "Embarked", "Age", "Fare", "SibSp", "Parch"]
df = pd.read_csv("./titanic/train.csv")
df_test = pd.read_csv("./titanic/test.csv")

if __name__ == "__main__":
    # print("----------------------- TASK A ----------------------- ")
    # draw_graph(decision_tree_learning(df, attributes_non))
    # print("The accuracy with non continous attributes: ", test_all(df, df_test, attributes_non))
    print("----------------------- TASK B ----------------------- ")
    draw_graph(decision_tree_learning(df, attributes_continous))
    print("The accuracy with continous attributes: ", test_all(df, df_test, attributes_continous))