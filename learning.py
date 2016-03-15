import numpy as np
import urllib

import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import tree
import os

class Learning:

	def __init__(self,training):
		self.training = training
		self.X, self.y, self.target = self.init_data()
		self.model = self.get_model()
		self.tree = self.tree_structure()
		self.num_nodes = self.num_nodes()
		self.children_right, self.children_left = self.children()
		self.feature = self.feature()
		self.threshold = self.threshold()

	def init_data(self):
		d2 = np.loadtxt(self.training, dtype=str, delimiter="\t")

		# print d2[1]
		y = d2[:,0]
		X = d2[:,1:,]

		target = X[0]
		# self.clean_target(target)
		X = X.astype(int)
		X = np.delete(X, 0, 0)
		y = np.delete(y, 0)
		# print X2[1]
		# print Y2[1]
		return X, y, target

	# def clean_target(self,target):
	# 	for item in target:
	# 		if '"' in item:
	# 			rows = item.split('"')
	# 			target[item] = rows[1]
	# 			# print target[item]

	def get_model(self):
		model = DecisionTreeClassifier(criterion='entropy', presort=True)
		model.fit(self.X, self.y)
		return model

	def num_features(self):
		return self.model.n_features_

	# tree properties
	def tree_structure(self):
		return self.model.tree_

	def num_nodes(self):
		return self.tree.node_count

	def predict(self, guess):
		return self.model.predict(guess)

	def children(self):
		children_left = self.tree.children_left
		children_right = self.tree.children_right
		return children_right, children_left

	def feature(self):
		return self.tree.feature

	def threshold(self):
		return self.tree.threshold

	def tree_print(self, *args, **kwargs):
		node_depth = np.zeros(shape=self.num_nodes)
		is_leaves = np.zeros(shape=self.num_nodes, dtype=bool)
		stack = [(0, -1)]  # seed is the root node id and its parent depth
		while len(stack) > 0:
		    node_id, parent_depth = stack.pop()
		    node_depth[node_id] = parent_depth + 1

		    # If we have a test node
		    if (self.children_left[node_id] != self.children_right[node_id]):
		        stack.append((self.children_left[node_id], parent_depth + 1))
		        stack.append((self.children_right[node_id], parent_depth + 1))
		    else:
		        is_leaves[node_id] = True

		print("The binary tree structure has %s nodes and has "
		      "the following tree structure:"
		      % self.num_nodes)
		for i in range(self.num_nodes):
		    if is_leaves[i]:
		        print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
		    else:
		        print("%snode=%s test node: go to node %s if X[:, %s] <= %ss else to "
		              "node %s."
		              % (node_depth[i] * "\t",
		                 i,
		                 self.children_left[i],
		                 self.feature[i],
		                 self.threshold[i],
		                 self.children_right[i],
		                 ))
		return

	def test(self):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0)
		
		leave_id = self.model.apply(X_test)
		data = {}
		print X_test
		for i in leave_id:
			data[leave_id[i]] = X_test[i]

		print data

	def produce_image(self):
		tree.export_graphviz(self.model, out_file='tree.dot', class_names=self.y)
		os.system("dot -Tpng tree.dot -o tree.png")
		os.system("open tree.png")
		return

	def treeToJson(self, feature_names=None):
		decision_tree = self.model
		from warnings import warn

		js = ""

		def node_to_str(tree, node_id, criterion):
			if not isinstance(criterion, sklearn.tree.tree.six.string_types):
				criterion = "impurity"

			value = tree.value[node_id]
			if tree.n_outputs == 1:
				value = value[0, :]

			jsonValue = ', '.join([str(x) for x in value])

			if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
				return '"id": "%s", "criterion": "%s", "impurity": "%s", "samples": "%s", "value": [%s]' \
							% (node_id, 
							criterion,
							tree.impurity[node_id],
							tree.n_node_samples[node_id],
							jsonValue)
			else:
				if feature_names is not None:
					feature = feature_names[tree.feature[node_id]]
				else:
					feature = tree.feature[node_id]

				print feature
				ruleType = "<="
				ruleValue = "%.4f" % tree.threshold[node_id]
				# if "=" in feature:
				# 	ruleType = "="
				# 	ruleValue = "false"
				# else:
				# 	ruleType = "<="
				# 	ruleValue = "%.4f" % tree.threshold[node_id]

				return '"id": "%s", "rule": "%s %s %s", "%s": "%s", "samples": "%s"' \
						% (node_id, 
						feature,
						ruleType,
						ruleValue,
						criterion,
						tree.impurity[node_id],
						tree.n_node_samples[node_id])

		def recurse(tree, node_id, criterion, parent=None, depth=0):
			tabs = "  " * depth
			js = ""

			left_child = tree.children_left[node_id]
			right_child = tree.children_right[node_id]

			js = js + "\n" + \
				tabs + "{\n" + \
				tabs + "  " + node_to_str(tree, node_id, criterion)

			if left_child != sklearn.tree._tree.TREE_LEAF:
				js = js + ",\n" + \
					tabs + '  "left": ' + \
					recurse(tree, \
							left_child, \
							criterion=criterion, \
							parent=node_id, \
							depth=depth + 1) + ",\n" + \
					tabs + '  "right": ' + \
					recurse(tree, \
							right_child, \
							criterion=criterion, \
							parent=node_id,
							depth=depth + 1)

			js = js + tabs + "\n" + \
				tabs + "}"

			return js

		if isinstance(decision_tree, sklearn.tree.tree.Tree):
			js = js + recurse(decision_tree, 0, criterion="impurity")
		else:
			js = js + recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

		print js

	def rules(self, node_index=0):
	    """Structure of rules in a fit decision tree classifier

	    Parameters
	    ----------
	    clf : DecisionTreeClassifier
	        A tree that has already been fit.

	    features, labels : lists of str
	        The names of the features and labels, respectively.

	    """
	    clf = self.model
	    features = self.target
	    labels = self.y
	    node = {}
	    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
	        count_labels = zip(clf.tree_.value[node_index, 0], labels)
	        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
	                                  for count, label in count_labels))
	    else:
	        feature = features[clf.tree_.feature[node_index]]
	        threshold = clf.tree_.threshold[node_index]
	        node['name'] = '{} > {}'.format(feature, threshold)
	        left_index = clf.tree_.children_left[node_index]
	        right_index = clf.tree_.children_right[node_index]
	        node['children'] = [self.rules(right_index),
	                            self.rules(left_index)]
	    return node



