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

	def get_question(self, iden):

		option = {
		# Frequent urination
			0:"Do you urinate frequently?",
		# "Do you urinate frequency?"
		# Hoarse voice
			1:"Is your voice hoarse?",
		# Lacrimation
			2:"Do your eyes tear up excessively?",
		# Stomach bloating
			3:"Is your stomach bloated?",
		# Too little hair
			4:"Is there little hair in the region?",
		# Melena
			5:"Is your stool black in color?",
		# Leg swelling
			6:"Is your leg(s) swollen?",
		# Burning chest pain
			7:"Are you experiencing burning chest pain?",
		# Jaundice
			8:"Is a portion of your skin yellow?",
		# Pain in testicles
			9:"Are you experiencing pain in your testicles?",
		# Decreased appetite
			10:"Do you have decreased appetite?",
		# Hemoptysis
			11:"Are you coughing blood?",
		# Lip swelling
			12:"Is your lip swollen?",
		# Back pain
			13:"Do you have back pain?",
		# Elbow swelling
			14:"Is there swelling in your elbow?",
		# Mouth symptoms
			15:"Is it in your mouth?",
		# Problems with movement
			16:"Are you experiencing problems with movement",
		# Weakness
			17:"Are you feeling weak?",
		# Itchy scalp
			18:"Is your scalp itchy?",
		# Dizziness
			19:"Do you feel dizzy?",
		# Vomiting blood
			20:"Are you vomiting blood?",
		# Ear symptoms
			21:"Are the symptoms in your ear?",
		# Palpitations
			22:"Are you experiencing palpitations?",
		# Drug abuse
			23:"Do you use drugs?",
		# Mouth pain
			24:"Are you experiencing mouth pain?",
		# Feeling cold
			25:"Do you constantly feel cold?",
		# Infertility
			26:"Are you infertile?",
		# Pain of the anus
			27:"Are you experiencing pain around your anus?",
		# Difficulty eating
			28:"Do you find it difficult to eat?",
		# Abnormal involuntary movements
			29:"Are you experiencing abnormal involuntary movement?",
		# Flu-like syndrome
			30:"Do you feel 'flu-ish'?",
		# Loss of sensation
			31:"Are you experiencing loss of sensation in the area?",
		# "Muscle cramps, contractures, or spasms"
			32:"Are you experiencing muscle cramps, muscle contractions or spasms?",
		# Frontal headache
			33:"Are you experiencing headaches at the front of your head?",
		# Suprapubic pain
			34:"Are you experiencing at the front of your pelvis?",
		# Depressive or psychotic symptoms
			35:"Do you often feel severly depressed or distant from other people?",
		# Knee swelling
			36:"Is there swelling in your knee?",
		# Blood in stool
			37:"Is there blood in your stool?",
		# Back mass or lump
			38:"Is there a lump or mass on your back?",
		# Fainting
			39:"Have you recently fainted?",
		# Mouth dryness
			40:"Is your mouth dry?",
		# Mass on eyelid
			41:"Is there a lump on your eyelid?",
		# Acne or pimples
			42:"Is there acne/pimples in the are?",
		# Arm weakness
			43:"Is there weakness in your arm?",
		# Ringing in ear
			44:"Are your ears ringing?",
		# Hand or finger pain
			45:"Do you have pain in your hands or fingers?",
		# Foot or toe swelling
			46:"Do you have swelling in your foot or toes?",
		# Decreased heart rate
			47:"Do you have a decreased heart rate?",
		# Incontinence of stool
			48:"Have you lost control of your stool?",
		# Sinus problems
			49:"Do you have sinus problems?",
		# Pain or soreness of breast
			50:"Are your breasts sore?",
		# Low urine output
			51:"Do you have low urine output?",
		# Mouth ulcer
			52:"Do you have a sore (or ulcer) inside your mouth?",
		# Lower abdominal pain
			53:"Do you have lower abdominal pain?",
		# Skin growth
			54:"Are there callus', corns or skin tags in the area?",
		# Unusual color or odor to urine
			55:"Is there an Unusual colour or odor to your urine?",
		# Obsessions and compulsions
			56:"Do you have obsessiosn and compulsions?",
		# 57Neck mass
			57:"Is there a mass in your neck?",
		# 58Low back symptoms
			58:"Are the symptoms in your lower back?",
		# 59Eye redness
			59:"Are your eyes red?",
		# 60Visual disturbance
			60:"Are you experiencing visual disturbances?",
		# 61Vaginal bleeding after menopause
			61:"Do you have vaginal bleeding after menopause?",
		# 62Headache
			62:"Are you experiencing headaches?",
		# 63Diminished hearing
			63:"Has your hearing diminished?",
		# 64Long menstrual periods
			64:"Do you have long menstrual periods?",
		# 65Irregular appearing scalp
			65:"Do you have a dry scalp and/or a rash on your scalp?",
		# 66Hand or finger lump or mass
			66:"Is there a lump present on your hands or fingers?",
		# 67Leg cramps or spasms
			67:"Are you experiencing leg cramps or spasms?",
		# 68Difficulty in swallowing
			68:"Are you having difficulty swallowing?",
		# 69Hysterical behavior
			69:"Are you behaving hysterically?",
		# 70Coughing up sputum
			70:"Are you coughing up mucus?",
		# 71Neck swelling
			71:"Is your neck swollen?",
		# 72Sleepiness
			72:"Do you feel drowsy/sleepy?",
		# 73Vaginal bleeding
			73:"Do you have vaginal bleeding?",
		# 74Irregular belly button
			74:"Is your belly button protruding irregularly?",
		# 75Seizures
			75:"Are you experiencing seizures?",
		# 76Irregular appearing nails
			76:"Do you have irregularly appearing nails?",
		# 77Blood clots during menstrual periods
			77:"Do you have bloog clots during mentrual periods?",
		# 78Menopausal symptoms
			78:"Are you undergoing menopause?",
		# 79Foot and toe symptoms
			79:"Are the symptoms present in your foot or toes?",
		# 80Ache all over
			80:"Do you ache all over?",
		# 81Symptoms of eye
			81:"Are the symptoms in your eye(s)?",
		# 82Swelling of scrotum
			82:"Is your scrotum swollen?",
		# 83Groin pain
			83:"Are you experiencing groin pain?",
		# 84Spotting or bleeding during pregnancy
			84:"Are you experiencing spotting or bleeding while pregnant?",
		# 85Elbow symptoms
			85:"Are the symptoms in or around your elbow?",
		# 86Nasal congestion
			86:"Do you have nasal congestion?",
		# 87Abnormal breathing sounds
			87:"Do you have abnormal breating sounds?",
		# 88Skin on leg or foot looks infected
			88:"Does the skin on your leg or foot look infected?",
		# 89Bedwetting
			89:"Are you wetting the bed?",
		# 90Gum pain
			90:"Are you experiencing pain or soreness in your gums?",
		# 91Fatigue
			91:"Are you tired, fatigued or feel as if you have no energy?",
		# 92Excessive appetite
			92:"Are you constantly hungry?",
		# 93Excessive urination at night
			93:"Do you excessively urinate at night?",
		# 94Joint pain
			94:"Are you experiencing joint pain?",
		# 95Hostile behavior
			95:"Do you behave violently or agressively?",
		# 96Wrist pain
			96:"Do you have wrist pain?",
		# 97Neurological symptoms
			97:"Are you experiencing neurological symptoms?",
		# 98Heartburn
			98:"Do you have heartburn?",
		# 99Stiffness all over
			99:"Do you feel stiff all over?",
		# 100 Pelvic pressure
			100:"Do you feel pressure in your pelvic region?",
		# 101: Shoulder symptoms
			101:"Are the symptoms in your shoulder?",
		# 102: Painful menstruation
			102:"Do you experience painful mentruations?",
		# 103: Rectal bleeding
			103:"Are you bleeding from your rectum?",
		# 104: Throat swelling
			104:"Is your throat swollen?",
		# 105: Thirst
			105:"Are you constantly thirsty?",
		# 106: Musculoskeletal deformities
			106:"Do you have a crooked back?",
		# 107: Swollen or red tonsils
			107:"Do you have swollen or red tonsils?",
		# 108: Leg symptoms
			108:"Are you experiencing your symptoms in your leg?",
		# 109: Shoulder stiffness or tightness
			109:"Is your shoulder stiff or tight?",
		# 110: Leg lump or mass
			110:"Do you have a lump or mass in your leg?",
		# 111: Hand or finger weakness
			111:"Do your hands or fingers feel weak?",
		# 112: White discharge from eye
			112:"Do you have white discharge from your eye?",
		# 113: Pain during pregnancy
			113:"Are you experiencing pain while pregnant?",
		# 114: Back symptoms
			114:"Are the symptoms you are experiencing in your back?",
		# 115: Itchiness of eye
			115:"Are your eye(s) itchy?",
		# 116: Sneezing
			116:"Are you constantly sneezing?",
		# 117: Double vision
			117:"Do you have double vision?",
		# 118: Redness in or around nose
			118:"Do you have redness in or around your nose?",
		# 119: Penis symptoms
			119:"Are the symptoms you are experiencing around your penis?",
		# 120: Problems during pregnancy
			120:"Are you experiencing problems while pregnant?",
		# 121: Eye burns or stings
			121:"Does your eye burn or sting?",
		# 122: Swollen lymph nodes
			122:"Do you have swollen lymph nodes?",
		# 123: Ankle symptoms
			123:"Are you experiencing ankle symptoms?",
		# 124: Vaginal itching
			124:"Are you experiencing vaginal itchiness?",
		# 125: Eye discharge
			125:"Is there discharge coming out of your eye(s)?",
		# 126: Sweating
			126:"Are you constantly sweating and/or do you have experience cold sweats?",
		# 127: Bleeding or discharge from nipple
			127:"Are you bleeding and/or is there discharge from your nipple?",
		# 128: Side pain
			128:"Are you experiencing side pain?",
		# 129: Arm lump or mass
			129:"Do you have a lump or mass in your arm?",
		# 130: Skin looks infected
			130:"Does your skin look infected?",
		# 131: Involuntary urination
			131:"Are you experiencing involuntary urination?",
		# 132: Hand or finger swelling
			132:"Is your hand or finger swollen?",
		# 133: Low self-esteem
			133:"Do you have low self-esteem?",
		# 134: Spots or clouds in vision
			134:"Do you have spots or clouds in your vision?",
		# 135: Cramps and spasms
			135:"Are you experiencing cramps and spasms?",
		# 136: Arm stiffness or tightness
			136:"Is your arm stiff or tight?",
		# 137: Lump or mass of breast
			137:"Is there a lump or mass in your breast?",
		# 138: Pelvic pain
			138:"Are you experiencing pelvic pain?",
		# 139: Itching of skin
			139:"Is you skin itchy?",
		# 140: Jaw swelling
			140:"Is your jaw swollen?",
		# 141: Sharp abdominal pain
			141:"Are you experiencing sharp abdominal pain?",
		# 142: Symptoms of the female reproductive system
			142:"Are your symptoms in your genital region?",
		# 143: Tongue lesions
			143:"Do you have tongue lesions?",
		# 144: Abnormal appearing skin
			144:"Do you have abnormally appearing skin?",
		# 145: Symptoms of bladder
			145:"Are the symptoms you are experiencing in your bladder?",
		# 146: Vaginal pain
			146:"Do you have vaginal pain?",
		# 147: Arm symptoms
			147:"Do you have arm symptoms?",
		# 148: Lump in throat
			148:"Do you have a lump in your throat?",
		# 149: Neck pain
			149:"Are you experiencing neck pain?",
		# 150: Blood in urine
			150:"Do you have blood in your urine?",
		# 151: Leg pain
			151:"Are you experiencing leg pain?",
		# 152: "Skin dryness, peeling, scaliness, or roughness"
			152:"Do you have skin dryness, peeling, scaliness or roughness?",
		# 153: Neck stiffness or tightness
			153:"Do you have neck stiffnes or tightness?",
		# 154: Breathing fast
			154:"Are you rapidly breathing?",
		# 155: Sharp chest pain
			155:"Are you experiencing chest pain?",
		# 156: Uterine contractions
			156:"Are you experiencing uterine contractions?",
		# 157: Warts
			157:"Do you have warts?",
		# 158: Delusions or hallucinations
			158:"Are you experiencing delusions or hallucinations?",
		# 159: Ankle swelling
			159:"Are your ankles swollen?",
		# 160: Disturbance of memory
			160:"Are you experiencing memory disturbances?",
		# 161: Lymphedema
			161:"Do you have swollen lymph nodes or vericose veins?",
		# 162: Eyelid swelling
			162:"Is your eyelid swollen?",
		# 163: Blindness
			163:"Are you experiencing blindness?",
		# 164: Diarrhea
			164:"Do you have Diarrhea?",
		# 165: Groin mass
			165:"Is there a mass in your groin?",
		# 166: Vaginal discharge
			166:"Do you have vaginal discharge?",
		# 167: Swollen eye
			167:"Is your eye swollen?",
		# 168: Ear pain
			168:"Are you experiencign ear pain?",
		# 169: Sore throat
			169:"Do you have a sore throat?",
		# 170: Peripheral edema
			170:"Are both of your ankles or both of your legs swelling?",
		# 171: Antisocial behavior
			171:"Do you have antisocial behaviour?",
		# 172: Neck symptoms
			172:"Are the symptoms you are experiencing in your neck?",
		# 173: Low back pain
			173:"Are you experiencing lower back pain?",
		# 174: Skin swelling
			174:"Are you experiencing skin swelling?",
		# 175: Symptoms of the face
			175:"Are your symptoms in your face?",
		# 176: Restlessness
			176:"Are you constantly restless?",
		# 177: Pain during intercourse
			177:"Do you experience pain during intercourse?",
		# 178: Penis redness
			178:"Is your penis appearing red?",
		# 179: Fears and phobias
			179:"Do you have intense fears and phobias?",
		# 180: Changes in stool appearance
			180:"Has there recently been changes in your stool appearance or colour?",
		# 181: Constipation
			181:"Are you experiencing constipation?",
		# 182: Throat feels tight
			182:"Does your throat feel tight?",
		# 183: Skin pain
			183:"Are you experiencing skin pain?",
		# 184: Difficulty speaking
			184:"Are you having difficulty speaking?",
		# 185: Nausea
			185:"Do you feel nauseous?",
		# 186: Arm swelling
			186:"Is your arm swollen?",
		# 187: Lack of growth
			187:"Are you experiencing lack of growth?",
		# 188: Burning abdominal pain
			188:"Are you experiencing burning abdominal pain?",
		# 189: Leg weakness
			189:"Does your leg feel weak?",
		# 190: Leg stiffness or tightness
			190:"Does your leg feel stiff or tight?",
		# 191: Weight loss
			191:"Have you recently lost weight?",
		# 192: Behavioral disturbances
			192:"Do you feel agitated, a lack of control or have a similar behavior problem?",
		# 193: Sinus congestion
			193:"Are you experiencing sinus congestion?",
		# 194: Wheezing
			194:"Are you wheezing?",
		# 195: Difficulty breathing
			195:"Do you have difficulty breathing?",
		# 196: Sleep disturbance
			196:"Do you have frequent nightmares/ night terrors?",
		# 197: Hand or finger stiffness or tightness
			197:"Do your hands/ fingers feel stiff or tight?",
		# 198: Wrist swelling
			198:"Is your wrist swollen?",
		# 199: Excessive anger
			199:"Are you excessively angry/agresive?",
		# 200: Facial pain
			200:"Are you experiencing facial pain?",
		# 201: Pulling at ears
			201:"Do you feel as if your ears are being pulled?",
		# 202: Vaginal symptoms
			202:"Are the symptosm you are experiencing in your vaginal region?",
		# 203: Congestion in chest
			203:"Are you experiencing chest congestion?",
		# 204: Problems with shape or size of breasts
			204:"Are there problems with the size or shape of your breasts?",
		# 205: Insomnia
			205:"Do you have difficulty sleeping?",
		# 206: Knee weakness
			206:"Does your knee feel weak?",
		# 207: Back cramps or spasms
			207:"Are you experiencign back cramps or spasms?",
		# 208: Bleeding from ear
			208:"Are you bleeding from your ear?",
		# 209: Diaper rash
			209:"Do you have a diaper rash?",
		# 210: Changes in bowel function
			210:"Have you experienced recent changes in your bowel function?",
		# 211: Skin lesion
			211:"Is ther a lesion/sore on your skin?",

		# 212: Painful sinuses
			212:"Are you experiencing painful sinuses?",
		# 213: Skin moles
			213:"Are there moles on your skin?",
		# 214: Itchy ear(s)
			214:"Are your ears itchy?",
		# 215: Swollen tongue
			215:"Is your tongue swollen?",
		# 216: Eyelid lesion or rash
			216:"Is there a lesion/sore or rash on your eyelid?",
		# 217: Penile discharge
			217:"Do you have discharge from your penis?",
		# 218: Feeling hot
			218:"Do you feel hot?",
		# 219: Increased heart rate
			219:"Is your heart racing?",
		# 220: Chest pain
			220:"Do you have chest pain?",
		# 221: Hip symptoms
			221:"Are your symptoms in your hip?",
		# 222: Infant spitting up
			222:"Is your infant spitting up?",
		# 223: Irregular heartbeat
			223:"Do you have an irregular heartbeat?",
		# 224: Hip pain
			224:"Are you experiencing hip pain?",
		# 225: Diminished vision
			225:"Do you have blurred vison, have trouble reading or trouble focusing?",
		# 226: Symptoms of the breast
			226:"Are your symptoms in your breast?",
		# 227: Lower body pain
			227:"Are you experiencing lower body pain?",
		# 228: Skin on arm or hand looks infected
			228:"Does the skin on your arm or hand look infected?",
		# 229: Elbow pain
			229:"Do you ave pain in your elbow?",
		# 230: Impotence
			230:"Are you unable to get an erection?",
		# 231: Slurring words
			231:"Do you slur your words?",
		# 232: Allergic reaction
			232:"Do you have hives and/or trouble breathing?",
		# 233: Recent pregnancy
			233:"Were you recently pregnant?",
		# 234: Anxiety and nervousness
			234:"Are you anxious or nervous?",
		# 235: Intermenstrual bleeding
			235:"Are you bleeding in between menstrual periods?",
		# 236: Hurts to breath
			236:"Does it hurt to breath?",
		# 237: Foot or toe pain
			237:"Do you have foot or toe pain?",
		# 238: Temper problems
			238:"Do you have a temper problem/ are you angry often?",
		# 239: Painful urination
			239:"Is it painful to urinate?",
		# 240: Fever
			240:"Do you feel feverish/ are you warm to the touch?",
		# 241: Absence of menstruation
			241:"Have you not had a menstrual cycle in a while?",
		# 242: Knee symptoms
			242:"Are you experiencing knee symptoms?",
		# 243: Irritable infant
			243:"Do you have an irritable infant?",
		# 244: Nosebleed
			244:"Do you have a nosebleed?",
		# 245: Wrist symptoms
			245:"Do you have wrist symptoms?",
		# 246: Unpredictable menstruation
			246:"Is your menstrual cycle unpredictable?",
		# 247: Shoulder pain
			247:"Do you have shoulder pain?",
		# 248: Smoking problems
			248:"Do you smoke a lot?",
		# 249: Paresthesia
			249:"Do you feel burning or prickling on your skin?",
		# 250: Toothache
			250:"Do you have a toothache?",
		# 251: Arm pain
			251:"Do you have arm pain?",
		# 252: Pelvic symptoms
			252:"Are your symptoms in or around your pelvis?",
		# 253: Foreign body sensation in eye
			253:"Do you feel like there is something in your eye?",
		# 254: Chest tightness
			254:"Are you experiencing tightness in your chest?",
		# 255: Ankle pain
			255:"Do you have ankle pain?",
		# 256: Hand and finger symptoms
			256:"Are your symptoms in your hand/finger?",
		# 257: Abnormal movement of eyelid
			257:"Are you excessively blinking, is your eyelid drooping or are you squinting excessively?",
		# 258: Feeling ill
			258:"Do you generall feel ill?",
		# 259: Heavy menstrual flow
			259:"Do you have a heavy mentrual flow?",
		# 260: Bones are painful
			260:"Are your bones painful?",
		# 261: Apnea
			261:"Do you snore loudly or stop breathing while you sleep?",
		# 262: Throat irritation
			262:"Does your through feel itchy/scratchy?",
		# 263: Symptoms of prostate
			263:"Are your symptoms in or around your prostate?",
		# 264: Mass in scrotum
			264:"Do you have a mass in your scrotum?",
		# 265: Symptoms of the skin
			265:"Are your symptoms on your skin?",
		# 266: Focal weakness
			266:"Do you feel weak on one side of your body?",
		# 267: Abnormal growth or development
			267:"Are you experiecning abnormal growth or development?",
		# 268: Disturbance of smell or taste
			268:"Do you have problems with your sense of smell or taste?",
		# 269: Nose symptoms
			269:"Are your symptoms in your nose?",
		# 270: Knee stiffness or tightness
			270:"Are you experiencing knee stiffness or tightness?",
		# 271: Plugged feeling in ear
			271:"Does your ear(s) feel plugged?",
		# 272: Symptoms of the kidneys
			272:"Are your symptoms in your kidney/lower back?",
		# 273: Symptoms of the anus
			273:"Are your symptoms in or around your anus?",
		# 274: Cough
			274:"Do you have a cough?",
		# 275: Regurgitation
			275:"Are you regurgitating?",
		# 276: Coryza
			276:"Do you have a stuffy nose?",
		# 277: Pain in eye
			277:"Do you have pain in your eye?",
		# 278: Vomiting
			278:"Are you vomiting?",
		# 279: Fluid retention
			279:"Are you retaining a lot of fluid?",
		# 280: Hot flashes
			280:"Are you experiencing hot flashes?",
		# 281: Upper abdominal painful
			281:"Do you have upper abdominal pain?",
		# 282: Skin rash
			282:"Do you have a skin rash?",
		# 283: Rib pain
			283:"Do you have pain in your ribs?",
		# 284: Swollen abdomen
			284:"Is your abdomen swollen?",
		# 285: Infant feeding problem
			285:"Are you having trouble feeding your infant?",
		# 286: Knee pain
			286:"Do you have knee pain?",
		# 287: Redness in ear
			287:"Is there redness in your ear?",
		# 288: Flatulence
			288:"Are you overly flatulent?",
		# 289: Chills
			289:"Do you have chills?",
		# 290: Depression
			290:"Are you depressed/ are you sad all of the time?",
		# 291: Weight gain
			291:"Have you recently undergone weight gain?",
		# 292: Muscle pain
			292:"Do you have pain in your muscles?",
		# 293: Fluid in ear
			293:"Is there fluid in your ear. does your ear feel full?",
		# 294: Penis pain
			294:"Does your penis hurt?",
		# 295: Abusing alcohol
			295:"Do you drink alcohol excessively?",
		# 296: Retention of urine
			296:"Are you unable to urinate or do you have trouble urinating?",
		# 297: Drainage in throat
			297:"Is your nose draining mucus into your throat?",
		# 298: Skin irritation
			298:"Is your skin irritated/ do you have a rash?",
		# 299: Shortness of breath
			299:"Are you experiencing shortness of breath?"

		}
		return option[iden]



