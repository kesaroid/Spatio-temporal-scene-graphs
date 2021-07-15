from os import times
import random
import more_itertools as mit
import networkx as nx

class QA:
    def __init__(self, ag_video, G, cat2pred, mode='groundtruth') -> None:
        
        self.types = ('repeat', 'range', 'temporal')
        self.wh_words = ('How', 'Which', 'What')
        self.t_verbs = ('before', 'after')
        self.all_questions = {t:[] for t in self.types}
        self.all_qtypes = {t:[] for t in self.types}
        self.all_answers = {t:[] for t in self.types}
        invalid_contacts = ["have_it_on_the_back","not_contacting","other_relationship"]

        self.G = G
        self.cat2pred = cat2pred
        
        self.objs = list(ag_video['objs'])
        self.objs.remove('person')
        self.rels = ag_video['rels']
        self.contacts = [rel for rel in self.rels if (rel in self.cat2pred["contact"]) and (rel not in invalid_contacts)]
        self.timesteps = len(ag_video['frames'])

    def answer(self, que, obj, rel):
        # pos, stats = self.get_pos(que)
        # result = self.pipeline(stats, identifier, que, pos)
        def answer_repeat(que, obj, rel):
            if obj and rel:
                all_paths = list(nx.all_simple_paths(self.G, 'person', obj))
                valid_paths = [i for i in all_paths if i[2] == rel]
                # return len(valid_paths)
            elif obj and not rel:
                valid_paths = list(nx.all_simple_paths(self.G, 'person', obj))
            elif not obj and rel:
                valid_paths = list(nx.all_simple_paths(self.G, 'person', rel))
            
            times = [i[1] for i in valid_paths]
            return len(list(mit.consecutive_groups(set(times))))

        if que.split()[0] == self.wh_words[0]:
            result = answer_repeat(que, obj, rel)
    
        return result
    
    def get_interactions(self, A, B, num):
        return [duo for t, duo in enumerate(A) if self.G.edges[duo][B]]

    def create_questions(self, roiObj):

        def appendo(que, typei, args):
            if que not in self.all_questions[typei]:
                ans = self.answer(*args)
                # print(que, ans)
                self.all_answers[typei].append(ans)
                self.all_questions[typei].append(que)
            else:
                return False

        range_q1 = "Which object was interacted with the most?"
        self.all_questions["range"].append(range_q1)

        # for obj in self.objs:
        #     for rel in self.contacts:
        #         if self.G.has_edge(rel, obj):
        #             repeat_q1 = f"How many times did the Person {rel} the {obj}?"
        #             appendo(repeat_q1, "repeat", [repeat_q1, obj, rel])
        #             repeat_q4 = f"How many objects did the Person {rel}?"
        #             appendo(repeat_q4, "repeat", [repeat_q4, None, rel])

        #     repeat_q2 = f"How many times did the Person interact with {obj}?"
        #     appendo(repeat_q2, "repeat", [repeat_q2, obj, None])
        repeat_q3 = f"How many times did the Person looking_at the {roiObj}?"
        appendo(repeat_q3, "repeat", [repeat_q3, roiObj, 'looking_at'])


        # for i in self.all_questions.keys():
        #     self.all_questions[i] = list(set(self.all_questions[i]))
        
        return self.all_questions, self.all_answers
    
    def get_pos(self, text):
        tokens = nltk.word_tokenize(text)
        pos = nltk.pos_tag(tokens)

        illegal = ('Person', 'object', 'objects', 'interacting', 'interact', 'did', 'was', 'doing', 'times', 'with', 'while', 'at')
        multi_obj = ["closet/cabinet", "cup/glass/bottle", "paper/notebook", "phone/camera", "sofa/couch"]
        multi_rel = ["covered_by", "drinking_from", "have_it_on_the_back", "leaning_on", "lying_on", "sitting_on", "standing_on", "writing_on"]
        out = {'object':[], 'type': None, 'question': None, 'action': []}
        
        for ipos in pos:
            if ipos[0] in self.wh_words:
                out['question'] = ipos[0]
                out['type'] = self.types[self.wh_words.index(ipos[0])]

            elif ipos[0] not in illegal and (ipos[-1] in ['NN', 'NNS'] or ipos[0] in multi_obj):
                out['object'].append(ipos[0])
            
            elif (ipos[-1] in ['VBP', 'VBG', 'VBD', 'VB', 'IN'] or ipos[0] in multi_rel) and ipos[0] not in illegal:
                out['action'].append(ipos[0])
            
            elif ipos[0] in self.t_verbs:
                out['question'] = ipos[0]
            
            elif ipos[-1] == 'CD':
                out['object'] = ipos[0]

        return pos, out 
    
    def pipeline(self, pos, identifier, que, temp):
        result = None
        label = identifier.test([que])
        # print(que, label)
        if label[0] == 1: result = self.grapher.get_number(pos['object'][0], pos['action'][0])
        elif label[0] == 2: result = self.grapher.get_number(pos['object'][0])
        elif label[0] == 3: result = self.grapher.get_number(pos['object'][1], pos['object'][0])
        elif label[0] == 5: result = self.grapher.get_object(mode=self.types[1])
        elif label[0] == 6: result = self.grapher.get_object(pos['object'][0], pos['action'][0], mode=self.types[1])
        elif label[0] == 7 or label[0] == 8: result = self.grapher.get_object(pos['object'][0], pos['action'][0], mode=self.types[2])
        elif label[0] == 9: result = self.grapher.get_object(int(pos['object']), pos['action'][0], mode=self.types[2])

        return result

        # if pos['type'] == self.types[0]:
        #     if len(pos['object']) == 1 and len(pos['action']) == 1: # type 1
        #         return self.grapher.get_number(pos['object'][0], pos['action'][0])
        #     elif len(pos['object']) == 1 and len(pos['action']) == 0: # type 2
        #         return self.grapher.get_number(pos['object'][0])
        #     elif len(pos['object']) == 2 and len(pos['action']) == 0: # type 3
        #         return self.grapher.get_number(pos['object'][1], pos['object'][0])
        #         # type 4 exists
        # elif pos['type'] == self.types[1]:
        #     if len(pos['object']) == 0 and len(pos['action']) == 0: # type 1
        #         return self.grapher.get_object(mode=self.types[1])
        #     elif len(pos['object']) == 1 and len(pos['action']) == 1: # type 2
        #         return self.grapher.get_object(pos['object'][0], pos['action'][0], mode=self.types[1])
        # elif pos['type'] == self.types[2]: 
        #     if pos['question'] in self.t_verbs: # type 1
        #         return self.grapher.get_object(pos['object'][0], pos['question'], mode=self.types[2])
        #     elif pos['object'].isdigit(): # type 2
        #         return self.grapher.get_object(int(pos['object']), pos['action'][0], mode=self.types[2])


from sklearn import model_selection, preprocessing, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas

class Text_classification():
    def __init__(self, videos) -> None:
        self.instances = videos.keys()
        self.questions = []; self.labels = []
        for instance in self.instances:
            video = videos[instance]
            self.questions.extend(video['Questions'])
            self.labels.extend(video['Q_types'])


    def train(self):
        # create a dataframe using texts and lables
        trainDF = pandas.DataFrame()
        trainDF['text'] = self.questions
        trainDF['label'] = self.labels

        # split the dataset into training and validation datasets 
        train_x, val_x, train_y, val_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        val_y = encoder.fit_transform(val_y)

        # ngram level tf-idf 
        self.count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        self.count_vect.fit(trainDF['text'])
        xtrain_count = self.count_vect.transform(train_x)
        xval_count = self.count_vect.transform(val_x)

        self.classifier = naive_bayes.MultinomialNB()
        self.classifier.fit(xtrain_count, train_y)
        
        # predict the labels on validation dataset
        predictions = self.classifier.predict(xval_count)
        
        accuracy = metrics.accuracy_score(predictions, val_y)
        print("NB, WordLevel TF-IDF: ", accuracy)
        return self.classifier

    def test(self, question):
        xque_vector = self.count_vect.transform(question)
        prediction = self.classifier.predict(xque_vector)

        return prediction+1

def create_questions2(video, G, cat2pred):
    objs = list(video['objs'])
    objs.remove('person')
    rels = video['rels']
    timesteps = len(video['frames'])                

    all_questions = []; all_types = []
    t_temporal = []; t_contact = []
    invalid_contacts = ["have_it_on_the_back","not_contacting","other_relationship"]
    range_q1 = "Which object was interacted with the most?"
    all_questions.append(range_q1); all_types.append(5)

    for obj in objs:
        for t in range(timesteps):
            if G.has_edge(t, obj):
                contact = G[t][obj]['contact']
                if contact not in invalid_contacts:
                    repeat_q1 = f"How many times did the Person {contact} the {obj}?"
                    repeat_q4 = f"How many objects did the Person {contact}"
                    if repeat_q1 not in all_questions:
                        all_questions.extend([repeat_q1, repeat_q4]); all_types.extend([1,4])

                    for edge in G.edges(t):
                        if G.get_edge_data(*edge)['contact'] not in invalid_contacts:
                            range_q2 = f"Which other object was the Person interacting with while {contact} the {obj}"
                            if range_q2 not in all_questions:
                                all_questions.extend([range_q2]); all_types.append(6)
                                break

                spatial = G[t][obj]['spatial']
                if spatial == "in_front_of":
                    t_temporal.append(t)

        temporal_q1 = f"What was the Person doing before interacting with the {obj}?"
        temporal_q2 = f"What was the Person doing after interacting with the {obj}?"
        if t_temporal:
            temporal_q3 = f"What was in_front_of the Person at {random.choice(t_temporal)}?"
            all_questions.extend([temporal_q1, temporal_q2, temporal_q3]) 
            all_types.extend([7, 8, 9])
        else:
            all_questions.extend([temporal_q1, temporal_q2]) 
            all_types.extend([7, 8])
        repeat_q2 = f"How many times did the Person interact with {obj}?"
        repeat_q3 = f"How many times did the Person look at the {obj}?"
        all_questions.extend([repeat_q2, repeat_q3])
        all_types.extend([2, 3])
    
    return all_questions, all_types

def ag_metrics(results):
    
    def encode_labels(ser):
        if ser.dtype == 'O':
            return label_encoder.fit_transform(ser)
        else:
            return ser

    label_encoder = preprocessing.LabelEncoder() 

    e_encoded = results.apply(encode_labels)
    print(e_encoded)