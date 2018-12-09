from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import networkx as nx
import numpy as np
import csv

class NameExtract:
    VOLDEMORT = ["You-Know-Who", "He-Who-Must-Not-Be-Named", "The Dark Lord", "Dark Lord"]

    def __init__(self, bench=6):
        self.st = StanfordNERTagger('./stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                                    './stanford-ner/stanford-ner.jar',
                                    encoding='utf-8')
        self.all_names = defaultdict(int)
        self.bench = bench

    def _update_set(self, new_dict):
        for key in new_dict:
            self.all_names[key] += new_dict[key]
        return

    def voldemort_replace(self, new_text):
        for char_name in self.VOLDEMORT:
            new_text = new_text.replace(char_name, 'Voldemort')
        return new_text

    def extract_names(self, new_text):
        new_text = self.voldemort_replace(new_text)
        tokenized_text = word_tokenize(new_text)
        tokenized_text = [word for word in tokenized_text if word.isalpha()]
        classified_text = self.st.tag(tokenized_text)
        lemmas_list = []
        prev_person = False
        names_num = defaultdict(int)
        for entity in classified_text:
            if entity[1] == 'PERSON':
                if prev_person:
                    lemmas_list[-1] = lemmas_list[-1] + '_' + entity[0]
                else:
                    lemmas_list.append(entity[0])
                prev_person = True
            else:
                if prev_person:
                    names_num[lemmas_list[-1]] += 1
                lemmas_list.append(entity[0])
                prev_person = False
        names_num = self.filter_names(names_num)
        names_num = self.recognize_names(lemmas_list, names_num)
        arr = list(names_num.keys())
        for key in arr:
            if names_num[key] < self.bench:
                del names_num[key]
        for key in names_num:
            self.all_names[key] += names_num[key]
        return lemmas_list, names_num

    def filter_names(self, names_num):
        arr = list(names_num.keys())
        for key in arr:
            if names_num[key] < self.bench:
                del names_num[key]
        return {name.lower() for name in names_num}

    @staticmethod
    def recognize_names(text_arr, names_num):
        res_set = defaultdict(int)
        for name in text_arr:
            if name.lower() in names_num:
                res_set[name.lower()] += 1
        return res_set


class CollectionSummarizer:
    def __init__(self, path_to_collection, mode='books', mapping=None):
        if not os.path.isdir(path_to_collection):
            raise ValueError('Bad path to directory')
        if mode != 'books' and mode != 'scripts':
            raise ValueError('Invalid mode, try "books" or "scripts"')

        self.docs = []
        self.names = []
        self.mode = mode
        self._data = []
        self._summarized_data = None
        self._processed = False
        for filename in tqdm(os.listdir(path_to_collection)):
            with open(os.path.join(path_to_collection, filename), 'r', encoding='utf-8') as file_handler:
                self.docs.append(file_handler.read())
                self.names.append(filename)
        self.mapping = None
        if mapping:
            self.mapping = dict()
            with open(mapping) as file_handler:
                for line in file_handler:
                    line = line.strip()
                    if '-' in line:
                        old, new = line.split('-')
                        if new != 'none':
                            new = new.split(',')
                            self.mapping[old] = new

    def _apply_mapping(self):
        for key in self.mapping:
            if key in self._summarized_data:
                del self._summarized_data[key]
        for lemmas, names_num in self._data:
            for key in self.mapping:
                if key in names_num:
                    del names_num[key]
            for i, lemma in enumerate(lemmas):
                if lemma not in self.mapping:
                    lemmas[i] = (lemmas[i],)
                else:
                    lemmas[i] = self.mapping[lemma]
                    for item in self.mapping[lemma]:
                        self._summarized_data[item] += 1
                        names_num[item] += 1

    def _squeeze_data(self):
        for i, (lemmas, name_sets) in enumerate(self._data):
            tmp_names = []
            tmp_nums = [0]
            tmp_names_num = defaultdict(int)
            for lemma in lemmas:
                if lemma in self._summarized_data:
                    tmp_names.append(lemma)
                    tmp_nums.append(0)
                    tmp_names_num[lemma] += 1
                else:
                    tmp_nums[-1] += 1
            self._data[i] = [tmp_names, tmp_nums, tmp_names_num]
        self._summarized_data.clear()
        for *_, names_num in self._data:
            for name in names_num:
                self._summarized_data[name] += names_num[name]

    def process_docs(self, bench=6):
        extractor = NameExtract(bench=bench)
        for name, doc in tqdm(zip(self.names, self.docs)):
            lemmas, names_num = extractor.extract_names(doc)
            lemmas = [x.lower() for x in lemmas]
            self._data.append([lemmas, names_num])
        self._summarized_data = extractor.all_names
        if self.mapping:
            self._apply_mapping()

        for i, (lemmas, _) in enumerate(self._data):
            self._data[i][0] = [x for arr in lemmas for x in arr]
        self._squeeze_data()
        self._processed = True

    def save_data(self, save_path):
        if not self._processed:
            raise RuntimeError('Process\load data first')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        with open(os.path.join(save_path, 'data.pckl'), 'wb') as file_handler:
            pickle.dump(self._data, file_handler)
        with open(os.path.join(save_path, 'sum_data.pckl'), 'wb') as file_handler:
            pickle.dump(self._summarized_data, file_handler)

    def load_data(self, load_path):
        with open(os.path.join(load_path, 'data.pckl'), 'rb') as file_handler:
            self._data = pickle.load(file_handler)
        with open(os.path.join(load_path, 'sum_data.pckl'), 'rb') as file_handler:
            self._summarized_data = pickle.load(file_handler)
        self._processed = True

    def output_for_mapping(self, mapping_path):
        with open(mapping_path, 'w') as file_handler:
            tmp = sorted(self._summarized_data.items(), key=lambda a: a[1])
            file_handler.write('\n'.join((f'{key}-{value}' for key, value in tmp)))

    @staticmethod
    def generate_edges(names, distances, max_distance=25):
        res_dict = defaultdict(int)
        i = 0
        crr_dist = 0
        for j in range(1, len(names)):
            crr_dist += distances[j]
            while crr_dist > max_distance:
                i += 1
                crr_dist -= distances[i]
            for k in range(i, j):
                if names[i] != names[j]:
                    res_dict[(min(names[i], names[j]), max(names[i], names[j]))] += 1
        return res_dict

    def to_networkx(self, max_distance=25, save_file='graph.nx'):
        if not self._processed:
            raise RuntimeError('Process\load data first')
        new_graph = nx.Graph()
        new_graph.add_nodes_from(((key, {'all_use_num': value}) for key, value in self._summarized_data.items()))
        book_dist = []
        all_dist = defaultdict(int)
        for i, book in enumerate(self._data):
            names, distances, set_names = book
            for key in set_names:
                new_graph.nodes[key][f'{i}_use_num'] = set_names[key]
            edges = self.generate_edges(names, distances, max_distance=max_distance)
            book_dist.append(edges)
            for key in edges:
                all_dist[key] += edges[key]
        new_graph.add_weighted_edges_from((*key, value) for key, value in all_dist.items())
        for edge in new_graph.edges:
            for i in range(len(book_dist)):
                new_graph.edges[edge][f'{i}_weight'] = book_dist[i][edge]
        nx.write_gpickle(new_graph, save_file)

    def to_gephi(self, max_distance=25, out_dir='./gephi'):
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        book_dist = []
        book_nodes = []
        all_dist = defaultdict(int)
        for i, book in enumerate(self._data):
            names, distances, set_names = book
            edges = self.generate_edges(names, distances, max_distance=max_distance)
            book_dist.append(edges)
            book_nodes.append(set_names)
            for key in edges:
                all_dist[key] += edges[key]
        key_to_id = dict()
        with open(os.path.join(out_dir, 'node.csv'), 'w', newline='') as nodes_file:
            nodes_writer = csv.writer(nodes_file)
            csv_row = ['Id', 'Label', 'All_uses'] + [f'Uses_{i}' for i in range(1, 8)]
            nodes_writer.writerow(csv_row)
            for i, (key, value) in enumerate(self._summarized_data.items()):
                key_to_id[key] = i
                csv_row = [i, key, value] + [book[key] for book in book_nodes]
                nodes_writer.writerow(csv_row)

        with open(os.path.join(out_dir, 'edges.csv'), 'w', newline='') as edges_file:
            nodes_writer = csv.writer(edges_file)
            csv_row = ['Source', 'Target', 'Weight'] + [f'edges_{i}' for i in range(1, 8)]
            nodes_writer.writerow(csv_row)
            for key, value in all_dist.items():
                csv_row = [key_to_id[key[0]], key_to_id[key[1]], value] + [book[key] for book in book_dist]
                nodes_writer.writerow(csv_row)


if __name__ == '__main__':
    model = CollectionSummarizer('./data/books', mapping='mapping.txt')
    model.process_docs()
    model.save_data('./output_data')
    for dist in range(5, 31, 5):
        model.to_gephi(max_distance=dist, out_dir=f'out_gephi/{dist}')
