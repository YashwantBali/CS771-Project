import numpy as np
from collections import defaultdict, Counter
import itertools

class WordSequencer:
    def __init__(self):
        # Initialize bigrams_dict with defaultdict(list)
        self.bigrams_dict = defaultdict(list)
        self.words = None

    def fit(self, word_dict):
        # Clearing and rebuilding the bigrams_dict
        self.bigrams_dict.clear()
        for word in word_dict:
            # Using set to remove duplicates before inserting
            bigrams = set(''.join(bigram) for bigram in zip(word, word[1:]))
            for bigram in bigrams:
                self.bigrams_dict[bigram].append(word)
        self.words = list(word_dict)

    def predict(self, bigrams, lookahead=False):
        if lookahead:
            return self.predict_lookahead(bigrams)
        else:
            return self.predict_greedy(bigrams)

    def predict_greedy(self, bigrams):
        # Counting occurrences of each word using Counter
        word_count = Counter()
        for bigram in bigrams:
            if bigram in self.bigrams_dict:
                word_count.update(self.bigrams_dict[bigram])

        # Filtering words to contain all bigrams
        filtered_words = [word for word in word_count if all(bigram in word for bigram in bigrams)]
        
        # Sorting words by count in descending order
        sorted_words = sorted(filtered_words, key=lambda word: word_count[word], reverse=True)
        return sorted_words[:5]

    def predict_lookahead(self, bigrams):
        # Generating all possible words for each bigram
        word_lists = [self.bigrams_dict[bigram] for bigram in bigrams if bigram in self.bigrams_dict]

        # Generating all possible combinations using itertools.product
        combinations = list(itertools.product(*word_lists))
        unique_combinations = set(''.join(comb) for comb in combinations)

        # Filtering combinations to include only words that contain all bigrams
        filtered_combinations = [word for word in unique_combinations if all(bigram in word for bigram in bigrams)]
        
        # Sorting combinations by length in descending order
        filtered_combinations.sort(key=len, reverse=True)
        return filtered_combinations[:5]

    def lookahead(self, node, depth, max_depth, filtered_combinations):
        if depth >= max_depth or len(node) <= 5:  # Convert to leaf node after 5 levels or when node size is small
            return node
        best_subtree = None
        best_avg_precision = 0
        best_avg_queries = float('inf')
        for query in self.possible_queries:
            child_nodes = self.split_node(node, query)
            subtree = []
            for child_node in child_nodes:
                subtree.extend(self.lookahead(child_node, depth + 1, max_depth, filtered_combinations))
            avg_precision = sum(1 for word in subtree if word in filtered_combinations) / len(subtree)
            avg_queries = sum(self.query_cost(query) for query in subtree) / len(subtree)
            if avg_precision > best_avg_precision or (avg_precision == best_avg_precision and avg_queries < best_avg_queries):
                best_subtree = subtree
                best_avg_precision = avg_precision
                best_avg_queries = avg_queries
        return best_subtree

    def split_node(self, node, query):
        # Split the node into child nodes based on the query
        child_nodes = []
        for word in node:
            if query in word:
                child_nodes.append([word])
            else:
                child_nodes.append([])
        return child_nodes

    def query_cost(self, query):
        # Calculate the cost of the query
        return len(query)

    def is_good_split(self, node, query):
        # Check if the split is good based on entropy reduction
        child_nodes = self.split_node(node, query)
        entropy_reduction = self.entropy(node) - sum(len(child_node) / len(node) * self.entropy(child_node) for child_node in child_nodes)
        return entropy_reduction > 0.5  # Adjust the threshold as needed

    def entropy(self, node):
        # Calculate the entropy of the node
        word_counts = {}
        for word in node:
            word_counts[word] = word_counts.get(word, 0) + 1
        entropy = -sum(count / len(node) * np.log2(count / len(node)) for count in word_counts.values())
        return entropy

def my_fit(words):
    model = WordSequencer()
    model.fit(words)
    return model

def my_predict(model, bigrams, lookahead=False):
    guess_list = model.predict(bigrams, lookahead)
    guess_list = guess_list[:5]
    return guess_list


