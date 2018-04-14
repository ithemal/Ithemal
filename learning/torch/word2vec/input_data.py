import numpy
from collections import deque
numpy.random.seed(12345)


#this holds summaries of data, but does not hold data itself
#and processing functions

class Data:
    
    def __init__(self,
                 num_skips,
                 skip_window,
                 min_count,
                 n_words):
        self.data_index = 0
        self.n_words = n_words
        self.min_count = min_count
        self.num_skips = num_skips
        self.skip_window = skip_window
        
    def get_common_words(self, words):
        
        word_frequency = dict()
        for w in words:
            try:
                word_frequency[w] += 1
            except:
                word_frequency[w] = 1
        self.word2id = dict()
        self.id2word = dict()
        wid = 1

        #filter and assign new ids
        if n_words == None: #we are filtering based on min_count
            self.word_frequency = dict()
            for w, c in word_frequency.items():
                if c < min_count:
                    self.word_frequency[0] += c
                else:
                    self.word2id[w] = wid
                    self.id2word[wid] = w
                    self.word_frequency[wid] = c
                    wid += 1
            
        elif min_count == None: #we are filtering based on the first n most common words
            sorted_freq = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)
            for i,(k,v) in enumerate(sorted_freq):
                if i >= n_words:
                    self.word_frequency[0] += c
                else:
                    self.word2id[w] = wid
                    self.id2word[wid] = w
                    self.word_frequency[wid] = c
                    wid += 1

        self.word_count = len(self.word2id)

        data = []
        for w in words:
            data.append(word2id.get(w,0))

        return data

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.word_frequency.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

        # num skips = amount of context words sampled for given target word  
    def generate_pos_pairs(self, data, batch_size):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        batch = []
        labels = []
        buffer = collections.deque(maxlen=span)
        if self.data_index + span > len(data):
            self.data_index = 0
        buffer.extend(data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch.append(buffer[skip_window])
                labels.append(buffer[context_word])
            if self.data_index == len(data):
                buffer.clear()
                buffer.extend(data[:span])
                self.data_index = span
            else:
                buffer.append(data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(data) - span) % len(data)
    
        return batch, labels

    
    def generate_neg_words(self, batch_size, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(batch_size, count)).tolist()
        return neg_v


    
