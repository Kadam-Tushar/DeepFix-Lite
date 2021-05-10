class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token
    OOV_token = 3   # Out of vocab token

    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD":0,"SOS":1,"EOS":2,"OOV_Token":3}
        self.word2count = {"PAD":0,"SOS":0,"EOS":0,"OOV_Token":0}
        self.index2word = {self.PAD_token: "PAD", self.SOS_token: "SOS", self.EOS_token: "EOS",self.OOV_token: 'OOV_Token'}
        self.num_words = 4
        self.num_sentences = 0  # sentence is list of tokens 
        self.longest_sentence = 0
        self.avg_len = 0 
        

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence:
            sentence_len += 1
            self.add_word(word)
        
        self.avg_len += sentence_len
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]
      
    
    
  
