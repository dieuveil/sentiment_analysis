import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords


# load doc into memory
def load_doc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# turn a doc into clean tokens
def clean_doc(doc):
  # split into tokens by white space
  tokens = doc.split()
  # prepare regex for char filtering
  re_punc = re.compile('[%s]' % re.escape(string.punctuation))
  # remove punctuation from each word
  tokens = [re_punc.sub('', w) for w in tokens]
  # remove remaining tokens that are not alphabetic
  tokens = [word for word in tokens if word.isalpha()]
  # filter out stop words
  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]
   # filter out short tokens
  tokens = [word for word in tokens if len(word) > 1]
  return tokens

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
  # load doc
  doc = load_doc(filename)
  # clean doc
  tokens = clean_doc(doc)
  # update counts
  vocab.update(tokens)

# load all docs in a directory
def process_docs(directory, vocab):
  # walk through all files in the folder
  for filename in listdir(directory):
  # skip any reviews in the test set
    if filename.startswith('cv9'):
      continue
  # create the full path of the file to open
  path = directory + '/' + filename
  # add doc to vocab
  add_doc_to_vocab(path, vocab)


# define vocab
def vocab_definition():
  vocab = Counter()
  # add all docs to vocab
  process_docs('txt_sentoken/pos', vocab)
  process_docs('txt_sentoken/neg', vocab)
  return vocab

# save list to file
def save_list(lines, filename):
  # convert lines to a single blob of text
  data = '\n'.join(lines)
  # open file
  file = open(filename, 'w')
  # write text
  file.write(data)
  # close file
  file.close()

# Vocab selection
def vocab_selection():
  min_count=2
  vocab = vocab_definition()
  token = [t for t,c in vocab.items() if c >= min_count]
  print(token)
  print(len(vocab))
  save_list(token, 'vocab.txt')


vocab_selection()
