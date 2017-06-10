#All the imports.
import gensim
import codecs
from gensim import corpora, models, similarities
import nltk
import csv
import pandas as pd
import tempfile
import codecs
import csv

model = models.Word2Vec.load(input('Where are your word embeddings coming from, shitbags? '))

word = model.wv.vocab

readfile = input('Where are your fucking morphemes and tags, fuckface???  ')

readinto_COLUMNS = ['word', 'tag']

df_readinto = pd.read_csv(readfile, names=readinto_COLUMNS, skipinitialspace=True)

label_values = list(df_readinto['tags'].values)
vec_values = list(df_readinto['word'].values)
for item in label values:
    if lexeme in item:
        array.append(list(model[vec_values[label_values.index(item)]]))
        array.append(inty)
        databuilder.writerow(array)
        array=[]

#Just some notes when the program is run/
print('Make sure to (1) run build_DNNarray(\'the lexeme you\'re interested in\'), and then (2) run_rabbit_run()')

#Pulls data from the corpus and formats it
def build_DNNarray(lexeme, WORD=word, MODEL=model):
    inty=input('What example number is this? (note: 0 is anything that isn\'t being classified)  ')
    array = []
    label_values = list(df_readinto['tags'].values)
    vec_values = list(df_readinto['word'].values)
    with codecs.open(input('Where is your training data going, fuck-face?? '), 'a', 'utf-8') as csvfile:
        databuilder = csv.writer(csvfile, delimiter=',',
                                 quotechar='|',
                                 quoting=csv.QUOTE_MINIMAL)
        for item in label_values:
            if lexeme in item:
                array.append(list(model[vec_values[label_values.index(item)]]))
                array.append(inty)
                databuilder.writerow(array)
                array=[]
    csvfile.close()


#Components for the DNN. Since we're playing with vectors,
#I ended up de-activating sections relating to categorical columns--they weren't necessary.
COLUMNS = list(range(100)) + ['label']

LABEL_COLUMN = 'label'

CONTINUOUS_COLUMNS = COLUMNS

import tensorflow as tf

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols.items())
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)

model_dir = tempfile.mkdtemp()

features = []

#transforms the inputs into real_value_columns that TF can manipulate.
def make_features(columns=CONTINUOUS_COLUMNS):
    for k in CONTINUOUS_COLUMNS:
        for item in list(range(len(CONTINUOUS_COLUMNS))):
            item = tf.contrib.layers.real_valued_column(k)
            features.append(item)

#The following two lists are place-holders prior to running actual model in run_rabbit_run()
wide_columns=[0]
deep_columns=[]

#The actual model.
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50],
    n_classes=2)

#run this to put everything together after you've built some training data.
def run_rabbit_run():
    make_features()
    wide_columns = features
    deep_columns = []
    df_train = pd.read_csv(input('Can I have your training data? Can I has it? ', names=COLUMNS, skipinitialspace=True))
    #df_test = pd.read_csv('/Users/ZaqRosen/Desktop/ARAPAHO_test_data.csv', names=COLUMNS, skipinitialspace=True, skiprows=1))
    wide_collumns = make_features()
    m.fit(input_fn=train_input_fn, steps=2000)
    #results = m.evaluate(input_fn=eval_input_fn, steps=20)
    #print(results)
    var = tf.trainable_variables()
    print(var)
