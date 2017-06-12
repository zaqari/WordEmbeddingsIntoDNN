#Note: everything in here is grouped according to hared function.
# That means that everything to do with import files will be up top,
# all data manipulations for generating the training data is grouped
# together, etc.

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
import tensorflow as tf

#In order to build a tag-embedding layer for EVERYTHING.
# Works by formatting input from a separate document of morpheme or word glosses
# compiled by the researcher, and then using the gloss data as the precise label
# used in classifying the vectors from there.
# --Begin Code--
readfile = input('Where is your gloss data coming from?  ')
readinto_COLUMNS = ['word', 'tags']
df_readinto = pd.read_csv(readfile, names=readinto_COLUMNS, skipinitialspace=True)
NCLASSES = len(df_readinto['tags']) + 1
#Inputting the variables from the pre-made word embedding vectors
Word2Vecmodel = input('Where is your word embedding vector model coming from?  ')
training_data_location = input('where are you putting your training data (remember to give the file a name as well, or if you\'re adding new data to an old file, to indicate the old file name)?  ')
model = models.Word2Vec.load(Word2Vecmodel)
word = model.wv.vocab

#Just some notes when the program is run/
print('Make sure to (1) run build_DNNarray(), and then (2) run_rabbit_run()')

#Pulls data from the corpus of word embedding vectors and transforms it into
# a row in a CSV that the machine learning model can use to make predictions of
# some sort.
def build_DNNarray(MODEL=model):
    array = []
    label_values = list(df_readinto['tags'].values)
    vec_values = list(df_readinto['word'].values)
    with codecs.open(training_data_location, 'a', 'utf-8') as csvfile:
        databuilder = csv.writer(csvfile, delimiter=',',
                                 quotechar='',
                                 quoting=csv.QUOTE_MINIMAL)
        for item in label_values:
                array.append(list(model[vec_values[label_values.index(item)]]))
                array.append(label_values.index(item))
                databuilder.writerow(array)
                array=[]
    csvfile.close()


#Components for the DNN. Since we're playing with vectors, I ended up
# de-activating sections relating to categorical columns--they weren't
# necessary.
COLUMNS = list(range(100)) + ['tag1']
LABEL_COLUMN = inty = tf.contrib.layers.sparse_column_with_hash_bucket('tag1', hash_bucket_size=int(1000), dtype=tf.string)
CONTINUOUS_COLUMNS = list(range(100))

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
deep_columns=[0]

#The actual model.
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100],
    n_classes=NCLASSES)

#Run this to put everything together after you've built some training data.
# If you want to run this as a DNN with x-number of layers, change the variable
# for dnn_hidden_units in the variable 'm' right about this function, and switch
# the variable deep_columns to be equal to feautures, and wide_columns to be
# equal to [] in run_rabbit_run(). If you want to evaluate the model, run 
# run_rabbit_run(True) to see evauative statistics of the model. Test data needs
# be built manually, however. There is no function in the script to build it for you.
def run_rabbit_run(Eval=False):
    make_features()
    wide_columns = features
    deep_columns = []
    df_train = pd.read_csv(training_data_location, names=COLUMNS, skipinitialspace=True)
    wide_collumns = make_features()
    m.fit(input_fn=train_input_fn, steps=2000)
    if Eval==True:
        df_test = pd.read_csv(input('Where is your test data coming from?  '), names=COLUMNS, skipinitialspace=True, skiprows=1)
        results = m.evaluate(input_fn=eval_input_fn, steps=20)
        print(results)
        #var = tf.trainable_variables()
        #print(var)
