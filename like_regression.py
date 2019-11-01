import pandas as pd
from nltk.corpus import stopwords
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

df_transcripts = pd.read_json("transcript1.txt", orient="index")

df_transcripts = df_transcripts.drop(["altmetric_id", "abstract"], axis=1)

df_transcripts = df_transcripts.dropna()
ids_with_transcripts = df_transcripts.id.values

df = pd.read_csv("combined_with_news_n_blogs.csv", header=0)
df= df[df.link !='0']

df_id_video = pd.DataFrame(columns = ["altmetric_id", "video_id", "abstract", "transcript"])

i=0
for index, row in df.iterrows():
    video_ids = ast.literal_eval(row['link'])
    for v_id in video_ids:
        if v_id in ids_with_transcripts:
            df_id_video.loc[i, "video_id"] = v_id
            df_id_video.loc[i, "altmetric_id"] = row['altmetric_id']
            df_id_video.loc[i, "abstract"] = row["abstract"]
    i=i+1
    print (index)
         
for index, row in df_id_video.iterrows():
    df_id_video.loc[index, "transcript"] = df_transcripts.loc[df_transcripts.id == row["video_id"], "transcript"].values[0]
    
df_id_video = df_id_video[df_id_video.abstract != "0"]

vectorizer = TfidfVectorizer()
def cosine_sim(d1, d2):
    tfidf = vectorizer.fit_transform([d1, d2])
    return ((tfidf * tfidf.T).A)[0,1]

df_id_video["score"] = 0
for index, row in df_id_video.iterrows():
    df_id_video.loc[index, "score"] = cosine_sim(row['abstract'], row['transcript'])

df_you = pd.read_json("merged_videoId_vs_Alt_id.txt", orient="index")

cols = ['Category', 'CommentCount', 'Number of Dimensions citations',
       'Video mentions', 'cited_by_accounts_count', 'cited_by_fbwalls_count', 'cited_by_feeds_count',
       'cited_by_gplus_count', 'cited_by_msm_count', 'cited_by_posts_count',
       'cited_by_rdts_count', 'cited_by_tweeters_count',
       'cited_by_videos_count', 'cited_by_wikipedia_count', 'description',
       'dislikes', 'likes', 'video_id', 'number_of_days', 'pubdate', 'subname',
       'subno', 'title', 'views']

df_you.rename(columns={'link':'video_id'}, inplace=True)

df_id_video = pd.merge(df_id_video, df_you[cols], on="video_id")

Category = ['Education', 'People & Blogs', 'Entertainment', 'Howto & Style', 'Science & Technology', 'Sports', 'News & Politics']

df_id_video = df_id_video[df_id_video.Category.isin(Category)]

data = df_id_video.loc[:, ['score', 'Category', 'CommentCount', 'Number of Dimensions citations', 'Video mentions', 
    'cited_by_accounts_count', 'cited_by_fbwalls_count','cited_by_feeds_count', 
       'cited_by_gplus_count', 'cited_by_msm_count','cited_by_posts_count', 'cited_by_rdts_count','cited_by_tweeters_count', 
       'cited_by_videos_count','cited_by_wikipedia_count','number_of_days','subno', 'views']]

data.Category = pd.Categorical(data.Category).codes

target = df_id_video.loc[:, 'likes']

from sklearn.model_selection import train_test_split
traindata,testdata,traintarget,testtarget = train_test_split(data, target, test_size=0.25)
from sklearn.preprocessing import StandardScaler

#feature scaling
sc = StandardScaler()
traindata = sc.fit_transform(traindata)
testdata = sc.transform(testdata)


import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(traindata.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  traindata, traintarget,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,1000])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(testdata, testtarget, verbose=2)
print("Testing set Mean Abs Error: {:5.2f}".format(mae))

test_predictions = model.predict(testdata).flatten()

plt.scatter(testtarget, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
