

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import re
import jieba
import gensim
from pprint import pprint
import gensim.corpora as corpora
import pyLDAvis

try:
    import pyLDAvis.gensim_models as gensimvis
except ImportError:
    import pyLDAvis.gensim as gensimvis


# Change CORPUS_NAME and INPUT_PATH for the current country-phase corpus.
CORPUS_NAME = "china_2009_2017"
TOPIC_COUNTS = {
    "china_2009_2017": 8,
    "china_2018_2025": 12,
    "us_2009_2017": 9,
    "us_2018_2025": 10,
}

INPUT_PATH = r"D:\zhongwen.csv"
STOPWORDS_PATH = r"D:\stop word.txt"
USER_DICTIONARY_PATH = r"D:\jiebaDic.txt"
OUTPUT_PATH = r"D:\data_result.csv"
DOMINANT_TOPIC_OUTPUT_PATH = r"D:\df_topic_sents_keywords.csv"
NUM_TOPICS = TOPIC_COUNTS[CORPUS_NAME]


# Load the review dataset.
data_all = pd.read_csv(INPUT_PATH)


# Remove punctuation and emoticons while retaining Chinese, English, and digits.
def clear_character(sentence):
    sentence = "" if pd.isna(sentence) else str(sentence)
    pattern = re.compile(r"[^\u4e00-\u9fa5A-Za-z0-9]")
    line = re.sub(pattern, "", sentence)
    return "".join(line.split())


train_text = [clear_character(review) for review in data_all["review"]]


# Load the custom jieba dictionary and tokenize the reviews.
jieba.load_userdict(USER_DICTIONARY_PATH)
train_seg_text = [jieba.lcut(text) for text in train_text]


# Load stopwords.
def get_stop_words():
    with open(STOPWORDS_PATH, "r", encoding="utf-8-sig") as file:
        return {item.strip() for item in file if item.strip()}


stopwords = get_stop_words()


def drop_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]


train_st_text = [drop_stopwords(tokens) for tokens in train_seg_text]


# Retain Chinese tokens containing at least two characters.
def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")
    for word in words:
        if len(word) >= min_length and re.search(rule, word):
            line_clear.append(word)
    return line_clear


train_fine_text = [
    is_fine_word(tokens, min_length=2) for tokens in train_st_text
]


# Build bigram and trigram models as in the original workflow.
bigram = gensim.models.Phrases(train_fine_text, min_count=5, threshold=5)
trigram = gensim.models.Phrases(bigram[train_fine_text], threshold=5)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigram(texts):
    return [bigram_mod[document] for document in texts]


def make_trigram(texts):
    return [trigram_mod[document] for document in texts]


data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)


# Build the dictionary and bag-of-words corpus.
# The original model used train_fine_text rather than the phrase outputs.
id2word = corpora.Dictionary(train_fine_text)
texts = train_fine_text
corpus = [id2word.doc2bow(text) for text in texts]


# Fit the LDA model using the parameter settings reported in the supplement.
lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=NUM_TOPICS,
    random_state=100,
    update_every=1,
    chunksize=500,
    passes=20,
    eta=0.1,  # beta in the manuscript
    alpha=0.1,
    per_word_topics=True,
)

print(lda_model.alpha)
print(lda_model.eta)
pprint(
    lda_model.print_topics(
        num_topics=NUM_TOPICS,
        num_words=30,
    )
)


# Export a numeric document-topic probability matrix.
doc_topic = []
for document in corpus:
    distribution = dict(
        lda_model.get_document_topics(
            bow=document,
            minimum_probability=0,
            per_word_topics=False,
        )
    )
    doc_topic.append(
        [distribution.get(topic_id, 0.0) for topic_id in range(NUM_TOPICS)]
    )

topic_score = pd.DataFrame(
    doc_topic,
    columns=[f"Topic {topic_id}" for topic_id in range(NUM_TOPICS)],
)

result = data_all.reset_index(drop=True).join(topic_score)
result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


# Export the dominant topic and its probability for each review.
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
    rows = []
    for document, content in zip(corpus, texts):
        distribution = ldamodel.get_document_topics(
            bow=document,
            minimum_probability=0,
            per_word_topics=False,
        )
        topic_num, prop_topic = max(distribution, key=lambda item: item[1])
        rows.append(
            {
                "Dominant_Topic": int(topic_num),
                "Perc_Contribution": round(float(prop_topic), 4),
                "Processed_Text": content,
            }
        )
    return pd.DataFrame(rows)


df_topic_sents_keywords = format_topics_sentences()
df_topic_sents_keywords.to_csv(
    DOMINANT_TOPIC_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# Visualize the fitted LDA model.
visualization = gensimvis.prepare(lda_model, corpus, id2word)
pyLDAvis.show(visualization)
