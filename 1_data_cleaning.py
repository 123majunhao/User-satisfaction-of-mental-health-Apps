import pandas as pd
import re
import jieba
import gensim


data_all = pd.read_csv(r"D:\zhongwen.csv")


def clear_character(sentence):
    if pd.isna(sentence):
        return ""
    sentence = str(sentence).replace("'", "").replace("’", "")
    pattern = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]")
    line = re.sub(pattern, " ", sentence)
    return " ".join(line.split())


train_text = [clear_character(s) for s in data_all["review"]]
data_all["review_clear"] = train_text

jieba.load_userdict(r"D:\jiebaDic.txt")

train_seg_text = [jieba.lcut(s) for s in train_text]
data_all["review_seg"] = train_seg_text

stop_words_path = r"D:\stop word.txt"


def get_stop_words():
    with open(stop_words_path, "r", encoding="utf-8-sig") as f:
        return {
            item.strip().lower().replace("'", "").replace("’", "")
            for item in f if item.strip()
        }


stopwords = get_stop_words()


def drop_stopwords(line):
    line_clear = []
    for word in line:
        word = word.strip().lower()
        if not word or word in stopwords:
            continue
        line_clear.append(word)
    return line_clear


train_st_text = [drop_stopwords(s) for s in train_seg_text]
data_all["review_st"] = train_st_text


def is_fine_word(words, min_length=2):
    line_clear = []
    rule = re.compile(r"^(?:[\u4e00-\u9fa5]+|[a-z]+)$")
    for word in words:
        if len(word) >= min_length and rule.fullmatch(word):
            line_clear.append(word)
    return line_clear


train_fine_text = [
    is_fine_word(s, min_length=2)
    for s in train_st_text
]
data_all["review_fine"] = train_fine_text

bigram = gensim.models.Phrases(
    train_fine_text, min_count=5, threshold=5
)
trigram = gensim.models.Phrases(
    bigram[train_fine_text], threshold=5
)

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = trigram


def make_bigram(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigram(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


data_words_bigrams = make_bigram(train_fine_text)
data_words_trigrams = make_trigram(train_fine_text)

data_all["bigrams"] = data_words_bigrams
data_all["trigrams"] = data_words_trigrams

data_all.to_csv(r"D:\lda_test1.csv", encoding="utf-8", index=False)
