from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
import jieba

# 每一行為一篇文章
with open('rmsp.txt', 'r', encoding = 'utf8') as f:
    texts = f.readlines()

# 將文章進行斷詞
corpus = []
for text in texts:
    text_cut = jieba.cut(text)
    text_new = ''
    for word in text_cut:
        text_new += word + ' '
    corpus.append(text_new)


vectorizer = CountVectorizer()
# 將語料轉換為詞頻矩陣
X = vectorizer.fit_transform(corpus)
# 取得詞袋內所有詞語
words = vectorizer.get_feature_names()
# print(word)
# print(X.toarray())

transformer = TfidfTransformer()
# 計算 tf-idf權值
tfidf = transformer.fit_transform(X)
weight = tfidf.toarray()
# print(weight)

for i in range(len(weight)):
    counter = Counter()
    for j in range(len(words)):
        counter[words[j]] = weight[i][j]
    print(counter.most_common(3))
