import urllib.request
import re
import pandas as pd
import json
import os
import sys
import time
import requests
import numpy as np
import nltk
from textblob import Word
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

search_keyword="football"
html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword)
video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
video_ids = [*set(video_ids)]
video_ids = video_ids[0:5]
urls=[]
for id in video_ids:
    urls.append("https://www.youtube.com/watch?v="+id)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

YOUTUBE_COMMENTS_AJAX_URL = 'https://www.youtube.com/comment_service_ajax'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'

FILE_NAME = 'ytb_comments.csv'

SORT_BY_POPULAR = 0
SORT_BY_RECENT = 1
COMMENT_LIMIT = 100

YT_CFG_RE = r'ytcfg\.set\s*\(\s*({.+?})\s*\)\s*;'
YT_INITIAL_DATA_RE = r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]|ytInitialData)\s*=\s*({.+?})\s*;\s*(?:var\s+meta|</script|\n)'


def regex_search(text, pattern, group=1, default=None):
    match = re.search(pattern, text)
    return match.group(group) if match else default


def ajax_request(session, endpoint, ytcfg, retries=5, sleep=20):
    url = 'https://www.youtube.com' + endpoint['commandMetadata']['webCommandMetadata']['apiUrl']

    data = {'context': ytcfg['INNERTUBE_CONTEXT'],
            'continuation': endpoint['continuationCommand']['token']}

    for _ in range(retries):
        response = session.post(url, params={'key': ytcfg['INNERTUBE_API_KEY']}, json=data)
        if response.status_code == 200:
            return response.json()
        if response.status_code in [403, 413]:
            return {}
        else:
            time.sleep(sleep)


def download_comments(YOUTUBE_VIDEO_URL, sort_by=SORT_BY_RECENT, language=None, sleep=0.1):
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    response = session.get(YOUTUBE_VIDEO_URL)

    if 'uxe=' in response.request.url:
        session.cookies.set('CONSENT', 'YES+cb', domain='.youtube.com')
        response = session.get(YOUTUBE_VIDEO_URL)

    html = response.text
    ytcfg = json.loads(regex_search(html, YT_CFG_RE, default=''))
    if not ytcfg:
        return  
    if language:
        ytcfg['INNERTUBE_CONTEXT']['client']['hl'] = language

    data = json.loads(regex_search(html, YT_INITIAL_DATA_RE, default=''))

    section = next(search_dict(data, 'itemSectionRenderer'), None)
    renderer = next(search_dict(section, 'continuationItemRenderer'), None) if section else None
    if not renderer:
        return

    needs_sorting = sort_by != SORT_BY_POPULAR
    continuations = [renderer['continuationEndpoint']]
    while continuations:
        continuation = continuations.pop()
        response = ajax_request(session, continuation, ytcfg)

        if not response:
            break
        if list(search_dict(response, 'externalErrorMessage')):
            raise RuntimeError('Error returned from server: ' + next(search_dict(response, 'externalErrorMessage')))

        actions = list(search_dict(response, 'reloadContinuationItemsCommand')) + \
                  list(search_dict(response, 'appendContinuationItemsAction'))
        for action in actions:
            for item in action.get('continuationItems', []):
                if action['targetId'] == 'comments-section':
                    continuations[:0] = [ep for ep in search_dict(item, 'continuationEndpoint')]
                if action['targetId'].startswith('comment-replies-item') and 'continuationItemRenderer' in item:
                    continuations.append(next(search_dict(item, 'buttonRenderer'))['command'])

        for comment in reversed(list(search_dict(response, 'commentRenderer'))):
            yield {'cid': comment['commentId'],
                   'text': ''.join([c['text'] for c in comment['contentText'].get('runs', [])]),
                   'time': comment['publishedTimeText']['runs'][0]['text'],
                   'author': comment.get('authorText', {}).get('simpleText', ''),
                   'channel': comment['authorEndpoint']['browseEndpoint'].get('browseId', ''),
                   'votes': comment.get('voteCount', {}).get('simpleText', '0'),
                   'photo': comment['authorThumbnail']['thumbnails'][-1]['url'],
                   'heart': next(search_dict(comment, 'isHearted'), False)}

        time.sleep(sleep)


def search_dict(partial, search_key):
    stack = [partial]
    while stack:
        current_item = stack.pop()
        if isinstance(current_item, dict):
            for key, value in current_item.items():
                if key == search_key:
                    yield value
                else:
                    stack.append(value)
        elif isinstance(current_item, list):
            for value in current_item:
                stack.append(value)


def main(url):
    df_comment = pd.DataFrame()
    try:
        youtube_url = url
        limit = COMMENT_LIMIT

        print('Downloading Youtube comments for video:', youtube_url)

        count = 0

        start_time = time.time()

        for comment in download_comments(youtube_url):

            df_comment = df_comment.append(comment, ignore_index=True)

            comment_json = json.dumps(comment, ensure_ascii=False)
            print(comment_json)

            count += 1

            if limit and count >= limit:
                break

        print("DataFrame Shape: ", df_comment.shape, "\nComment DataFrame: ", df_comment)

        if not os.path.isfile(FILE_NAME):
            df_comment.to_csv(FILE_NAME, encoding='utf-8', index=False)
        else:  
            df_comment.to_csv(FILE_NAME, mode='a', encoding='utf-8', index=False, header=False)

        print('\n[{:.2f} seconds] Done!'.format(time.time() - start_time))

    except Exception as e:
        print('Error:', str(e))
        sys.exit(1)

ytb_video_list = urls

for video_link in ytb_video_list:
    main(video_link)
    


data = pd.read_csv('text_emotion.csv')

data = data.drop('author', axis=1)

# Dropping rows with other emotion labels
data = data.drop(data[data.sentiment == 'anger'].index)
data = data.drop(data[data.sentiment == 'boredom'].index)
data = data.drop(data[data.sentiment == 'enthusiasm'].index)
data = data.drop(data[data.sentiment == 'empty'].index)
data = data.drop(data[data.sentiment == 'fun'].index)
data = data.drop(data[data.sentiment == 'relief'].index)
data = data.drop(data[data.sentiment == 'surprise'].index)
data = data.drop(data[data.sentiment == 'love'].index)
data = data.drop(data[data.sentiment == 'hate'].index)
data = data.drop(data[data.sentiment == 'neutral'].index)
data = data.drop(data[data.sentiment == 'worry'].index)

data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

data['content'] = data['content'].str.replace('[^\w\s]',' ')

stop = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)

X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)

count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_val_tfidf)
print('naive bayes tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# naive bayes tfidf accuracy 0.5289017341040463

# Model 2: Linear SVM
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_tfidf, y_train)
y_pred = lsvm.predict(X_val_tfidf)
print('svm using tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# svm tfidf accuracy 0.5404624277456648

# Model 3: logistic regression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_tfidf, y_train)
y_pred = logreg.predict(X_val_tfidf)
print('log reg tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# log reg tfidf accuracy 0.5443159922928709

# Model 4: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_tfidf, y_train)
y_pred = rf.predict(X_val_tfidf)
print('random forest tfidf accuracy %s' % accuracy_score(y_pred, y_val))
# random forest tfidf accuracy 0.5385356454720617

## Building models using count vectors feature
# Model 1: Multinomial Naive Bayes Classifier
nb = MultinomialNB()
nb.fit(X_train_count, y_train)
y_pred = nb.predict(X_val_count)
print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# naive bayes count vectors accuracy 0.7764932562620424

# Model 2: Linear SVM
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# lsvm using count vectors accuracy 0.7928709055876686

# Model 3: Logistic Regression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_count, y_train)
y_pred = logreg.predict(X_val_count)
print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# log reg count vectors accuracy 0.7851637764932563

# Model 4: Random Forest Classifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_count, y_train)
y_pred = rf.predict(X_val_count)
print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_val))
# random forest with count vectors accuracy 0.7524084778420038

#Below are 8 random statements. The first 4 depict happiness. The last 4 depict sadness

df=pd.read_csv('ytb_comments.csv',usecols = ['text'])
lst = df["text"].to_numpy()
arr = []
for sentence in lst:
	arr.append(sentence)

comments = pd.DataFrame(arr)

# Doing some preprocessing on these comments as done before
comments[0] = comments[0].str.replace('[^\w\s]',' ')

comments[0] = comments[0].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
from textblob import Word
comments[0] = comments[0].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Extracting Count Vectors feature from our comments
comment_count = count_vect.transform(comments[0])

#Predicting the emotion of the tweet using our already trained linear SVM
comment_pred = lsvm.predict(comment_count)

positive=0
negative=0

for ele in comment_pred:
    if(ele==0):
        positive+=1
    else:
        negative+=1

print("positive comment= ",positive, "\nnegative comment= ",negative)
