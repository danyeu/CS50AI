import nltk
import sys
import os, string, math
#nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

PUNCTUATION = string.punctuation
STOPWORDS = nltk.corpus.stopwords.words("english")



def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = os.listdir(directory)
    file_names = [f for f in files if f.endswith('.txt')]

    txt_dict = {}
    for file_name in file_names:
        path = os.path.join(directory, file_name)
        with open(path, 'r') as txt_file:
            txt_dict[file_name] = txt_file.read()

    return txt_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []
    for word in nltk.word_tokenize(document):
        word = word.lower()
        if word in PUNCTUATION or word in STOPWORDS:
            continue
        words.append(word)

    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    corpus_dict = {} #key=word, value=#docs in which word appears
    for document in documents:
        doc_set = set(documents[document])
        for word in doc_set:
            if word in corpus_dict:
                corpus_dict[word] += 1
            else:
                corpus_dict[word] = 1

    idfs = {}
    for word in corpus_dict:
        idfs[word] = math.log(len(documents) / corpus_dict[word])

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    query_count = {doc: {word: 0 for word in query} for doc in files}
    for doc in files:
        for word in files[doc]:
            if word in query:
                query_count[doc][word] += 1

    tfidf = {doc: 0 for doc in files}
    for doc in files:
        for q in query:
            if q not in idfs:
                continue
            tfidf[doc] += query_count[doc][q] * idfs[q]

    sorted_docs = sorted(tfidf.items(), key=lambda x: -x[1])
    return [x[0] for x in sorted_docs[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    mwms = []
    qtds = []
    for sentence in sentences:
        matching_word_measure = 0
        query_count = 0
        queries_found = set()
        for word in sentences[sentence]:
            if word in query:
                query_count += 1
                queries_found.add(word)
        for word in queries_found:
            matching_word_measure += idfs[word]
        mwms.append(matching_word_measure)
        qtds.append(query_count / len(sentences[sentence]))

    sorted_sentences = sorted(zip(sentences.keys(), mwms, qtds), key=lambda x: (-x[1], -x[2]))
    return [x[0] for x in sorted_sentences[:n]]


if __name__ == "__main__":
    main()
