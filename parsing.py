import json
from structures import *



def extract_sentences(filename: str) -> List[NormSentences]:
    """
    Given a file with tokenized parallel sentences and alignments in json format,
    return a list of sentence pairs and alignments for each sentence.

    Args:
        filename: Name of the file containing json markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
    """

    all_sentences = {}

    with open(filename) as json_file:
        data_json = json.load(json_file)
        for sentence_json in data_json["sentences"]:
            sentence = sentence_json["text"]
            id = sentence_json["para_alignment"][0]["para_id"]
            lang = sentence_json["lang"]
            all_sentences[id] = all_sentences.get(id, []) + [(lang, sentence)]
    
    result = []
    for _, sentence_pair in all_sentences.items():
        if len(sentence_pair) != 2:
            continue
        if len(sentence_pair[0][1]) == 0 or len(sentence_pair[1][1]) == 0:
            continue
        if sentence_pair[0][0] != 0:
            sentence_pair = (sentence_pair[1], sentence_pair[0])
        result.append(NormSentences(sentence_pair[0][1], sentence_pair[1][1]))
    return result