from structures import *
import numpy as np
import collections


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    """
    source_cnt = collections.Counter()
    target_cnt = collections.Counter()
    
    for sentence_pair in sentence_pairs:
        source_cnt.update(sentence_pair.source)
        target_cnt.update(sentence_pair.target)
    
    if freq_cutoff is None:
        freq_cutoff = max(len(source_cnt), len(target_cnt))

    source_list = [k[0] for k in source_cnt.most_common(freq_cutoff)]
    target_list = [k[0] for k in target_cnt.most_common(freq_cutoff)]
    
    return ({token: index for index, token in enumerate(source_list)},
            {token: index for index, token in enumerate(target_list)})
    


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_sentence_pairs = []
    
    for sentence_pair in sentence_pairs:
        source_tokens = np.array(list(map(lambda x: source_dict.get(x, -1), sentence_pair.source)))
        if np.any(source_tokens[source_tokens < 0]):
            continue
        
        target_tokens = np.array(list(map(lambda x: target_dict.get(x, -1), sentence_pair.target)))
        if np.any(target_tokens[target_tokens < 0]):
            continue
        
        tokenized_sentence_pairs.append(TokenizedSentencePair(source_tokens, target_tokens))
        
    return tokenized_sentence_pairs
