import pandas as pd
import numpy as np
from scipy import stats
import re
import string
import enum

def build_dataframe(filename):
    words = pd.read_csv(filename, header=None, names=['word'])['word'].str.lower()
    df = pd.DataFrame(index=words)
    return df

log = np.emath.logn  # log(base, x), more flexible than np.log

def H(df):
    # H(X), where X is a pandas DataFrame
    total = 2309
    remaining = df.index.size # The probability for all remaining words is always equal
    # the probability for eliminated words is 0% so we can omit them
    return -remaining * (1 / remaining) * log(total, 1 / remaining)  


def get_information_content(df, letter):
    total = df.index.size
    included = df.filter(regex=letter, axis=0).index.size # number of words containing a letter
    excluded = total - included # everything else
    return included * excluded / (total / 2) ** 2 # (half the total) squared in the denominator normalizes the values to a range of 0..1


def alphabet_ic(df):
    alphabet = string.ascii_lowercase
    ic = {c: get_information_content(df, c) for c in alphabet}
    return {k:v for k,v in sorted(ic.items(), key=lambda c: c[1], reverse=True)}


def score_word(df, word):
    # Repeated letters can yield a little extra information, but not so much that it was worth coding up, 
    # so I'll penalize repeated letters here by casting the letters into a set, so only unique letters are scored.
    return sum([get_information_content(df, c) for c in set(word)]) / 5 # average information content in of each letter in a word


def score_remaining_words(df, show_entropy = True):
    df2 = df.copy()
    df2['score'] = [score_word(df, word) for word in df.index]
    if show_entropy:
        print("Remaining entropy:", H(df2))
    return df2.sort_values(by='score', ascending=False, axis=0)


def update(df, word, result):
    assert(re.match(r'[a-z]{5}', word)) # word must be 5 letters long
    assert(re.match(r'[.xX]{5}', result)) # result must only contain '.', 'x', 'X' for a miss, partial match, and perfect match, respectively
    
    letters = {c: len([i for i in word if c == i]) for c in word}
    pattern = ['[^]'] * 5 # empty regex char negation, will be populated or replaced as we work through the guess
    
    for r,w,i in zip(result, word, range(0,5)):
        # Perfect match
        if r == 'X':
            pattern[i] = w

        elif r == 'x':
            df = df.filter(regex=w, axis=0) # Ensure all remaining words contain this letter
            pattern[i] = pattern[i][:-1] + w + pattern[i][-1]
            
        # Miss
        elif r == '.':
            for j in range(0,5):
                if pattern[j].startswith('[^') and w not in pattern[j]:
                    pattern[j] = pattern[j][:-1] + w + pattern[j][-1]

    # Adjust for partial matches which would otherwise get eliminated if duplicate letter appears later which is a miss
    for r,w,i in zip(result, word, range(0,5)):
        if r == 'x':
            for j in range(0,5):
                if i != j and word[j] != w:
                    pattern[j] = pattern[j].replace(w, '')
    # print("".join(pattern))
    return df.filter(regex="".join(pattern), axis=0)


class Strategy(enum.Enum):
    RANDOM = 'random'
    ENTROPIC = 'entropic'

def check_word(target, guess):
    # This creates the feedback from playing a word, using '.', 'x', and 'X' for gray, yellow, and green in the UI
    
    assert(re.match(r'[a-z]{5}', target))
    assert(re.match(r'[a-z]{5}', guess))
    result = "....."
    remaining_letters = {c: len([i for i in target if c == i]) for c in target}

    for t,g,i in zip(target, guess, range(0,5)):
        # Perfect match
        if g == t:
            result = result[:i] + "X" + result[i + 1:]
            remaining_letters[g] -= 1

    # Has to be handled separately to account for edge cases when an exact match is preceded by a miss of the same letter
    for t,g,i in zip(target, guess, range(0,5)):
        # Partial matches
        if g in target and remaining_letters[g] >= 1 and result[i] != 'X':
            result = result[:i] + "x" + result[i + 1:]
            remaining_letters[g] -= 1

    return result

def simulate_wordle(df, strategy, target=None, verbose=False):
    if not target:
        target = df.sample(n=1).index[0]
    if verbose:
        print("Target word:", target)
    guesses = []
    result = ""
    
    while True:
        """
        The first guess of an entropic playthrough is always always 'irate'. The first pass also takes the longest to score, as it
        has to score 2309 words. Scoring is only useful for finding a word to play, and if we know what word we're playing in advance,
        we can get a performance boost by skipping the first round of scoring and just manually passing 'irate'.
        """
        if strategy == Strategy.ENTROPIC and len(guesses) == 0:
            guess = 'irate'
            guesses.append('irate')
        elif strategy == Strategy.RANDOM:
                guess = df.sample(n=1).index[0]
                guesses.append(guess)
        else:
            guess = score_remaining_words(df, show_entropy=False).index[0]
            guesses.append(guess)
        result = check_word(target, guess)
        df = update(df, guess, result)
        if verbose:
            print(guess, result)
        
        if result == "XXXXX":
            break
    return { # Tabular data is best data
        "target": target,
        "guesses": guesses,
        "strategy": strategy,
        "guess_count": len(guesses)
    }


def print_summary_statistics(df, type):
    print("Summary statistics for:", type)
    print(df['guess_count'].describe())
    print("\n")
