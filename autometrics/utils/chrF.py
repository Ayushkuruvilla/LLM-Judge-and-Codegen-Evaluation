import sys
import math
import time
import string
from collections import defaultdict

def separate_characters(line):
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1]
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)

    return tokenized

def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order + 1):
            if i + j <= nWords:
                ngram = tuple(wordList[i:i + j])
                counts[j - 1][ngram] += 1

    return counts

def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)

    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])

    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount

def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)

    factor = beta ** 2

    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order] / hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order] / reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor * ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1 + factor) * ngramPrec[order] * ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16

    return ngramF, ngramRec, ngramPrec

def computeChrF(references, hypotheses, nworder, ncorder, beta):
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0
    bestMatchingCount = defaultdict(float)
    bestRefCount = defaultdict(float)
    bestHypCount = defaultdict(float)
    bestChrMatchingCount = defaultdict(float)
    bestChrRefCount = defaultdict(float)
    bestChrHypCount = defaultdict(float)


    nsent = 0
    for hline, rline in zip(hypotheses, references):
        nsent += 1

        # preparation for multiple references
        maxF = 0.0

        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts,
                                                                                                 hypChrNgramCounts)

            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount,
                                                         beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount,
                                                                  totalChrHypNgramCount, beta)

            sentRec = (sum(chrNgramRec.values()) + sum(ngramRec.values())) / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF = (sum(chrNgramF.values()) + sum(ngramF.values())) / norder

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount
        # all the references are done

        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    if nsent == 0:
        nsent = 1  # prevent division by 0

    # all sentences are done

    # total precision, recall and F (arithmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(totalChrMatchingCount, totalChrRefCount,
                                                                         totalChrHypCount, beta)

    totalF = (sum(totalChrNgramF.values()) + sum(totalNgramF.values())) / norder
    averageTotalF = averageTotalF / nsent
    totalRec = (sum(totalChrNgramRec.values()) + sum(totalNgramRec.values())) / norder
    totalPrec = (sum(totalChrNgramPrec.values()) + sum(totalNgramPrec.values())) / norder

    return totalF, averageTotalF, totalPrec, totalRec

def chrf_score(ncorder=6, nworder=2, beta=2.0):
    with open('reference', 'r') as rtxt, open('subject', 'r') as stxt:
        reference = rtxt.readlines()
        subject = stxt.readlines()

    totalF, averageTotalF, totalPrec, totalRec = computeChrF(reference, subject, nworder, ncorder, beta)

    print(f"chrF F1-score: {totalF:.2f}")
    print(f"chrF F1-score (macro-averaged): {totalF:.2f}")
    print(f"chrF Precision: {totalF:.2f}")
    print(f"chrF Recall: {totalF:.2f}")

    return totalF


