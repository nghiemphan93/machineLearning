import collections

word = open("words", "r")

wordList = word.readlines()


wordClean = sorted(list(set([word.strip().lower() for word in open("words", "r")])))

wordByLength = collections.defaultdict(list)

print(type(wordByLength))
print(wordByLength)

for word in wordClean:
    wordByLength[len(word)].append(word)

print(wordByLength)
