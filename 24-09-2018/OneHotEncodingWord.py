import numpy as np

samples = ["The cat sat on the mat.", "The dog ate my homework"]

tokenIndex = {}
for sample in samples:
    for word in sample.split():
        if word not in tokenIndex:
            tokenIndex[word] = len(tokenIndex) + 1

'''
for key, value in tokenIndex.items():
    print(key, ": ", value)
'''

maxLength = 10
results = np.zeros(shape=(len(samples),
                          maxLength,
                          max(tokenIndex.values()) + 1))
'''
i = 0
for sample in samples:
    j = 0
    for word in sample.split():
        index = tokenIndex.get(word)
        results[i, j, index] = 1
        j += 1
        print(i, j)
    i += 1

print(results.shape)
print(results)
'''

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:maxLength]:
        index = tokenIndex.get(word)
        results[i, j, index] = 1

print(results)