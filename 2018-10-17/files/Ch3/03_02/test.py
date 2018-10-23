wordList = sorted(list(set([word.strip().lower() for word in open("words", "r")])))

dict = {}

for word in wordList:
   if dict.get(len(word)) == None:
      dict[len(word)] = []
      dict[len(word)].append(word)
   else:
      dict.get(len(word)).append(word)

for key, value in dict.items():
   print(key)


