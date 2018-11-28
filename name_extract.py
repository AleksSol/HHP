from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

#1665
class NameExtract:
    VOLDEMORT = ["You-Know-Who", "He-Who-Must-Not-Be-Named", "The Dark Lord", "Dark Lord"]

    def __init__(self):
        self.st = StanfordNERTagger('./stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                                    './stanford-ner/stanford-ner.jar',
                                    encoding='utf-8')

    def voldemort_replace(self, new_text):
        for name in self.VOLDEMORT:
            new_text = new_text.replace(name, 'Voldemort')
        return new_text

    def extract_names(self, new_text: str):
        new_text = self.voldemort_replace(new_text)
        tokenized_text = word_tokenize(new_text)
        classified_text = self.st.tag(tokenized_text)
        res = [0]
        prev_person = False
        for entity in classified_text:
            if entity[1] == 'PERSON':
                if prev_person:
                    res[-1] = res[-1] + '_' + entity[0]
                else:
                    res.append(entity[0])
                prev_person = True
            else:
                if prev_person:
                    res.append(1)
                else:
                    res[-1] += 1
                prev_person = False
        return res[1::2], res[0::2]


if __name__ == '__main__':
    with open('./data/books/1.txt', encoding='utf-8') as f:
        text = f.read()
    text = text
    extractor = NameExtract()
    names, pauses = extractor.extract_names(text)
    print(set(names))
    print(len([x for x in pauses if x <= 20]), len([i for i in range(len(pauses) - 1) if pauses[i] + pauses[i+1] <= 20]))
