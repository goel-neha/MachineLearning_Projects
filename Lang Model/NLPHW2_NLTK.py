from __future__ import division
import glob
import nltk
import os
import collections
import random
from lxml import etree
from lxml import html
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import RegexpTokenizer
from nltk import bigrams, trigrams
from nltk.util import ngrams
from collections import Counter, defaultdict

def get_xml_files(path):
    xml_list = []
    for filename in os.listdir(path):
        if filename.endswith(".xml"):
            xml_list.append(filename)
    xml_list = xml_list[:200]
    return xml_list

def get_xml_data(xml_list):
    list2 = []
    stop_words = set(stopwords.words('english'))
    exclude = str.maketrans('', '', string.punctuation)
#    dataframe = pd.DataFrame()

    for xmlfile in xml_list:
        parser = etree.XMLParser(recover = True)
        tree = html.parse(loc + xmlfile)
        contend = etree.tostring(tree, pretty_print=True, encoding = 'unicode', method='text')
        contend = contend.translate(exclude)
        # Tokenizing
        sentences = nltk.sent_tokenize(contend)
        words = nltk.word_tokenize(contend)
        cleanTxt = [w for w in words if not w in stop_words]
#        print(len(cleanTxt))
        list2.append(cleanTxt)
    
    return (list2)  

def n_grams(list_tokens):
    list_bigram = []
    list_trigram = []
    for all_tokens in list_tokens:
        bigrams = ngrams(all_tokens,2)
        trigrams = ngrams(all_tokens,3)
#        list_bigram = list(bigrams)
        list_bigram.append(Counter(bigrams))
        list_trigram.append(Counter(trigrams))
#    print(list_bigram)
#    print(list_trigram)

def lang_model(list_tokens):
    model = defaultdict(lambda: defaultdict(lambda: 0.01))
 #Used trigrams to compute the lang model
    for each_list in list_tokens:
        for w1, w2, w3 in trigrams(each_list, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1
 
 #Prints the count of the evaluation of the lang model for different sentences
    print ("the following are count of different sentences")
    print (model["I", "dont"]["think"]) 
    print (model["Jessica", "going"]["kill"]) 
    print (model[None, None]["I"])
    print (model["Buenas","tardes"]["Hola"])
 
# transforming the above counts to probabilities
    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
#        print(total_count)
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
#prints the probabilities for the sentences in the lang model 
    print ("Probability of I Dont think - ", model["I", "dont"]["think"]) 
    print ("Probability of Jessica going kill - ", model["Jessica", "going"]["kill"]) 
    print ("Prob. of I - ", model[None, None]["I"]) 
    print ("Prob of Spanish Sentence - ",model["Buenas","tardes"]["Hola"])

# code to generate random senetences from the lang model    
    text = [None, None]
    prob = 1.0  # <- Init probability
 
    sentence_finished = False
 
    while not sentence_finished:
        r = random.random()
        accumulator = .0
 
        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
 
            if accumulator >= r:
                prob = ((prob * model[tuple(text[-2:])][word]) + 1.0 ) # <- Updated the probability with the conditional proba
                text.append(word)
                break
 
        if text[-2:] == [None, None]:
            sentence_finished = True
    print ("Probability of text=", prob)  # <- Print the probability of the text
    print ("GENERATED TEXT")
    print (' '.join([t for t in text if t]))
#    return(model)
 
#computes perplexity of the trigram model on a testset  
#def perplexity(testset, model):
#    text = [None, None]
    testset = "(CNN)After an apparent hourslong standoff between police and a gunman, three women were found dead Friday night at the Veterans Home of California in Yountville, authorities said.The suspect was also found dead, Chris Childs, assistant chief of the California Highway Patrol's Golden Gate Division, told reporters.The three victims were earlier described as employees of The Pathway Home, a counseling service for veterans who suffer post-traumatic stress disorder (PTSD), which is on the property.The suspect had been a client at facility until he left two weeks ago, according to State Senator Bill Dodd.It's unclear if the women were chosen at random or had a connection with the gunman.Hostage negotiators had spent hours trying to contact the gunman at the facility north of San Francisco but were not able to do so, according to California Highway Patrol spokesman Robert Nacke. After the victims were found, investigators were left to determine when during the standoff the deaths occurred.Officers are at the scene Friday at the Veterans Home of California in Yountville.Officers are at the scene Friday at the Veterans Home of California in Yountville. nd our deputy, Robertson said. There were many bullets fired.Law enforcement officers responded to a shots fired report around 10:20 a.m. local time from the facility that houses about 1,000 veterans, Robertson said. "
    
    testset = testset.split()
#    print ("perplexity = ", model.perplexity(testset))
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = (perplexity * (1/model[tuple(text[-2:])][word]))
    perplexity = pow(perplexity, 1/(N)) 
    print (perplexity)
#    return perplexity      
     

def main():
    location = "blogs/"
    filesList = get_xml_files(location)
    list_tokens = get_xml_data(filesList)
    n_grams(list_tokens)
    model = lang_model(list_tokens)
    
#    perplexity(testtext,model)
    

#   test(langModel)

if __name__ == "__main__":
    main()
