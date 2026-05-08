import spacy

# load spacy
nlp = spacy.load("en_core_web_sm")

#swap based on gender lexicons
direct_swaps = {
    "she": "he", 
    "he": "she",
    "herself": "himself", 
    "himself": "herself",

    "woman": "man", 
    "man": "woman",
    "women": "men", 
    "men": "women",
    "female": "male", 
    "male": "female",
    "females": "males", 
    "males": "females",
    "lady": "gentleman", 
    "gentleman": "lady",
    "ladies": "gentlemen", 
    "gentlemen": "ladies",

    "girl": "boy", 
    "boy": "girl",
    "girls": "boys", 
    "boys": "girls",

    "gal": "guy", 
    "guy": "gal",
    "gals": "guys", 
    "guys": "gals",

    "mother": "father", 
    "father": "mother",
    "mom": "dad", 
    "dad": "mom",
    "daughter": "son", 
    "son": "daughter",
    "sister": "brother", 
    "brother": "sister",
    "wife": "husband", 
    "husband": "wife",
    "aunt": "uncle", 
    "uncle": "aunt",
    "niece": "nephew", 
    "nephew": "niece",
    "grandmother": "grandfather", 
    "grandfather": "grandmother"
}


def swap_gender(text):
    #spacy split to tokens
    doc = nlp(text)
    new_words = []
    
    for token in doc:
        word_lower = token.lower_
        
        # edge cases
        if word_lower == "her":
            # PRP$ = Possessive Pronoun
            if token.tag_ == "PRP$": 
                new_word = "his"  # "her book" -> "his book"
            else: 
                new_word = "him"  # "I saw her" -> "I saw him"
                
        elif word_lower == "his":
            if token.tag_ == "PRP$":
                new_word = "her"  # "his car" -> "her car"
            else:
                new_word = "hers" # "the car is his" -> "the car is hers"
                
        # basic lexicon
        elif word_lower in direct_swaps:
            new_word = direct_swaps[word_lower]
            
        else:
            new_word = token.text
            
        if token.is_title:
            new_word = new_word.capitalize()
            
        new_words.append(new_word)
        
    # back to text
    flipped_text = "".join([new_words[i] + doc[i].whitespace_ for i in range(len(doc))])
    
    return flipped_text

def main():

  print("Starting counterfactual generation...")
  test_sentence = "She told her brother that the car is his, but she still loves her car."

  print("Original:", test_sentence)
  print("Flipped: ", swap_gender(test_sentence))
  print("Done.")
