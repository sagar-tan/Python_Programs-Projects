import re
def to_uppercase(text):
    return text.upper()

def split_into_words(text):
    return text.split()

def remove_special_characters(text):
    return re.sub(r"[,'\s]+", ' ', text)

def remove_numbers_and_extra_spaces(text):
    text_without_numbers = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text_without_numbers).strip()

def find_root_word(word):
    suffixes = ["ing", "ed", "es", "ly", "er", "ment", "ness", "ful"]
    for suffix in suffixes:
        if word.endswith(suffix):
            word = word[:-len(suffix)]
    return word
text = "This is the NLP class, this is a sample text, these are the numbers 12331343423455 which is invisible cuz they're cleaned"

uppercase_text = to_uppercase(text)

words = split_into_words(uppercase_text)

cleaned_text = remove_special_characters(uppercase_text)

cleaned_text_without_numbers = remove_numbers_and_extra_spaces(cleaned_text)

root_word = find_root_word("Sprinting")

print("Uppercase Text: ", uppercase_text)
print("Words: ", words)
print("Cleaned Text without special characters: ", cleaned_text)
print("Text after removing numbers and extra spaces: ", cleaned_text_without_numbers)
print("Root word of 'learning': ", root_word)
