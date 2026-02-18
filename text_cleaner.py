import re
import string

class TextCleaner:
    """
    Utility for cleaning and normalizing text.
    """
    
    def __init__(self):
        self.punctuation = string.punctuation
    
    def clean_text(self, text: str) -> str:
        """
        Clean text: lowercase, strip source whitespace, remove special characters.
        """
        text = text.lower()
        text = text.strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def tokenize(self, text: str) -> list:
        """Split text into individual words."""
        cleaned = self.clean_text(text)
        return cleaned.split()
    
    def get_word_count(self, text: str) -> int:
        """Count words in text."""
        return len(self.tokenize(text))

if __name__ == "__main__":
    cleaner = TextCleaner()
    print(cleaner.clean_text("  Hello, World!!!  "))