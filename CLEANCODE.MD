# Clean Code Quick Guide

This guide outlines essential clean code practices to improve readability, maintainability, and collaboration in your NLP/LLM project.



## Meaningful Naming

- Use clear, descriptive names:

```python
# Good
tokenized_sentences = tokenize_sentences(raw_text)

# Bad
ts = tk_s(raw)
```



## Small, Focused Functions

- Functions should perform one clearly defined task.
- Aim for no more than 10-15 lines per function.

```python
def remove_stopwords(tokens, stop_words):
    return [token for token in tokens if token not in stop_words]
```



## Clear, Informative Comments

- Comments should explain the "why", not the "what".

```python
# Good
# Normalize text to lower case for consistency
text = text.lower()

# Bad
# Convert text
text = text.lower()
```



## Error and Edge-Case Handling

- Clearly handle exceptions and potential issues:

```python
try:
    data = load_data(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
```


## Function and Class Documentation

- Document each function and class clearly:

### Function Example
```python
def tokenize_text(text):
    """
    Tokenizes input text into individual tokens.

    :param text: Raw input string
    :type text: str
    :return: List of tokens
    :rtype: list
    """
    tokens = text.split()
    return tokens
```

### Class Example
```python
class TextProcessor:
    """
    Class for processing and cleaning text data.
    """

    def __init__(self, stop_words):
        """
        Initializes the TextProcessor class.

        :param stop_words: List of stop words
        :type stop_words: list
        """
        self.stop_words = stop_words

    def remove_stopwords(self, tokens):
        """
        Removes stop words from token list.

        :param tokens: List of tokens
        :type tokens: list
        :return: Cleaned list of tokens
        :rtype: list
        """
        return [token for token in tokens if token not in self.stop_words]
```


## Python Naming Conventions

**Variables and functions**: lowercase with underscores (`snake_case`)
```python
def load_data(file_path):
    """...
    """
    data_frame = pd.read_csv(file_path)
    return data_frame
```

**Classes**: Capitalized words without underscores (`CamelCase`)
```python
class DataProcessor:
    """...
    """
    pass
```

**Constants**: Uppercase with underscores
```python
MAX_SEQ_LENGTH = 512
```




