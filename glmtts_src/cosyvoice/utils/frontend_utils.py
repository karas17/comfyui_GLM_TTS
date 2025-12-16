# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#               2025 Zhipu AI Inc (authors: CogAudio Group Members)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unicodedata
import emoji
import pronouncing

# Regular expression constants
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff]+')
PUNCTUATION_CHARS = r'。？！；：、.?!;:，,'
PUNCTUATION_PATTERN = re.compile(fr'([{PUNCTUATION_CHARS}])')


def is_phoneme(item):
    """
    Determine if an item is a phoneme.
    Phonemes typically contain letters/numbers, whereas punctuation and spaces do not.
    """
    if not isinstance(item, str) or not item:
        return False
    
    # If it contains only whitespace, punctuation, or is a hyphen, it's not a phoneme
    if item.strip() == '' or item in PUNCTUATION_CHARS or item == '-':
        return False
    else:
        return True

def contains_chinese(text):
    """Check if the text contains Chinese characters."""
    return bool(CHINESE_CHAR_PATTERN.search(text))


def replace_corner_mark(text):
    """Replace superscript numbers (corner marks) with Chinese characters."""
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    text = text.replace('^二', '平方')
    text = text.replace('^三', '立方')
    return text


def remove_bracket(text, lang='zh'):
    """Remove various types of brackets from the text."""
    brackets_to_remove = [
        ('(', ')'), ('（', '）'), ('【', '】'), ('「', '」'),
        ('`', '`'), ('《', '》'), ('『', '』'),
        ('{', '}'), ('[', ']')
    ]
    if lang != 'en':
        brackets_to_remove.append(('‘', '’'))
    for left, right in brackets_to_remove:
        text = text.replace(left, '').replace(right, '')
    
    return text


def spell_out_number(text, inflect_parser):
    """Convert numbers in the text to their English word representation."""
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


def split_into_units(s):
    """Split string into units (Chinese characters or English words)."""
    def is_chinese(char):
        try:
            name = unicodedata.name(char)
            return 'CJK' in name or 'IDEOGRAPH' in name
        except ValueError:
            return False

    result = []
    buffer = []
    allowed_symbols = {"'", '-'}  # Symbols allowed within English words

    for char in s:
        if is_chinese(char):
            if buffer:
                result.append(''.join(buffer))
                buffer = []
            result.append(char)
        elif char.isalpha() or (buffer and char in allowed_symbols):
            # If it's a letter or an allowed symbol (when buffer is not empty)
            buffer.append(char)
        else:
            if buffer:
                result.append(''.join(buffer))
                buffer = []
            result.append(char)
            
    # Process remaining buffer
    if buffer:
        result.append(''.join(buffer))

    # Spaces are not separate units; merge them into the previous unit
    result1 = []
    for i, u in enumerate(result):
        if u.isspace() and len(result1) > 0:
            result1[-1] += u  # Merge
        else:
            result1.append(u)
    return result1


def is_all_english(s):
    return len(s) > 0 and s.isalpha() and s.isascii()


def count_syllables_re(word):
    """Fallback syllable counter using regex."""
    word = word.lower()
    # Match vowel sequences not ending with 'e', or words containing only 'e'
    syllables = re.findall(r'(?!e$)[aeiouy]+', word, flags=re.I)
    if re.fullmatch(r'^[^aeiouy]*e$', word, flags=re.I):
        syllables.append('e')
    return max(len(syllables), 1)


def count_syllables(word):
    """Count syllables in a word using 'pronouncing' lib or fallback regex."""
    try:
        return pronouncing.syllable_count(pronouncing.phones_for_word(word)[0])
    except:
        return max(count_syllables_re(word), 1)
    

def count_char(units): 
    """
    Count total equivalent Chinese characters.
    English syllables are weighted (approx 0.7x of a Chinese character duration).
    """
    ratio_en_per_zh = 0.7
    res = 0
    for u in units:
        if is_all_english(u.strip()): # Special handling for English
            res += count_syllables(u.strip()) * ratio_en_per_zh
        else:
            res += 1
    return res


def split_into_min_sentence(text, min_sentence_len=5):
    """
    Split text into minimal sentences. A minimal sentence must:
    1. End with a punctuation mark.
    2. Have a length >= min_sentence_len.
    """
    res = []
    cur_units = []
    at_least_one_sentence = False 

    units = split_into_units(text)
    for i, u in enumerate(units):
        cur_units.append(u)
        if u.strip() in list(PUNCTUATION_CHARS):
            if count_char(cur_units) >= min_sentence_len:
                at_least_one_sentence = True
                res.append(cur_units)
                cur_units = []
    if cur_units:
        res.append(cur_units)
    return res, at_least_one_sentence


def split_hard(sent_units, max_text_len=40):
    """
    Hard split for very long sentences without punctuation.
    sent_units: list of list; sentences x units_of_a_sentence
    """
    result = []
    for sent in sent_units:
        assert '' not in map(str.strip, sent) # Ensure no pure space units
        while count_char(sent) > max_text_len: # Sentence too long, cut it
            if sent[max_text_len].strip() in list(PUNCTUATION_CHARS): 
                # If the cut point is a symbol, include the next char to avoid starting next line with punctuation
                result.append(sent[:max_text_len+1])
                sent = sent[max_text_len+1:]
            else:
                # Direct cut
                result.append(sent[:max_text_len])
                sent = sent[max_text_len:]
        if sent:
            result.append(sent)
    return result


def replace_space(text: str) -> str:
    """
    Preserve spaces between (English word/number) pairs.
    Remove other spaces.
    """
    alphanumeric_pattern = r'[a-zA-Z0-9]'
    punctuation_pattern = r'[.,!?;:]'
    
    # Compress multiple spaces
    text = re.sub(r' +', ' ', text)
    result = ""
    i = 0
    while i < len(text):
        current_char = text[i]
        if current_char != ' ':
            result += current_char
            i += 1
            continue
        
        # Handle space logic
        prev_char = text[i-1] if i > 0 else ''
        next_char = text[i+1] if i+1 < len(text) else ''
        
        # Case 1: Between English words/numbers
        if re.match(alphanumeric_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += ' '
        # Case 2: Between English punctuation and English word/number
        elif re.match(punctuation_pattern, prev_char) and re.match(alphanumeric_pattern, next_char):
            result += ' '
        
        i += 1
    return result


def multi_line_process(plain_text):
    """
    Process multi-line text:
    1. Remove empty lines.
    2. Ensure ends with punctuation.
    3. Remove line breaks (merge).
    """
    lines = []
    for line in plain_text.split('\n'):
        line = line.strip()
        if line == '': # skip empty lines
            continue

        # Ensure line ends with punctuation
        if line[-1] not in {'.', '!', '?', ';', ':', '：', '。', '！', '？', '；', '，'}:
            if contains_chinese(line):
                line = line.rstrip() + '。'
            else:
                line = line + '. '
        lines.append(line)

    if contains_chinese(''.join(lines)):
        return ''.join(lines)
    else:
        return ' '.join(lines)


def emoji_norm(text):
    """Remove all emoji characters from text."""
    return emoji.replace_emoji(text, replace='')


def markdown_norm(markdown_text):
    """Convert Markdown to plain text, preserving ordered list numbers."""
    # Convert '1. ' to '1。' to avoid period confusion
    markdown_text = re.sub(r'^(\d+)\. ', r'\1。', markdown_text)
    markdown_text = markdown_text.replace('\\n', '\n') 
    return markdown_text


def number_to_chinese(number):
    """Convert number to Chinese pronunciation."""
    units = ["", "十", "百", "千", "万", "十万", "百万", "千万", "亿", "十亿", "百亿", "千亿"]
    nums = "零一二三四五六七八九"
    if isinstance(number, (float, str)):
        str_num = str(float(number))
        if '.' in str_num:
            int_part, decimal_part = str_num.split('.')
            if decimal_part.strip('0') == '':
                return number_to_chinese(int(int_part))
            chinese_int = number_to_chinese(int(int_part)) if int(int_part) != 0 else '零'
            chinese_decimal = ''.join(nums[int(d)] for d in decimal_part)
            return f"{chinese_int}点{chinese_decimal}"
    
    if number == 0:
        return "零"
    if number < 0:
        return "负" + number_to_chinese(abs(number))
        
    result = []
    unit_position = 0
    zero_flag = False
    last_unit = ""
    while number > 0:
        num = number % 10
        current_unit = units[unit_position] if unit_position < len(units) else ""
        
        if num == 0:
            if not zero_flag and result and last_unit not in ["万", "亿"]:
                result.append(nums[num])
                zero_flag = True
        else:
            if current_unit:
                result.append(current_unit)
            result.append(nums[num])
            zero_flag = False
            last_unit = current_unit
        unit_position += 1
        number //= 10
        
    result.reverse()
    
    # Cleanup special cases
    text = ''.join(result)
    text = re.sub(r'^一十', '十', text)
    text = re.sub(r'零+', '零', text)
    text = re.sub(r'零$', '', text)
    
    return text


def tn_scientific_notation(case):
    """Normalize scientific notation to Chinese description."""
    pattern = re.compile(r'(.*?)(-?\d+(?:\.\d+)?)[*x×](\d+(?:\.\d+)?)\^(-?\d+(?:\.\d+)?)(.*?)$')
    match = pattern.match(case)
    if match:
        prefix, base, multiplier, exponent, suffix = match.groups()
        description = f"{number_to_chinese(base)}乘{number_to_chinese(multiplier)}的{number_to_chinese(exponent)}次方"
        result = f"{prefix}{description}{suffix}"
    else:
        result = case
    return result


def replace_asterisk_with_multiply(text, lang='zh'):
    """Replace asterisk (*) with multiplication word based on context."""
    if lang == 'en':
        replace_name = 'multiply'
        replace_number = '0-9'
    else:
        replace_name = '乘'
        replace_number = '一二三四五六七八九十百千万亿'

    # Rule 0: * between numbers or letters
    rule0 = rf'(?<=[{replace_number}a-zA-Z])\s*\*\s*(?=[{replace_number}a-zA-Z])'

    # Rule 1: ...number) * number...
    rule1 = rf'(?<=[{replace_number}a-zA-Z]\))\s*\*\s*(?=[{replace_number}a-zA-Z])'
    
    # Rule 2: ...number * (number...
    rule2 = rf'(?<=[{replace_number}a-zA-Z])\s*\*\s*(?=\([{replace_number}a-zA-Z])'
    
    # Rule 3: ...number) * (number...
    rule3 = rf'(?<=[{replace_number}a-zA-Z]\))\s*\*\s*(?=\([{replace_number}a-zA-Z])'
    
    pattern = f'{rule0}|{rule1}|{rule2}|{rule3}'
    return re.sub(pattern, replace_name, text).replace('*', '')


def special_replace(text):
    """
    Perform specific replacements requiring context or custom rules.
    """
    # Remove backslashes
    text = text.replace('\\', '')

    # '额' -> '呃' when surrounded by boundaries/punctuation
    text = f" {text} " # Add padding spaces for regex matching
    pattern = r'(?<=\W)额(?=\W)'
    text = re.sub(pattern, '呃', text)
    
    # Replace tildes with period
    text = re.sub(r'[~～]+', '。', text)
    
    # Handle asterisk replacement
    text = replace_asterisk_with_multiply(text, 'zh')

    # Remove padding spaces
    text = text.strip()
    return text


def ensure_proper_ending(text):
    """Ensure the text ends with appropriate punctuation."""
    if text and text[-1] in '？?': 
        return text
        
    if text and text[-1] in PUNCTUATION_CHARS:
        # If ending with punctuation, ensure it matches the language
        if contains_chinese(text):
            text = text[:-1] + '。'
        else:
            text = text[:-1] + '.'
    else:
        # If no punctuation, add it
        if contains_chinese(text):
            text += '。'
        else:
            text += '.'
    
    return text


def ensure_proper_en_ending(text):
    """Ensure English text ends with a period."""
    if text:
        text = text.rstrip('.,?!:- ')
        while text and not text[-1].isalnum():
            text = text[:-1]
        text = text + '.'
    return text


def normalize_punctuation(text, punctuation_chars):
    """Normalize punctuation: handle consecutive marks, map specific symbols."""
    text = replace_space(text)
    # Deduplicate consecutive identical punctuation
    text = re.sub(fr'([{punctuation_chars}])\1+', r'\1', text)
    # Handle different consecutive punctuation (keep first)
    text = re.sub(fr'([{punctuation_chars}])(?=[{punctuation_chars}])', '', text)

    text = text.replace("#", "")
    text = text.replace("！", "。")
    text = text.replace("!", ".")
    return text


if __name__ == "__main__":
    text = "This passage also introduced Buddhism, Islam, as well as products such as grapes, walnuts, pomegranates, cucumbers, glass, and perfume to us."
    # Test min sentence split
    texts, _ = split_into_min_sentence(text, min_sentence_len=30)
    print(texts)
    # Test hard split
    texts = split_hard(texts, max_text_len=60)
    print(texts)