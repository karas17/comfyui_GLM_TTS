# Copyright (c) 2025 Zhipu AI Inc (authors: CogAudio Group Members)
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
import logging
from typing import List, Union
import jieba.posseg as psg
import jieba
from pypinyin import Style, pinyin

# --- Initialization & Setup ---
# Setting logging level for jieba to suppress non-essential messages
jieba.setLogLevel(logging.INFO) 
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s') # Set up basic logging for warnings

# --- Minimal Separator Class ---
class Separator:
    """Minimal structure for phoneme/syllable separators."""
    def __init__(self, word: str, syllable: str, phone: str) -> None:
        self.word = word 
        self.syllable = syllable 
        self.phone = phone 

def is_chinese(char):
    """
    Checks if a character is a Chinese Hanzi (Simplified/Traditional).
    """
    cp = ord(char)
    return (
        0x4E00 <= cp <= 0x9FFF or
        0x3400 <= cp <= 0x4DBF or
        0x20000 <= cp <= 0x2A6DF or
        0x2A700 <= cp <= 0x2B73F or
        0x2B740 <= cp <= 0x2B81F or
        0x2B820 <= cp <= 0x2CEAF or
        0x2CEB0 <= cp <= 0x2EBEF or
        0x30000 <= cp <= 0x3134F or
        0x31350 <= cp <= 0x323AF or
        0xF900  <= cp <= 0xFAFF or
        0x2F800 <= cp <= 0x2FA1F or
        0x2E80  <= cp <= 0x2EFF
    )

def split_sentence(sentence):
    """
    Splits the sentence into Chinese and Non-Chinese parts.
    Simplified to handle the G2P input segmentation.
    """
    parts = []
    current_part = ''
    prev_is_chinese = None

    for char in sentence:
        current_is_chinese = is_chinese(char)
        
        if prev_is_chinese is None:
            # First character initialization
            current_part = char
            prev_is_chinese = current_is_chinese
            continue
        
        if current_is_chinese == prev_is_chinese:
            # Same language/type continues
            current_part += char
        else:
            # Boundary found
            parts.append((current_part, 'chinese' if prev_is_chinese else 'other'))
            current_part = char
            prev_is_chinese = current_is_chinese

    if current_part:
        parts.append((current_part, 'chinese' if prev_is_chinese else 'other'))
            
    return parts


def remove_endsyllable(lst, syllable):
    """
    Removes trailing syllable separators.
    """
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] != syllable:
            break
        else:
            del lst[i]
    return lst

class PyMixBackend:
    """
    Specialized G2P backend, only for Chinese using pypinyin and custom lexicon.
    """
    def __init__(
        self, punctuation_marks: Union[str, re.Pattern] = '，。；：！？、'
    ) -> None:
        self.punctuation_marks = punctuation_marks
        
    def phonemize(self, text: List[str], separator: Separator) -> List[str]:
        """
        Converts Chinese text to a sequence of phonemes (pinyin initials and finals).
        """
        assert isinstance(text, List)
        phonemized_list = []
        for _text in text:
            
            # --- Text Preprocessing (Simplified) ---
            _text = re.sub(r'[",!\.\?\-—…“”‘’\s]', '，', _text) # Standardize common non-chinese-style punctuation to '，'
            _text = re.sub(r'[，。；：！？、\s]+', '，', _text).strip() # Consolidate and strip 
            _text = _text.replace("嗯", "恩").replace("呣", "母") # Pypinyin fixes

            phones = []
            
            # --- Segmentation and Pinyin Conversion ---
            sent_parts = split_sentence(_text)
            for sent_part, _ in sent_parts:
                tmp_phones = []
                sent_part = sent_part.strip() 
                
                # Jieba POS-tagging for better word segmentation (crucial for accurate pinyin)
                seg_cut = psg.lcut(sent_part)
                
                for word, _ in seg_cut:
                    # Convert word to pinyin initials and finals (tone 3 style, neutral tone as '5')
                    sub_initials = [p[0] for p in pinyin(word, neutral_tone_with_five=True, strict=True, style=Style.INITIALS)]
                    sub_finals = [p[0] for p in pinyin(word, neutral_tone_with_five=True, strict=True, style=Style.FINALS_TONE3)] 
                    
                    for shengmu, yunmu in zip(sub_initials, sub_finals):
                        py_ = shengmu + yunmu
                        
                        if all([c in self.punctuation_marks for c in py_]): 
                            if len(tmp_phones) and tmp_phones[-1] == separator.syllable:
                                # Remove preceding syllable separator before adding punctuation
                                tmp_phones.pop(-1)
                            tmp_phones.extend(list(py_[0])) 
                            continue # Move to next word/pinyin
                        else:
                            # Standard pinyin from pypinyin (The core G2P output)
                            new_shengmu, new_yunmu = shengmu, yunmu

                        # 3. Special vowel mapping for TTS models (i -> iii / ii)
                        # Handle 'i' after retroflex initials (zh, ch, sh, r) -> iii
                        # e.g., "chi1" -> "ch iii1"
                        if new_yunmu.startswith("i") and new_shengmu in {"zh", "ch", "sh", "r"}:
                            new_yunmu = "iii" + new_yunmu[1:]
                        
                        # Handle 'i' after dental sibilants (z, c, s) -> ii
                        # e.g., "zi3" -> "z ii3"
                        # Note: Comment out the elif block below if your model does not distinguish 'ii'
                        elif new_yunmu.startswith("i") and new_shengmu in {"z", "c", "s"}:
                            new_yunmu = "ii" + new_yunmu[1:]
                        
                        # 4. Append phonemes in TTS format: initial | final -
                        if new_shengmu:
                            tmp_phones.extend([new_shengmu, separator.phone, new_yunmu, separator.syllable])
                        else:
                            tmp_phones.extend([new_yunmu, separator.syllable])
                            
                tmp_phones = remove_endsyllable(tmp_phones, separator.syllable) 
                phones.extend(tmp_phones)

            # Final cleanup
            phonemized = [p for p in phones if (p != separator.phone and p != '')] 
            phonemized = remove_endsyllable(phonemized, separator.syllable)
            phonemized_list.append(phonemized)
            
        return phonemized_list[0]


class G2P_zh:
    """
    Minimal wrapper for the G2P backend, needed to match the required call signature.
    """
    def __init__(
        self,
        separator=Separator(word="_", syllable="-", phone="|"),
        punctuation_marks: Union[str, re.Pattern] = '，。；：！？、',
    ) -> None:
        phonemizer = PyMixBackend(punctuation_marks=punctuation_marks)
        self.backend = phonemizer
        self.separator = separator

    def __call__(self, text, strip=True) -> List[List[str]]:
        """The main call method."""
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(text, separator=self.separator)
        
        return [phonemized] # Returns [[ph1, ph2, ...]]

def process_one(text: str, tokenizer: G2P_zh):
    """
    REQUIRED INTERFACE: Converts text to phoneme strings.
    """
    phonemes = tokenizer([text.strip()])[0]
    for ph in phonemes:
        # uar5 -> uar1 is a specific rule from the original codebase, preserved for compatibility.
        if ph == "uar5":
            ph = "uar1"
    return phonemes

# --- Example Usage (Main) ---
if __name__ == "__main__":
    text = "我非常地爱吃人参片。我不爱参数。"
    glm_G2P = G2P_zh()
    phonemes = process_one(text, glm_G2P)
    print(f"Input Text: {text}")
    print(f"Phonemes (shengmu|yunmu): {phonemes}")