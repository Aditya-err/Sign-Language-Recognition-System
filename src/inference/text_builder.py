"""
Text Buffer and Fingerspelling Assembly Manager.
===============================================

Author: Senior Python Developer & Computer Vision Engineer (Mentoring MCA Fresher)
Module: sign_language_v2.src.inference.text_builder

Technical Overview for Interviews:
----------------------------------
Fingerspelling sign recognition identifies isolated alphanumeric gestures ('A'-'Z', '1'-'9').
Converting recognized character streams into natural human communication requires
a text accumulation buffer supporting standard editing semantics:
  - Alphanumeric Character Appending
  - Space Insertion (word separation)
  - Backspace (deletion of previous character)
  - Buffer Clearing

Scope Clarification for Academic Viva / Interviews:
  This system implements Fingerspelling to Text conversion.
  It is NOT full Natural Language Processing (NLP) or Continuous Sign Language Translation
  (which requires gloss-to-sentence translation with non-manual markers, facial expressions,
  and contextual language models). Stating this clearly demonstrates engineering honesty.
"""

from typing import List


class TextBuilder:
    """
    Manages interactive text accumulation and editing for fingerspelling translation.
    """

    def __init__(self, max_length: int = 120) -> None:
        """
        Initializes the text builder.

        Args:
            max_length: Maximum allowed characters in the active sentence buffer.
        """
        self.max_length = max_length
        self.characters: List[str] = []

    def add_character(self, char: str) -> bool:
        """
        Appends a recognized alphanumeric character.

        Args:
            char: Alphanumeric character string (e.g. 'A', '5').

        Returns:
            True if character was appended, False if buffer was full or invalid.
        """
        if not char or len(self.characters) >= self.max_length:
            return False

        # Only accept clean single characters
        clean_char = str(char).strip().upper()[:1]
        if clean_char:
            self.characters.append(clean_char)
            return True
        return False

    def add_space(self) -> bool:
        """
        Appends a space character if the previous character is not already a space.

        Returns:
            True if space was appended, False otherwise.
        """
        if len(self.characters) == 0 or len(self.characters) >= self.max_length:
            return False

        if self.characters[-1] != " ":
            self.characters.append(" ")
            return True
        return False

    def backspace(self) -> bool:
        """
        Deletes the last character in the buffer.

        Returns:
            True if a character was removed, False if buffer was already empty.
        """
        if len(self.characters) > 0:
            self.characters.pop()
            return True
        return False

    def clear(self) -> None:
        """Clears all characters from the sentence buffer."""
        self.characters.clear()

    def get_text(self) -> str:
        """
        Returns the accumulated sentence string.
        """
        return "".join(self.characters)

    def get_full_text(self) -> str:
        """Alias for get_text()."""
        return self.get_text()

    def append_char(self, char: str) -> bool:
        """Alias for add_character()."""
        return self.add_character(char)

    @property
    def current_word(self) -> str:
        """Returns the current word being assembled (after last space)."""
        full = self.get_text()
        if " " in full:
            return full.split(" ")[-1]
        return full

    def is_empty(self) -> bool:
        """Returns True if the text buffer is empty."""
        return len(self.characters) == 0

    def __len__(self) -> int:
        return len(self.characters)

    def __str__(self) -> str:
        return self.get_text()
