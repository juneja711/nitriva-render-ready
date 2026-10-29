"""
Nitriva NLP Contextual Sentence Generator Engine.
Translates selected keywords (from eye-gaze / wink selection) into fluent,
context-aware natural language sentences for Text-to-Speech playback.
"""

from typing import List, Dict

# Single keyword standard contextual templates
SINGLE_KEYWORD_MAP: Dict[str, str] = {
    "Water": "I am thirsty. Please give me some water to drink.",
    "Pain": "I am in pain and feeling uncomfortable. Please help me.",
    "Washroom": "I need assistance to use the washroom.",
    "Family": "I would like to call and see my family.",
    "Hungry": "I am feeling hungry. Could I please have something to eat?",
    "Cold": "I am feeling cold. Please provide a warm blanket or adjust the temperature.",
    "Doctor": "Please call the doctor or nurse immediately.",
    "CLEAR / BACK": "Selection cleared."
}

# Combinatorial keyword map (sorted tuples of lowercase keywords)
KEYWORD_PAIR_MAP: Dict[tuple, str] = {
    ("pain", "water"): "I am in pain and urgently need a sip of water.",
    ("doctor", "pain"): "I am experiencing severe pain, please call the doctor immediately.",
    ("cold", "pain"): "I am shivering with cold and feeling sharp pain. Please help.",
    ("hungry", "pain"): "I am in pain and need something light to eat or medication.",
    ("family", "pain"): "I am in significant pain and need my family beside me.",
    ("pain", "washroom"): "I am in pain and urgently need help going to the washroom.",

    ("doctor", "water"): "Please ask the doctor if I am allowed to drink water.",
    ("cold", "water"): "Please bring me some warm water to drink.",
    ("hungry", "water"): "I need food and water, please.",
    ("family", "water"): "Can someone in my family please bring me some water?",
    ("washroom", "water"): "I drank water and now need assistance to go to the washroom.",

    ("doctor", "washroom"): "Please have the medical attendant help me to the washroom.",
    ("family", "washroom"): "Could my family member please assist me to the washroom?",
    ("cold", "washroom"): "It is very cold in the room and I need to use the washroom.",
    ("hungry", "washroom"): "Please help me use the washroom before I eat.",

    ("doctor", "family"): "Please call both my family and the attending doctor.",
    ("cold", "family"): "I am feeling cold and would like my family to help me with blankets.",
    ("family", "hungry"): "Could my family please bring me my preferred meal?",

    ("cold", "doctor"): "I have intense chills, please have the doctor examine me.",
    ("doctor", "hungry"): "Please ask the doctor if I can have my scheduled meal now.",
    ("cold", "hungry"): "I am cold and hungry, please bring me warm food."
}


class NLPEngine:
    def __init__(self):
        self.single_map = SINGLE_KEYWORD_MAP
        self.pair_map = KEYWORD_PAIR_MAP

    def generate_sentence(self, keywords: List[str]) -> str:
        """
        Generate a grammatically sound, contextually accurate sentence
        from 1 or 2 selected keywords.
        """
        if not keywords:
            return "No keywords selected."

        clean_keywords = [k.strip() for k in keywords if k.strip() and k.strip().upper() != "CLEAR / BACK"]

        if not clean_keywords:
            return "Selection cleared."

        if len(clean_keywords) == 1:
            kw = clean_keywords[0]
            # Match directly or case-insensitive
            for k, v in self.single_map.items():
                if k.lower() == kw.lower():
                    return v
            return f"I need assistance regarding {kw}."

        # Two or more keywords
        kw1, kw2 = clean_keywords[0].lower(), clean_keywords[1].lower()
        key_pair = tuple(sorted([kw1, kw2]))

        if key_pair in self.pair_map:
            return self.pair_map[key_pair]

        # Natural fallback composition
        return (
            f"I have an urgent request regarding {clean_keywords[0]} "
            f"and {clean_keywords[1]}. Please assist me."
        )


nlp_engine = NLPEngine()
