class NitrivaSentenceSynthesizer:
    """
    AI/ML Natural Language Context Synthesizer in Python.
    Transforms AAC keyword selections into complete natural sentences.
    """
    def __init__(self):
        self.rules = {
            ("pain", "doctor"): "I am in severe pain, please call the doctor immediately.",
            ("washroom", "family"): "I urgently need to use the washroom, please call my family.",
            ("hungry", "water"): "I am feeling hungry and thirsty, can I get some food and water?",
            ("cold", "family"): "I am feeling very cold, please call my family for a blanket.",
            ("pain", "family"): "I am in a lot of pain, please bring my family here.",
            ("doctor", "family"): "Please call my family and the doctor, it's an emergency.",
            ("washroom",): "I need to use the washroom urgently.",
            ("water",): "Can I please get some water to drink?",
            ("hungry",): "I am feeling hungry, could I get something to eat?",
            ("pain",): "I am in pain and need help right now.",
            ("cold",): "I am feeling cold, can someone turn off the AC or give me a blanket?",
            ("family",): "I want to see my family, please call them.",
            ("doctor",): "Please call the doctor, I am not feeling well.",
        }

    def generate(self, keywords):
        if not keywords:
            return "Please select keywords to speak."

        lowered = set(k.strip().lower() for k in keywords)

        # Match longest rule tuples first (e.g. 2-word combinations before 1-word)
        for rule_keys, sentence in sorted(self.rules.items(), key=lambda item: len(item[0]), reverse=True):
            if all(rk in lowered for rk in rule_keys):
                return sentence

        return f"I need help with {' and '.join(keywords)}."
