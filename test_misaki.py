from misaki import en
try:
    g2p = en.G2P(trf=False, british=False, fallback=None)
    text = "Hello world."
    phonemes = g2p(text)
    print(f"Text: {text}")
    print(f"Phonemes: {phonemes}")
    print(f"Type: {type(phonemes)}")
    
    # Check coverage against a few symbols from config.json
    # config.json has: ɑ, ɐ, ɒ, æ, etc.
    print("Checking specific symbols:")
    print(f"Analysis: {g2p('analysis')}") # Should contain æ or similar
except Exception as e:
    print(f"Error: {e}")
