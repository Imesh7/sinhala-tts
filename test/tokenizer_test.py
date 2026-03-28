from sinlib import Tokenizer


if __name__ == "__main__":
    
    text = "ආයුබෝවන්"
    
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    enc = tokenizer("ආයුබෝවන්")
    enc.input_ids
    print(enc.input_ids)
    print("Tokeinzer Test completed!") 