from models.pipeline import IntentionPipeline

if __name__ == "__main__":
    pipe = IntentionPipeline()
    while True:
        q = input("Enter query (or 'quit'): ")
        if q.lower() == "quit":
            break
        out = pipe.process(q)
        print(out)
