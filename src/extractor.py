class Extractor:
    def __init__(self):
        self.next_id = 0
        
    def new_extract(self, extract):
        # make id
        extract["id"] = self.next_id
        self.next_id += 1

        # speaker things
        extract["speaker"] = None
        extract["speaker_embedding"] = extract["spk"]
        del extract["spk"]

        return extract
