import copy



class Extract:
    def __init__(self, extractor, extract_id, extract_dict=None):
        self.id = extract_id
        self.extractor = extractor

        if extract_dict is None:
            self.speaker = None
            self.speaker_embedding = None
            self.words = None
            self.start = None
            self.end = None
        else:
            self.speaker = extract_dict["speaker"]
            self.speaker_embedding = extract_dict["spk"]
            self.words = extract_dict["result"]
            self.start = self.words[0]["start"]
            self.end = self.words[-1]["end"]


    def to_dict(self):
        return {"id"                : self.id,
                "speaker"           : self.speaker.name,
                "speaker_embedding" : self.speaker_embedding,
                "words"             : self.words,
                "start"             : self.start,
                "end"               : self.end
               }


    @classmethod
    def from_dict(cls, data, extractor, speakers=None):
        if data["id"] >= extractor.next_id:
            raise ValueError("id is bigger than the extractor next id value")
        extract = cls(extractor, int(data["id"]))
        if speakers is not None:
            extract.speaker = speakers[data["speaker"]]
        extract.speaker_embedding = data["speaker_embedding"]
        extract.words = data["words"]
        extract.start = data["start"]
        extract.end   = data["end"  ]
        return extract


    def split(self, at):
        # create new id
        second_id = self.extractor.next_id
        self.extractor.next_id += 1

        # create second
        second = Extract(self.extractor, second_id)
        second.words = self.words[at:]
        second.start = second.words[0]["start"]
        second.end = self.end

        # ajust this one
        self.words = self.words[:at]
        self.end = self.words[-1]["end"]
        self.speaker_embedding = None
        return second


    def correct_text(self, word_ranges, corrections):
        # TODO sort corrections and ranges according to their start position
        offset = 0
        for word_range, correction in zip(word_ranges, corrections):
            word_range[0] -= offset
            word_range[1] -= offset
            start = self.words[word_range[0]]["start"]
            end = self.words[word_range[1] - 1]["end"]
            for i in range(word_range[0], word_range[1]):
                del self.words[word_range[0]]
            self.words.insert(word_range[0], {"word"  : correction,
                                              "start" : start,
                                              "end"   : end,
                                              })
            offset += word_range[1] - word_range[0] - 1


    def timestamped_text(self):
        words_str = [f"{word["word"]}({word["start"]:.2f}-{word["end"]:.2f})" for
                     word in self.words]
        return " ".join(words_str)


    def text(self):
        words_str = [word["word"] for word in self.words]
        return " ".join(words_str)

    def __str__(self):
        return self.extractor.extract_string(self)




class Extractor:
    def __init__(self):
        self.next_id = 0
        self.string_format = {"timestamp" : False,
                              "word timestamp" : False,
                              "embedding" : False,
                              "id_num" : False }


    def to_dict(self):
        return {"next_id"       : self.next_id,
                "string_format" : self.string_format
               }


    @classmethod
    def from_dict(cls, data):
        extractor = cls()
        extractor.next_id = data["next_id"]
        extractor.string_format = data["string_format"]
        return extractor


    def new_extract(self, extract_dict, time_offset=0.0):
        # offset the time
        if time_offset != 0.0:
            for word in extract_dict["result"]:
                word["start"] += time_offset
                word["end"] += time_offset

        # no speaker yet
        extract_dict["speaker"] = None

        # create it and add a new id
        extract = Extract(self, self.next_id, extract_dict)
        self.next_id += 1
        return extract


    def set_id_format(self, activate):
        self.string_format["id_num"] = activate


    def set_timestamp_format(self, activate):
        self.string_format["timestamp"] = activate


    def set_word_timestamp_format(self, activate):
        self.string_format["word timestamp"] = activate


    def stringify_extract_text(self, extract):
        if self.string_format["word timestamp"]:
            return extract.timestamped_text()
        else:
            return extract.text()


    def stringify_extract_speaker(self, extract):
        if extract.speaker is None:
            emb = extract.speaker_embedding
            if emb is None:
                return "[None]"
            return f"[{emb[0]:2f}, ..., {emb[0]:2f}]"
        else:
            return extract.speaker.name


    def extract_string(self, extract):
        speaker_str = self.stringify_extract_speaker(extract)
        text_str = self.stringify_extract_text(extract)

        id_str = ""
        if self.string_format["id_num"]:
            id_str = f"({extract.id:06d})"

        timestamp_str = ""
        if self.string_format["timestamp"]:
            timestamp_str = f"{{{extract.start:.2f}-{extract.end:.2f}}}"

        return f"{id_str}[{speaker_str}]{timestamp_str} : {text_str}"
