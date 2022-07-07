from colorama import init

init(autoreset=True)


class VideoRecord(object):
    def __init__(self, i, row, num_segments):
        self._data = row
        self._index = i
        self._seg = num_segments

    @property
    def segment_id(self):
        if 'narration_id' in self._data:
            return self._data.narration_id
        narration_id = self._data['video_id'] + '_' + str(self._data['uid'])
        return narration_id

    @property
    def path(self):
        return self._index

    @property
    def num_frames(self):
        return int(self._seg)  # self._data[1])

    @property
    def label(self):
        if "verb_class" in self._data:
            return int(self._data.verb_class)
        else:
            return 0

