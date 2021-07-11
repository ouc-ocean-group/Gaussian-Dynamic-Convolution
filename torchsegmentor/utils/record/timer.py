import time


class Timer(object):
    def __init__(self):
        self.start_time = 0
        self.iter_length = 0

    def start(self, iter_length):
        self.iter_length = iter_length
        self.start_time = time.time()

    def stamp(self, step):
        time_duration = time.time() - self.start_time
        rest_time = time_duration / (step + 1) * (self.iter_length - step - 1)
        cur_hour, cur_min, cur_sec = self.convert_format(time_duration)
        rest_hour, rest_min, rest_sec = self.convert_format(rest_time)
        time_string = "[{}:{}:{} < {}:{}:{}]".format(
            cur_hour, cur_min, cur_sec, rest_hour, rest_min, rest_sec
        )
        return time_string

    @staticmethod
    def convert_format(sec):
        hour = "{:02}".format(int(sec // 3600))
        minu = "{:02}".format(int((sec % 3600) // 60))
        sec = "{:02}".format(int(sec % 60))
        return hour, minu, sec
