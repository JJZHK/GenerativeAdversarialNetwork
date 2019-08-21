import sys
import time, datetime


class Timer(object):
    def __init__(self):
        self.init_time = time.time()
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0
        self.average_time = 0
        self.remain_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def remain(self, iters, max_iters):
        if iters == 0:
            self.remain_time = 0
        else:
            self.remain_time = (time.time() - self.init_time) * (max_iters - iters) / iters

        return str(datetime.timedelta(seconds=int(self.remain_time)))

    def average(self):
        return str("%.3f" % self.average_time)

    def averageTostr(self):
        return str(datetime.timedelta(seconds=int(time.time() - self.start_time))), int(time.time() - self.start_time)
        # if type(self.average_time)==type(1):
        #     h=self.average_time/3600
        #     sUp_h=self.average_time-3600*h
        #     m=sUp_h/60
        #     sUp_m=sUp_h-60*m
        #     s=sUp_m
        #     return ":".join(map(str,(h,m,s)))
        # else:
        #     return "[InModuleError]:itv2time(iItv) invalid argument type"

class ProgressBar():
    def __init__(self, epoch_count, one_batch_count, pattern, **kwargs):
        self.total_count = one_batch_count
        self.current_index = 0
        self.current_epoch = 1
        self.epoch_count = epoch_count
        self.train_timer = Timer()
        self.pattern = pattern
        self.iter = False
        if 'iter' in kwargs:
            self.iter = True
        self.showIndex = 0

    def show(self,currentEpoch, *args):
        self.showIndex += 1
        if self.iter == True and self.showIndex == 1:
            self.current_index = self.epoch_count
        else:
            self.current_index += 1
        if self.showIndex == 1:
            self.train_timer.tic()
        self.current_epoch = currentEpoch
        perCount = int(self.total_count / 100) # 7
        perCount = 1 if perCount == 0 else perCount
        percent = int(self.current_index / perCount)

        if self.total_count % perCount == 0:
            dotcount = int(self.total_count / perCount)
        else:
            dotcount = int(self.total_count / perCount)

        s1 = "\rEpoch:%d / %d [%s%s] %d / %d "%(
            self.current_epoch,
            self.epoch_count if self.iter == False else 1,
            "*"*(int(percent)),
            " "*(dotcount-int(percent)),
            self.current_index,
            self.total_count
        )

        s2 = self.pattern % tuple([float("{:.3f}".format(x)) for x in args])

        s3 = "%s,%s,remain=%s" % (
            s1, s2, self.train_timer.remain(self.current_index, self.total_count))
        sys.stdout.write(s3)
        sys.stdout.flush()
        if self.current_index == self.total_count :
            total_time_str, total_time = self.train_timer.averageTostr()
            self.train_timer.toc()
            s3 = "%s,%s,total=%s" % (
                s1, s2, total_time_str)
            sys.stdout.write(s3)
            sys.stdout.flush()
            self.current_index = 0
            self.showIndex = 0
            print("\r")
            return total_time

        return 0