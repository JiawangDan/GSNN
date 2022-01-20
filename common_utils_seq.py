from datetime import datetime


def dt2ts(dt):
    return datetime.strptime(dt, '%Y%m%d').timestamp()
