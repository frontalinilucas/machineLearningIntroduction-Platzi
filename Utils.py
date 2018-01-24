import abc


@abc.abstractmethod
def average(data):
    return sum(data) / len(data)
