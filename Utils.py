import abc


@abc.abstractmethod
def average(data):
    return sum(data) / len(data)

@abc.abstractmethod
def difference(data):
    return max(data) - min(data)
