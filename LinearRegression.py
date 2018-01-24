from concurrent.futures import ThreadPoolExecutor
import Utils
from exceptions.DistinctLengthList import DistinctLengthList


class LinearRegression:

    def __init__(self, variables, results):
        if len(variables) != len(results):
            raise DistinctLengthList()
        count = len(variables)
        thread_pool = ThreadPoolExecutor(max_workers=6)
        first_numerator = thread_pool.submit(self.first_numerator, count, variables, results)
        second_numerator = thread_pool.submit(self.second_numerator, variables, results)
        first_denominator = thread_pool.submit(self.first_denominator, count, variables)
        second_denominator = thread_pool.submit(self.second_denominator, variables)
        variables_average = thread_pool.submit(Utils.average, variables)
        results_average = thread_pool.submit(Utils.average, results)

        self.incline = (first_numerator.result() - second_numerator.result()) / (first_denominator.result() - second_denominator.result())
        self.intersection = results_average.result() - (self.incline * variables_average.result())

    def first_numerator(self, count, variables, results):
        multiply = [variable * result for variable, result in zip(variables, results)]
        return sum(multiply) * count

    def second_numerator(self, variables, results):
        return sum(results) * sum(variables)

    def first_denominator(self, count, variables):
        first_denominator = 0
        for variable in variables:
            first_denominator = first_denominator + pow(variable, 2)
        return first_denominator * count

    def second_denominator(self, variables):
        return pow(sum(variables), 2)

    def calculate(self, variable):
        return self.intersection + (self.incline * variable)
