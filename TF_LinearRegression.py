import tensorflow as tf
import numpy as np
import Utils
from exceptions.DistinctLengthList import DistinctLengthList
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TensorFlowLinearRegression:

    RANGE_TRAINING = 500
    LEARNING_RATE = 0.05

    def __init__(self, variables, results):
        if len(variables) != len(results):
            raise DistinctLengthList()

        self.variables = variables
        self.results = results
        variables = self.normalize(variables)
        results = self.normalize(results)

        incline = tf.Variable(tf.random_uniform([1]))
        intersection = tf.Variable(tf.zeros([1]))
        y = tf.add(tf.multiply(incline, variables), intersection)

        # Crea la funcion de costos (cost_function), para la regresion lineal
        cost_function = tf.reduce_mean(tf.square(y - results))

        # Crea el algoritmo de Gradient Descent
        gradient_descent = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)

        # Minimiza el valor de la funcion de costos, con un radio de entrenamiento pasado al Gradient Descent
        train = gradient_descent.minimize(cost_function)

        # Crea la session de TensorFlow
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for step in range(self.RANGE_TRAINING):
            sess.run(train)

        self.intersection = sess.run(intersection)
        self.incline = sess.run(incline)

    def calculate(self, variable):
        variable = self.normalize_value(variable, self.variables)
        return self.desnormalize_value(self.intersection + (self.incline * variable), self.results)

    def normalize(self, data):
        return (np.array(data) - Utils.average(data)) / Utils.difference(data)

    def normalize_value(self, value, data):
        return (value - Utils.average(data)) / Utils.difference(data)

    def desnormalize_value(self, value, data):
        return (value * Utils.difference(data)) + Utils.average(data)
