from LinearRegression import LinearRegression
from TF_LinearRegression import TensorFlowLinearRegression

def main():
    #Example
    linear_regression = LinearRegression([5, 15, 20, 25, 10, 30, 38], [375, 450, 460, 500, 400, 568, 610])
    print(linear_regression.calculate(35))

    tf_linear_regression = TensorFlowLinearRegression([5, 15, 20, 25, 10, 30, 38], [375, 450, 460, 500, 400, 568, 610])
    print(tf_linear_regression.calculate(35))

if __name__ == '__main__':
    main()
