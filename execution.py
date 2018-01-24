from LinearRegression import LinearRegression


def main():
    #Example
    linear_regression = LinearRegression([5, 15, 20, 25], [375, 487, 450, 500])
    print(linear_regression.calculate(35))

if __name__ == '__main__':
    main()
