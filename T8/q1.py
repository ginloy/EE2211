def main():
    x = 2
    grad = 4 * (x ** 3)
    x -= 0.1 * grad
    print(x)

if __name__ == "__main__":
    main()
