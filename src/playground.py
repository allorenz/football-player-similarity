class Person:
    
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name


def main() -> None:

    p_name = Person('Alex').get_name()
    print(p_name)


if __name__ == '__main__':
    main()