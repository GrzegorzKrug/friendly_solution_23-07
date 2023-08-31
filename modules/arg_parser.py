import argparse

import os


def read_arguments_baseline():
    parser = argparse.ArgumentParser(description='Helper do skryptu')

    # Dodawanie flag binarnych
    # parser.add_argument('-h', '--help', action='help', help='Pomoc')
    parser.add_argument('-eo', '--evalonly', action='store_true', help='Eval without train')
    parser.add_argument('-ne', '--noeval', action='store_true', help="Don't plot eval when training")
    parser.add_argument('-s', '--skip', action='store_true', help='Skip first plot')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot gains')
    # parser.add_argument('-a', '--all', action='store_true', help='Skip first plot')

    # parser.add_argument('-', '--live', action='store_true', help='Start live')

    # Dodawanie parametrów liczbowych
    # parser.add_argument('-n', '--number', type=int, help='Parametr liczbowy')

    # Dodawanie parametru tekstowego
    parser.add_argument('-mt', '--modeltype', type=str.lower, help='Typ: PPO / DQN')
    parser.add_argument('-mn', '--modelnum', type=int, help='Model number')

    parser.add_argument('-t', '--train', type=int, help='Train amount')
    # parser.add_argument('-lr', '--learningrate', type=float, help='Change learning rate hyperparam')
    # parser.add_argument('-bt', '--batch', type=float, help='Change batchsize hyperparam')

    parser.add_argument('-r', '--reward', type=int, help='Wybierz reward function')

    parser.add_argument('-l', '--live', action='store_true', help='Start live. Add -pi & -po arguments.')
    parser.add_argument(
            '-pi', '--pathinput',
            type=os.path.abspath, help='Ścieżka wejściowa do pliku Live'
    )
    parser.add_argument(
            '-po', '--pathoutput',
            type=os.path.abspath, help='Ścieżka wyjściowa do Live'
    )

    args = parser.parse_args()
    # print(parser)
    # print(args)
    # if args.help:
    #     print("HELP:")
    print("====")
    parser.print_help()
    print("====")
    return args


if __name__ == "__main__":
    # Wywołanie funkcji i odczytanie argumentów
    arguments = read_arguments_baseline()
    print(type(arguments))
    print(arguments)
    arg_dict = vars(arguments)
    for key, val in arg_dict.items():
        print(f"{key:<10}: {val}, {type(val)}")

    num = arg_dict.get("modelnum", 5)
    print(num)
    # Odczytane wartości
    # print("Flaga -e:", arguments.eval)
# print("Flaga -s:", arguments.sample)
# print("Flaga -l:", arguments.live)
# print("Parametr -n:", arguments.number)
# print("Parametr -m:", arguments.model)
# print("Parametr -mt:", arguments.modeltype)
# print("Parametr -t:", arguments.train)
