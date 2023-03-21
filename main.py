# import torch


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    v = ('supercali'  # '(' begins logical line,
         'fragilistic'  # indentation is ignored
         'expialidocious')  # until closing ')'
    print(f'Hi, {name}, {v}')
    # Press Ctrl+F8 to toggle the breakpoint.
    # print(torch.__version__)
    # print(torch.backends.cudnn.m.is_available())
    # print(torch.cuda.is_available())
    #
    # input_strings = ['1', '2', 'a', '11']
    # valid_int_strings = [int_s for s in input_strings
    #                      if (int_s := safe_int(s)) is not None]
    # print(valid_int_strings)


def safe_int(s):
    try:
        return int(s)
    except BaseException:
        return None


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
