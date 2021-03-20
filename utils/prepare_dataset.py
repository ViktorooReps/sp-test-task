def delete_blanks(filename):
    lines = []
    with open(filename, "r") as f:
        lines = [line for line in f if len(line) > 1]

    with open(filename, "w") as f:
        f.writelines(lines)

if __name__ == '__main__':
    delete_blanks("conll2003/test.txt")
    delete_blanks("conll2003/train.txt")
    delete_blanks("conll2003/valid.txt")