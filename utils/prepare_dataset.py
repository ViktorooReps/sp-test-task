def delete_headers(filename):
    with open(filename, "r") as f:
        file_contents = f.read().replace("-DOCSTART- -X- -X- O\n\n", "").rstrip()

    with open(filename, "w") as f:
        f.write(file_contents)

if __name__ == '__main__':
    delete_headers("conll2003/test.txt")
    delete_headers("conll2003/train.txt")
    delete_headers("conll2003/valid.txt")