import argparse

from text_recognition.data.vocab import write_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read file with formulas and create vocab file")
    parser.add_argument('--data_path', help='path to the formulas')
    args = parser.parse_args()
    write_vocab(args.data_path)
